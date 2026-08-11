using Microsoft.Extensions.Logging;

namespace OnnxTextEmbeddings;

public enum EmbeddingServiceState
{
    Uninitialized = 0,
    ResolvingModel = 1,
    Downloading = 2,
    Validating = 3,
    Loading = 4,
    Ready = 5,
    Faulted = 6,
    Disposed = 7
}

public sealed record ModelRuntimeInfo(
    string ModelId,
    string SourceRevision,
    string EmbeddingSpaceFingerprint,
    int? ModelMaxTokens,
    int? Dimensions,
    int WorkerCount)
{
    public int ModelInstanceCount => WorkerCount;
    public int ThreadsPerModel { get; init; }
    public int ConcurrentRequestsPerModel { get; init; }
    public int TotalConcurrentRequests => ModelInstanceCount * ConcurrentRequestsPerModel;
    public int HealthyModelInstanceCount { get; init; }
    public int RecoveringModelInstanceCount { get; init; }
    public int ActiveRequests { get; init; }
    public IReadOnlyList<ModelInstanceRuntimeInfo> Instances { get; init; } = Array.Empty<ModelInstanceRuntimeInfo>();
}

public sealed record EmbeddingServiceStatus(
    EmbeddingServiceState State,
    string? Message = null,
    Exception? LastError = null);

public interface ITextEmbeddingService : IAsyncDisposable
{
    EmbeddingServiceStatus Status { get; }
    ModelRuntimeInfo? ModelInfo { get; }
    Task WaitUntilReadyAsync(CancellationToken cancellationToken = default);
    Task<bool> UpdateModelAsync(CancellationToken cancellationToken = default);
    Task<int> CountTokensAsync(string text, CancellationToken cancellationToken = default);
    Task<QueryTokenCount> CountQueryTokensAsync(string query, CancellationToken cancellationToken = default);
    Task<IReadOnlyList<TextEmbedding>> EmbedAsync(string text, CancellationToken cancellationToken = default);
    Task<IReadOnlyList<TextEmbedding>> EmbedDocumentAsync(string text, CancellationToken cancellationToken = default);
    Task<QueryEmbedding> EmbedQueryAsync(string query, CancellationToken cancellationToken = default);

    async Task<QueryTokenCount> CountQueryTokensAsync(
        string query,
        QueryEmbeddingRequestOptions requestOptions,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(requestOptions);
        var count = await CountQueryTokensAsync(query, cancellationToken).ConfigureAwait(false);
        if (requestOptions.MaxTokens is null)
            return count;
        if (requestOptions.MaxTokens <= 0)
            throw new ArgumentOutOfRangeException(nameof(requestOptions), "MaxTokens must be greater than zero.");
        return count with { QueryMaxTokens = requestOptions.MaxTokens.Value };
    }

    /// <summary>
    /// Embeds text using the requested return format for this call. Implementations may override this to encode
    /// directly from their native inference output. The default implementation converts the normal returned record.
    /// </summary>
    async Task<IReadOnlyList<TextEmbedding>> EmbedAsync(
        string text,
        EmbeddingVectorFormat format,
        CancellationToken cancellationToken = default)
    {
        var embeddings = await EmbedAsync(text, cancellationToken).ConfigureAwait(false);
        return embeddings.Select(item => item with { Vector = item.Vector.ConvertTo(format) }).ToArray();
    }

    /// <summary>Embeds a document using the requested return format for this call.</summary>
    async Task<IReadOnlyList<TextEmbedding>> EmbedDocumentAsync(
        string text,
        EmbeddingVectorFormat format,
        CancellationToken cancellationToken = default)
    {
        var embeddings = await EmbedDocumentAsync(text, cancellationToken).ConfigureAwait(false);
        return embeddings.Select(item => item with { Vector = item.Vector.ConvertTo(format) }).ToArray();
    }

    /// <summary>Embeds a single query vector using the requested return format for this call.</summary>
    async Task<QueryEmbedding> EmbedQueryAsync(
        string query,
        EmbeddingVectorFormat format,
        CancellationToken cancellationToken = default)
    {
        var embedding = await EmbedQueryAsync(query, cancellationToken).ConfigureAwait(false);
        return embedding with { Vector = embedding.Vector.ConvertTo(format) };
    }

    async Task<IReadOnlyList<TextEmbedding>> EmbedAsync(
        string text,
        EmbeddingRequestOptions requestOptions,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(requestOptions);
        if (requestOptions.MaxTokens is not null)
            throw new NotSupportedException("This ITextEmbeddingService implementation does not support per-call document token limits.");
        return requestOptions.VectorFormat is { } format
            ? await EmbedAsync(text, format, cancellationToken).ConfigureAwait(false)
            : await EmbedAsync(text, cancellationToken).ConfigureAwait(false);
    }

    async Task<IReadOnlyList<TextEmbedding>> EmbedDocumentAsync(
        string text,
        EmbeddingRequestOptions requestOptions,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(requestOptions);
        if (requestOptions.MaxTokens is not null)
            throw new NotSupportedException("This ITextEmbeddingService implementation does not support per-call document token limits.");
        return requestOptions.VectorFormat is { } format
            ? await EmbedDocumentAsync(text, format, cancellationToken).ConfigureAwait(false)
            : await EmbedDocumentAsync(text, cancellationToken).ConfigureAwait(false);
    }

    async Task<QueryEmbedding> EmbedQueryAsync(
        string query,
        QueryEmbeddingRequestOptions requestOptions,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(requestOptions);
        if (requestOptions.MaxTokens is not null)
            throw new NotSupportedException("This ITextEmbeddingService implementation does not support per-call query token limits.");
        return requestOptions.VectorFormat is { } format
            ? await EmbedQueryAsync(query, format, cancellationToken).ConfigureAwait(false)
            : await EmbedQueryAsync(query, cancellationToken).ConfigureAwait(false);
    }
}

internal sealed class TextEmbeddingService(
    OnnxTextEmbeddingsOptions options,
    ModelCacheManager modelCache,
    ILogger<TextEmbeddingService> logger) : ITextEmbeddingService
{
    private readonly SemaphoreSlim _initializationLock = new(1, 1);
    private volatile EmbeddingServiceStatus _status = new(EmbeddingServiceState.Uninitialized);
    private ModelSnapshot? _snapshot;
    private HuggingFaceEmbeddingTokenizer? _tokenizer;
    private StructuredTextChunker? _chunker;
    private InferenceWorkerPool? _workers;
    private ModelRuntimeInfo? _modelInfo;
    private bool _disposed;

    public EmbeddingServiceStatus Status => _status;

    public ModelRuntimeInfo? ModelInfo
    {
        get
        {
            var info = _modelInfo;
            var workers = _workers;
            if (info is null || workers is null)
                return info;
            var instances = workers.GetRuntimeInfo();
            return info with
            {
                HealthyModelInstanceCount = instances.Count(instance => instance.Health == ModelInstanceHealth.Healthy),
                RecoveringModelInstanceCount = instances.Count(instance => instance.Health is ModelInstanceHealth.Draining or ModelInstanceHealth.Recovering or ModelInstanceHealth.Faulted),
                ActiveRequests = instances.Sum(instance => instance.ActiveRequests),
                Instances = instances
            };
        }
    }

    public Task WaitUntilReadyAsync(CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        if (_status.State == EmbeddingServiceState.Ready)
            return Task.CompletedTask;
        return InitializeAsync(forceRemoteCheck: false, cancellationToken);
    }

    public async Task<bool> UpdateModelAsync(CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        await WaitUntilReadyAsync(cancellationToken).ConfigureAwait(false);
        return await InitializeAsync(forceRemoteCheck: true, cancellationToken).ConfigureAwait(false);
    }

    private async Task<bool> InitializeAsync(bool forceRemoteCheck, CancellationToken cancellationToken)
    {
        await _initializationLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        ModelCandidate? candidate = null;
        HuggingFaceEmbeddingTokenizer? candidateTokenizer = null;
        InferenceWorkerPool? candidateWorkers = null;
        try
        {
            if (!forceRemoteCheck && _status.State == EmbeddingServiceState.Ready)
                return false;

            var hadWorkingRuntime = _workers is not null && _tokenizer is not null && _snapshot is not null;
            _status = new EmbeddingServiceStatus(EmbeddingServiceState.ResolvingModel);
            try
            {
                candidate = await modelCache.ResolveCandidateAsync(cancellationToken, forceRemoteCheck).ConfigureAwait(false);
                if (hadWorkingRuntime && !candidate.RequiresPromotion &&
                    _snapshot!.EmbeddingSpaceFingerprint.Equals(candidate.Snapshot.EmbeddingSpaceFingerprint, StringComparison.Ordinal) &&
                    _snapshot.SourceRevision.Equals(candidate.Snapshot.SourceRevision, StringComparison.Ordinal))
                {
                    _status = new EmbeddingServiceStatus(EmbeddingServiceState.Ready);
                    return false;
                }

                ValidateTokenLimits(candidate.Snapshot);
                _status = new EmbeddingServiceStatus(EmbeddingServiceState.Loading);

                candidateTokenizer = new HuggingFaceEmbeddingTokenizer(candidate.Snapshot.TokenizerPath);
                candidateWorkers = new InferenceWorkerPool(
                    candidate.Snapshot.ModelPath,
                    options.Inference,
                    options.Model.JasperPrecision,
                    logger);
                var candidateChunker = new StructuredTextChunker(candidateTokenizer, options);

                await modelCache.PromoteAsync(candidate, cancellationToken).ConfigureAwait(false);

                var previousWorkers = _workers;
                var previousTokenizer = _tokenizer;
                _snapshot = candidate.Snapshot;
                _tokenizer = candidateTokenizer;
                _chunker = candidateChunker;
                _workers = candidateWorkers;
                candidateTokenizer = null;
                candidateWorkers = null;

                _modelInfo = new ModelRuntimeInfo(
                    candidate.Snapshot.ModelId,
                    candidate.Snapshot.SourceRevision,
                    candidate.Snapshot.EmbeddingSpaceFingerprint,
                    candidate.Snapshot.ModelMaxTokens,
                    _workers.EmbeddingDimensions,
                    _workers.ModelInstanceCount)
                {
                    ThreadsPerModel = _workers.ThreadsPerModel,
                    ConcurrentRequestsPerModel = _workers.ConcurrentRequestsPerModel
                };
                _status = new EmbeddingServiceStatus(EmbeddingServiceState.Ready);

                if (previousWorkers is not null)
                {
                    try { await previousWorkers.DisposeAsync().ConfigureAwait(false); }
                    catch (Exception ex) { logger.LogWarning(ex, "Unable to cleanly dispose the previous ONNX runtime after a model swap."); }
                }
                previousTokenizer?.Dispose();

                try
                {
                    await modelCache.CleanupOldSnapshotsAsync(candidate, cancellationToken).ConfigureAwait(false);
                }
                catch (Exception ex)
                {
                    logger.LogWarning(ex, "The new model is active, but an older cached snapshot could not be deleted. It will be retried on a future cleanup.");
                }

                logger.LogInformation(
                    "ONNX text embedding service is ready with model {ModelId} ({Revision}), {ModelInstances} model instance(s), {ThreadsPerModel} threads/model, and {ConcurrentRequestsPerModel} concurrent requests/model.",
                    candidate.Snapshot.ModelId,
                    candidate.Snapshot.SourceRevision,
                    _workers.ModelInstanceCount,
                    _workers.ThreadsPerModel,
                    _workers.ConcurrentRequestsPerModel);
                return candidate.RequiresPromotion || !hadWorkingRuntime;
            }
            catch (Exception ex)
            {
                if (candidateWorkers is not null)
                {
                    try { await candidateWorkers.DisposeAsync().ConfigureAwait(false); }
                    catch (Exception disposeError) { logger.LogDebug(disposeError, "Unable to dispose a failed candidate ONNX runtime."); }
                    candidateWorkers = null;
                }
                candidateTokenizer?.Dispose();
                candidateTokenizer = null;

                if (candidate is { RequiresPromotion: true })
                {
                    try { await modelCache.DiscardAsync(candidate, CancellationToken.None).ConfigureAwait(false); }
                    catch (Exception cleanupError) { logger.LogWarning(cleanupError, "Unable to remove a failed model candidate snapshot."); }
                }

                if (hadWorkingRuntime)
                {
                    _status = new EmbeddingServiceStatus(
                        EmbeddingServiceState.Ready,
                        $"Model update failed; continuing to use the existing runtime. {ex.Message}",
                        ex);
                    logger.LogWarning(ex, "Model update failed. The existing embedding runtime remains active.");
                }
                else
                    _status = new EmbeddingServiceStatus(EmbeddingServiceState.Faulted, ex.Message, ex);
                throw;
            }
        }
        finally
        {
            _initializationLock.Release();
        }
    }

    public async Task<int> CountTokensAsync(string text, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        await WaitUntilReadyAsync(cancellationToken).ConfigureAwait(false);
        return _tokenizer!.CountSourceTokens(text);
    }

    public Task<QueryTokenCount> CountQueryTokensAsync(string query, CancellationToken cancellationToken = default) =>
        CountQueryTokensAsync(query, new QueryEmbeddingRequestOptions(), cancellationToken);

    public async Task<QueryTokenCount> CountQueryTokensAsync(
        string query,
        QueryEmbeddingRequestOptions requestOptions,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(requestOptions);
        await WaitUntilReadyAsync(cancellationToken).ConfigureAwait(false);
        var queryMaxTokens = ResolveQueryMaxTokens(requestOptions);
        return CreateQueryTokenCount(query, queryMaxTokens, out _);
    }

    public Task<IReadOnlyList<TextEmbedding>> EmbedDocumentAsync(
        string text,
        CancellationToken cancellationToken = default) =>
        EmbedAsync(text, new EmbeddingRequestOptions(), cancellationToken);

    public Task<IReadOnlyList<TextEmbedding>> EmbedDocumentAsync(
        string text,
        EmbeddingVectorFormat format,
        CancellationToken cancellationToken = default) =>
        EmbedAsync(text, new EmbeddingRequestOptions { VectorFormat = format }, cancellationToken);

    public Task<IReadOnlyList<TextEmbedding>> EmbedDocumentAsync(
        string text,
        EmbeddingRequestOptions requestOptions,
        CancellationToken cancellationToken = default) =>
        EmbedAsync(text, requestOptions, cancellationToken);

    public Task<IReadOnlyList<TextEmbedding>> EmbedAsync(
        string text,
        CancellationToken cancellationToken = default) =>
        EmbedAsync(text, new EmbeddingRequestOptions(), cancellationToken);

    public Task<IReadOnlyList<TextEmbedding>> EmbedAsync(
        string text,
        EmbeddingVectorFormat format,
        CancellationToken cancellationToken = default) =>
        EmbedAsync(text, new EmbeddingRequestOptions { VectorFormat = format }, cancellationToken);

    public async Task<IReadOnlyList<TextEmbedding>> EmbedAsync(
        string text,
        EmbeddingRequestOptions requestOptions,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(requestOptions);
        await WaitUntilReadyAsync(cancellationToken).ConfigureAwait(false);
        if (text.Length == 0) return Array.Empty<TextEmbedding>();

        var maxTokens = ResolveDocumentMaxTokens(requestOptions);
        var format = requestOptions.VectorFormat ?? options.Vectors.DocumentFormat;
        var snapshot = _snapshot!;
        var chunks = _chunker!.Chunk(text, maxTokens);
        var tasks = chunks.Select(chunk => _workers!.RunAsync(chunk.ModelInput, cancellationToken)).ToArray();
        var vectors = await Task.WhenAll(tasks).ConfigureAwait(false);
        var identity = CreateIdentity(snapshot);
        var result = new TextEmbedding[chunks.Count];

        for (var i = 0; i < chunks.Count; i++)
        {
            var chunk = chunks[i];
            result[i] = new TextEmbedding
            {
                Vector = EmbeddingVector.FromFloat32(vectors[i], format),
                Identity = identity,
                Source = new EmbeddingSource
                {
                    DocumentTokenCount = chunk.DocumentTokenCount,
                    CharacterRange = chunk.CharacterRange,
                    TokenRange = chunk.TokenRange,
                    TokenCount = chunk.SourceTokenCount,
                    TokenCapacity = chunk.SourceTokenCapacity
                },
                Chunk = new EmbeddingChunkInfo
                {
                    Index = i,
                    Count = chunks.Count,
                    BoundaryKind = chunk.BoundaryKind,
                    HeadingPath = chunk.HeadingPath,
                    ContextTokenCount = chunk.ContextTokenCount,
                    ModelPrefixTokenCount = 0,
                    SpecialTokenCount = chunk.SpecialTokenCount,
                    InputTokenCount = chunk.ModelInput.TokenCount
                },
                Text = options.Chunking.IncludeChunkText ? chunk.SourceText : null,
                Context = chunk.Context
            };
        }
        return result;
    }

    public Task<QueryEmbedding> EmbedQueryAsync(
        string query,
        CancellationToken cancellationToken = default) =>
        EmbedQueryAsync(query, new QueryEmbeddingRequestOptions(), cancellationToken);

    public Task<QueryEmbedding> EmbedQueryAsync(
        string query,
        EmbeddingVectorFormat format,
        CancellationToken cancellationToken = default) =>
        EmbedQueryAsync(query, new QueryEmbeddingRequestOptions { VectorFormat = format }, cancellationToken);

    public async Task<QueryEmbedding> EmbedQueryAsync(
        string query,
        QueryEmbeddingRequestOptions requestOptions,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        ArgumentNullException.ThrowIfNull(requestOptions);
        await WaitUntilReadyAsync(cancellationToken).ConfigureAwait(false);
        var queryMaxTokens = ResolveQueryMaxTokens(requestOptions);
        var count = CreateQueryTokenCount(query, queryMaxTokens, out var input);
        if (!count.Fits)
            throw new QueryTokenLimitExceededException(count.SourceTokenCount, count.InputTokenCount, count.QueryMaxTokens, count.ModelMaxTokens);

        var values = await _workers!.RunAsync(input, cancellationToken).ConfigureAwait(false);
        return new QueryEmbedding
        {
            Vector = EmbeddingVector.FromFloat32(values, requestOptions.VectorFormat ?? options.Vectors.QueryFormat),
            Identity = CreateIdentity(_snapshot!),
            SourceTokenCount = count.SourceTokenCount,
            InputTokenCount = count.InputTokenCount
        };
    }

    private int ResolveDocumentMaxTokens(EmbeddingRequestOptions requestOptions)
    {
        var maxTokens = requestOptions.MaxTokens ?? options.DocumentChunkMaxTokens;
        if (maxTokens <= 0)
            throw new ArgumentOutOfRangeException(nameof(requestOptions), "MaxTokens must be greater than zero.");
        if (_snapshot!.ModelMaxTokens is { } hardLimit && maxTokens > hardLimit)
            throw new ArgumentOutOfRangeException(nameof(requestOptions), $"MaxTokens ({maxTokens}) exceeds the model maximum ({hardLimit}).");
        return maxTokens;
    }

    private int ResolveQueryMaxTokens(QueryEmbeddingRequestOptions requestOptions)
    {
        var maxTokens = requestOptions.MaxTokens ?? options.QueryMaxTokens;
        if (maxTokens <= 0)
            throw new ArgumentOutOfRangeException(nameof(requestOptions), "MaxTokens must be greater than zero.");
        return maxTokens;
    }

    private QueryTokenCount CreateQueryTokenCount(string query, int queryMaxTokens, out TokenizedModelInput input)
    {
        var sourceTokens = _tokenizer!.CountSourceTokens(query);
        input = _tokenizer.EncodeModelInput(query);
        return new QueryTokenCount(sourceTokens, input.TokenCount, queryMaxTokens, _snapshot!.ModelMaxTokens);
    }

    private static EmbeddingIdentity CreateIdentity(ModelSnapshot snapshot) => new()
    {
        ModelId = snapshot.ModelId,
        SourceRevision = snapshot.SourceRevision,
        EmbeddingSpaceFingerprint = snapshot.EmbeddingSpaceFingerprint,
        IsNormalized = true
    };

    private void ValidateTokenLimits(ModelSnapshot snapshot)
    {
        if (snapshot.ModelMaxTokens is not { } max) return;
        if (options.DocumentChunkMaxTokens > max)
            throw new ModelValidationException($"DocumentChunkMaxTokens ({options.DocumentChunkMaxTokens}) exceeds model maximum ({max}).");
        if (options.QueryMaxTokens > max)
            throw new ModelValidationException($"QueryMaxTokens ({options.QueryMaxTokens}) exceeds model maximum ({max}).");
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(TextEmbeddingService));
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;
        _status = new EmbeddingServiceStatus(EmbeddingServiceState.Disposed);
        if (_workers is not null)
            await _workers.DisposeAsync().ConfigureAwait(false);
        _tokenizer?.Dispose();
        _initializationLock.Dispose();
    }
}
