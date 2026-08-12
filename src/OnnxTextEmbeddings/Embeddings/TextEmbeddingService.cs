using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using OnnxModelRuntime;
using EmbeddingRuntime = OnnxModelRuntime.OnnxModelRuntime<OnnxTextEmbeddings.TokenizedModelInput, float[]>;

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

    async Task<IReadOnlyList<TextEmbedding>> EmbedAsync(
        string text,
        EmbeddingVectorFormat format,
        CancellationToken cancellationToken = default)
    {
        var embeddings = await EmbedAsync(text, cancellationToken).ConfigureAwait(false);
        return embeddings.Select(item => item with { Vector = item.Vector.ConvertTo(format) }).ToArray();
    }

    async Task<IReadOnlyList<TextEmbedding>> EmbedDocumentAsync(
        string text,
        EmbeddingVectorFormat format,
        CancellationToken cancellationToken = default)
    {
        var embeddings = await EmbedDocumentAsync(text, cancellationToken).ConfigureAwait(false);
        return embeddings.Select(item => item with { Vector = item.Vector.ConvertTo(format) }).ToArray();
    }

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
    EmbeddingArtifactManager modelArtifacts,
    ILogger<TextEmbeddingService> logger) : ITextEmbeddingService
{
    private readonly SemaphoreSlim _initializationLock = new(1, 1);
    private volatile EmbeddingServiceStatus _status = new(EmbeddingServiceState.Uninitialized);
    private ModelSnapshot? _snapshot;
    private HuggingFaceEmbeddingTokenizer? _tokenizer;
    private StructuredTextChunker? _chunker;
    private EmbeddingRuntime? _runtime;
    private ModelRuntimeInfo? _modelInfo;
    private bool _disposed;

    public EmbeddingServiceStatus Status => _status;

    public ModelRuntimeInfo? ModelInfo
    {
        get
        {
            var info = _modelInfo;
            var runtime = _runtime;
            if (info is null || runtime is null)
                return info;

            var runtimeInfo = runtime.GetRuntimeInfo();
            return info with
            {
                HealthyModelInstanceCount = runtimeInfo.HealthyModelInstanceCount,
                RecoveringModelInstanceCount = runtimeInfo.RecoveringModelInstanceCount,
                ActiveRequests = runtimeInfo.ActiveRequests,
                Instances = runtimeInfo.Instances.Select(MapInstance).ToArray()
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
        EmbeddingRuntime? candidateRuntime = null;
        try
        {
            if (!forceRemoteCheck && _status.State == EmbeddingServiceState.Ready)
                return false;

            var hadWorkingRuntime = _runtime is not null && _tokenizer is not null && _snapshot is not null;
            _status = new EmbeddingServiceStatus(EmbeddingServiceState.ResolvingModel);
            try
            {
                candidate = await modelArtifacts.ResolveCandidateAsync(cancellationToken, forceRemoteCheck).ConfigureAwait(false);
                if (candidate.IsOfflineFallback)
                    logger.LogWarning("Unable to resolve the configured remote model. Continuing with the known-good cached artifact snapshot.");

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
                var candidateExecutor = new EmbeddingOnnxExecutor();
                candidateRuntime = new EmbeddingRuntime(
                    candidate.Snapshot.ModelPath,
                    candidateExecutor,
                    options.Inference.ToRuntimeOptions());
                var candidateChunker = new StructuredTextChunker(candidateTokenizer, options);

                // ModelArtifacts.NET intentionally leaves application validation to us. Exercise the actual tokenizer,
                // ONNX tensor contract, runtime scheduler and pooling path before the candidate is made current.
                _status = new EmbeddingServiceStatus(EmbeddingServiceState.Validating);
                var validationInput = candidateTokenizer.EncodeModelInput("validation");
                _ = await RunInferenceAsync(candidateRuntime, validationInput, cancellationToken).ConfigureAwait(false);

                await modelArtifacts.PromoteAsync(candidate, cancellationToken).ConfigureAwait(false);

                var previousRuntime = _runtime;
                var previousTokenizer = _tokenizer;
                _snapshot = candidate.Snapshot;
                _tokenizer = candidateTokenizer;
                _chunker = candidateChunker;
                _runtime = candidateRuntime;
                candidateTokenizer = null;
                candidateRuntime = null;

                _modelInfo = new ModelRuntimeInfo(
                    candidate.Snapshot.ModelId,
                    candidate.Snapshot.SourceRevision,
                    candidate.Snapshot.EmbeddingSpaceFingerprint,
                    candidate.Snapshot.ModelMaxTokens,
                    candidateExecutor.EmbeddingDimensions,
                    _runtime.ModelInstanceCount)
                {
                    ThreadsPerModel = _runtime.ThreadsPerModel,
                    ConcurrentRequestsPerModel = _runtime.ConcurrentRequestsPerModel
                };
                _status = new EmbeddingServiceStatus(EmbeddingServiceState.Ready);

                if (previousRuntime is not null)
                {
                    try { await previousRuntime.DisposeAsync().ConfigureAwait(false); }
                    catch (Exception ex) { logger.LogWarning(ex, "Unable to cleanly dispose the previous ONNX runtime after a model swap."); }
                }
                previousTokenizer?.Dispose();

                try
                {
                    await modelArtifacts.CleanupAsync(cancellationToken).ConfigureAwait(false);
                }
                catch (Exception ex)
                {
                    logger.LogWarning(ex, "The new model is active, but an older cached artifact snapshot could not be deleted. It will be retried on a future cleanup.");
                }

                logger.LogInformation(
                    "ONNX text embedding service is ready with model {ModelId} ({Revision}), {ModelInstances} model instance(s), {ThreadsPerModel} threads/model, and {ConcurrentRequestsPerModel} concurrent requests/model.",
                    candidate.Snapshot.ModelId,
                    candidate.Snapshot.SourceRevision,
                    _runtime.ModelInstanceCount,
                    _runtime.ThreadsPerModel,
                    _runtime.ConcurrentRequestsPerModel);
                return candidate.RequiresPromotion || !hadWorkingRuntime;
            }
            catch (Exception ex)
            {
                if (candidateRuntime is not null)
                {
                    try { await candidateRuntime.DisposeAsync().ConfigureAwait(false); }
                    catch (Exception disposeError) { logger.LogDebug(disposeError, "Unable to dispose a failed candidate ONNX runtime."); }
                    candidateRuntime = null;
                }
                candidateTokenizer?.Dispose();
                candidateTokenizer = null;

                if (candidate is { RequiresPromotion: true })
                {
                    try { await modelArtifacts.DiscardAsync(candidate, CancellationToken.None).ConfigureAwait(false); }
                    catch (Exception cleanupError) { logger.LogWarning(cleanupError, "Unable to remove a failed model candidate artifact snapshot."); }
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
        var tasks = chunks.Select(chunk => RunInferenceAsync(_runtime!, chunk.ModelInput, cancellationToken)).ToArray();
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

        var values = await RunInferenceAsync(_runtime!, input, cancellationToken).ConfigureAwait(false);
        return new QueryEmbedding
        {
            Vector = EmbeddingVector.FromFloat32(values, requestOptions.VectorFormat ?? options.Vectors.QueryFormat),
            Identity = CreateIdentity(_snapshot!),
            SourceTokenCount = count.SourceTokenCount,
            InputTokenCount = count.InputTokenCount
        };
    }

    private static async Task<float[]> RunInferenceAsync(
        EmbeddingRuntime runtime,
        TokenizedModelInput input,
        CancellationToken cancellationToken)
    {
        try
        {
            return await runtime.RunAsync(input, cancellationToken).ConfigureAwait(false);
        }
        catch (OnnxModelExecutionException ex)
        {
            throw new InferenceException(ex.Message, ex);
        }
        catch (OnnxRuntimeException ex)
        {
            throw new InferenceException("ONNX embedding inference failed.", ex);
        }
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

    private static ModelInstanceRuntimeInfo MapInstance(global::OnnxModelRuntime.ModelInstanceRuntimeInfo instance) => new(
        instance.Index,
        (ModelInstanceHealth)(int)instance.Health,
        instance.ActiveRequests,
        instance.MaxConcurrentRequests,
        instance.Generation,
        instance.TotalRecoveries,
        instance.RecoveryAttempts,
        instance.LastFailure);

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(TextEmbeddingService));
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;
        _status = new EmbeddingServiceStatus(EmbeddingServiceState.Disposed);
        if (_runtime is not null)
            await _runtime.DisposeAsync().ConfigureAwait(false);
        _tokenizer?.Dispose();
        _initializationLock.Dispose();
    }
}
