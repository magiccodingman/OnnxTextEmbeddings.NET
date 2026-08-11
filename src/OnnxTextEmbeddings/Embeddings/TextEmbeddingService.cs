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
    int WorkerCount);

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
    Task<IReadOnlyList<TextEmbedding>> EmbedAsync(string text, CancellationToken cancellationToken = default);
    Task<IReadOnlyList<TextEmbedding>> EmbedDocumentAsync(string text, CancellationToken cancellationToken = default);
    Task<QueryEmbedding> EmbedQueryAsync(string query, CancellationToken cancellationToken = default);
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
    private bool _disposed;

    public EmbeddingServiceStatus Status => _status;
    public ModelRuntimeInfo? ModelInfo { get; private set; }

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
                candidateWorkers = new InferenceWorkerPool(candidate.Snapshot.ModelPath, options.Inference);
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

                ModelInfo = new ModelRuntimeInfo(
                    candidate.Snapshot.ModelId,
                    candidate.Snapshot.SourceRevision,
                    candidate.Snapshot.EmbeddingSpaceFingerprint,
                    candidate.Snapshot.ModelMaxTokens,
                    _workers.EmbeddingDimensions,
                    options.Inference.WorkerCount);
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
                    "ONNX text embedding service is ready with model {ModelId} ({Revision}).",
                    candidate.Snapshot.ModelId,
                    candidate.Snapshot.SourceRevision);
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
                    try
                    {
                        await modelCache.DiscardAsync(candidate, CancellationToken.None).ConfigureAwait(false);
                    }
                    catch (Exception cleanupError)
                    {
                        logger.LogWarning(cleanupError, "Unable to remove a failed model candidate snapshot.");
                    }
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
                {
                    _status = new EmbeddingServiceStatus(EmbeddingServiceState.Faulted, ex.Message, ex);
                }
                throw;
            }
        }
        finally
        {
            _initializationLock.Release();
        }
    }

    public Task<IReadOnlyList<TextEmbedding>> EmbedDocumentAsync(
        string text,
        CancellationToken cancellationToken = default) =>
        EmbedAsync(text, cancellationToken);

    public async Task<IReadOnlyList<TextEmbedding>> EmbedAsync(string text, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        await WaitUntilReadyAsync(cancellationToken).ConfigureAwait(false);
        if (text.Length == 0) return Array.Empty<TextEmbedding>();
        var snapshot = _snapshot!;
        var chunks = _chunker!.Chunk(text, options.DocumentChunkMaxTokens);
        var tasks = chunks.Select(chunk => _workers!.RunAsync(chunk.ModelInput, cancellationToken)).ToArray();
        var vectors = await Task.WhenAll(tasks).ConfigureAwait(false);
        var identity = CreateIdentity(snapshot);
        var result = new TextEmbedding[chunks.Count];

        for (var i = 0; i < chunks.Count; i++)
        {
            var chunk = chunks[i];
            result[i] = new TextEmbedding
            {
                Vector = EmbeddingVector.FromFloat32(vectors[i], options.Vectors.DocumentFormat),
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

    public async Task<QueryEmbedding> EmbedQueryAsync(string query, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        await WaitUntilReadyAsync(cancellationToken).ConfigureAwait(false);
        var sourceTokens = _tokenizer!.CountSourceTokens(query);
        var input = _tokenizer.EncodeModelInput(query);
        if (input.TokenCount > options.QueryMaxTokens)
            throw new QueryTokenLimitExceededException(sourceTokens, input.TokenCount, options.QueryMaxTokens, _snapshot!.ModelMaxTokens);
        if (_snapshot!.ModelMaxTokens is { } hardLimit && input.TokenCount > hardLimit)
            throw new QueryTokenLimitExceededException(sourceTokens, input.TokenCount, options.QueryMaxTokens, hardLimit);

        var values = await _workers!.RunAsync(input, cancellationToken).ConfigureAwait(false);
        return new QueryEmbedding
        {
            Vector = EmbeddingVector.FromFloat32(values, options.Vectors.QueryFormat),
            Identity = CreateIdentity(_snapshot),
            SourceTokenCount = sourceTokens,
            InputTokenCount = input.TokenCount
        };
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
