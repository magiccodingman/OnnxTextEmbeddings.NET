using System.Text.Json;
using ModelArtifacts;

namespace OnnxTextEmbeddings;

internal sealed record ModelSnapshot(
    string ModelId,
    string SourceRevision,
    string EmbeddingSpaceFingerprint,
    string Directory,
    string ModelPath,
    string TokenizerPath,
    int? ModelMaxTokens,
    bool NormalizeOutput);

internal sealed record ModelCandidate(ArtifactCandidate ArtifactCandidate, ModelSnapshot Snapshot)
{
    public bool RequiresPromotion => ArtifactCandidate.RequiresPromotion;
    public bool IsOfflineFallback => ArtifactCandidate.IsOfflineFallback;
}

/// <summary>
/// Embedding-specific policy over ModelArtifacts.NET. The dependency owns acquisition/cache mechanics; this type owns
/// which runtime assets matter and how an acquired artifact snapshot becomes an embedding-model snapshot.
/// </summary>
internal sealed class EmbeddingArtifactManager : IDisposable
{
    private static readonly ArtifactSelection RuntimeSelection = ArtifactSelection.Patterns(
        "onnx-text-embeddings-runtime-v1",
        "*.onnx",
        "*.json",
        "*.txt",
        "*.model",
        "*.data",
        "*.onnx_data");

    private readonly OnnxTextEmbeddingsOptions options;
    private readonly ArtifactManager manager;

    public EmbeddingArtifactManager(
        HttpClient httpClient,
        OnnxTextEmbeddingsOptions options)
    {
        ArgumentNullException.ThrowIfNull(httpClient);
        ArgumentNullException.ThrowIfNull(options);
        this.options = options;
        manager = new ArtifactManager(
            CreateSource(options.Model),
            CreateManagerOptions(options),
            httpClient);
    }

    public async Task<ModelCandidate> ResolveCandidateAsync(
        CancellationToken cancellationToken,
        bool forceRemoteCheck = false)
    {
        try
        {
            var candidate = await manager.ResolveCandidateAsync(forceRemoteCheck, cancellationToken).ConfigureAwait(false);
            return new ModelCandidate(candidate, LoadSnapshot(candidate.Snapshot));
        }
        catch (ArtifactSourceException ex)
        {
            throw new ModelSourceException(ex.Message, ex);
        }
        catch (ArtifactDownloadException ex)
        {
            throw new ModelDownloadException(ex.Message, ex);
        }
        catch (ArtifactException ex)
        {
            throw new ModelDownloadException(ex.Message, ex);
        }
    }

    public async Task PromoteAsync(ModelCandidate candidate, CancellationToken cancellationToken)
    {
        try
        {
            await manager.PromoteAsync(
                candidate.ArtifactCandidate,
                cleanupObsoleteSnapshots: false,
                cancellationToken).ConfigureAwait(false);
        }
        catch (ArtifactException ex)
        {
            throw new ModelDownloadException(ex.Message, ex);
        }
    }

    public async Task DiscardAsync(ModelCandidate candidate, CancellationToken cancellationToken)
    {
        try
        {
            await manager.DiscardAsync(candidate.ArtifactCandidate, cancellationToken).ConfigureAwait(false);
        }
        catch (ArtifactException ex)
        {
            throw new ModelDownloadException(ex.Message, ex);
        }
    }

    public async Task CleanupAsync(CancellationToken cancellationToken)
    {
        try
        {
            await manager.CleanupAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (ArtifactException ex)
        {
            throw new ModelDownloadException(ex.Message, ex);
        }
    }

    private ModelSnapshot LoadSnapshot(ArtifactSnapshot snapshot)
    {
        var directory = snapshot.DirectoryPath;
        var onnxFiles = snapshot.AssetPaths
            .Where(path => path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
            .Select(snapshot.GetAssetPath)
            .ToArray();

        string modelPath;
        if (!string.IsNullOrWhiteSpace(options.Model.ModelFile))
        {
            modelPath = snapshot.GetAssetPath(options.Model.ModelFile);
            if (!File.Exists(modelPath))
                throw new ModelValidationException($"Configured ONNX model '{options.Model.ModelFile}' does not exist in the snapshot.");
        }
        else
        {
            modelPath = onnxFiles.FirstOrDefault(path => Path.GetFileName(path).Equals("model.onnx", StringComparison.OrdinalIgnoreCase))
                ?? (onnxFiles.Length == 1
                    ? onnxFiles[0]
                    : throw new ModelValidationException("The model snapshot contains multiple ONNX files. Configure Model.ModelFile explicitly."));
        }

        var tokenizerRelative = snapshot.AssetPaths.FirstOrDefault(path =>
            Path.GetFileName(path).Equals("tokenizer.json", StringComparison.OrdinalIgnoreCase))
            ?? throw new ModelValidationException("The model snapshot does not contain tokenizer.json.");
        var tokenizerPath = snapshot.GetAssetPath(tokenizerRelative);
        var (manifestModelId, maxTokens, normalize) = ReadModelMetadata(directory);

        // ModelArtifacts.NET deliberately treats this as an artifact fingerprint. OnnxTextEmbeddings preserves its
        // existing embedding-space contract because the selected runtime asset set and fingerprint algorithm are the
        // same ordered path + file SHA-256 calculation used by the original implementation.
        return new ModelSnapshot(
            manifestModelId ?? snapshot.ArtifactSetId,
            snapshot.SourceRevision,
            snapshot.ArtifactFingerprint,
            directory,
            modelPath,
            tokenizerPath,
            maxTokens,
            normalize);
    }

    private static IModelArtifactSource CreateSource(ModelOptions model) => model.SourceKind switch
    {
        ModelSourceKind.HuggingFace => new HuggingFaceArtifactSource(
            model.RepositoryId,
            RuntimeSelection,
            model.Revision,
            model.AccessToken),
        ModelSourceKind.LocalDirectory => new LocalDirectoryArtifactSource(
            model.LocalDirectory ?? throw new ModelSourceException("No local model directory was configured."),
            RuntimeSelection),
        ModelSourceKind.HttpManifest => new HttpManifestArtifactSource(
            model.ManifestUri ?? throw new ModelSourceException("No HTTP model manifest URI was configured.")),
        _ => throw new ModelSourceException($"Unsupported model source: {model.SourceKind}.")
    };

    private static ArtifactManagerOptions CreateManagerOptions(OnnxTextEmbeddingsOptions options)
    {
        var cacheRoot = options.Cache.Directory;
        if (string.IsNullOrWhiteSpace(cacheRoot))
            cacheRoot = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "OnnxTextEmbeddings",
                "models");

        return new ArtifactManagerOptions
        {
            CacheDirectory = cacheRoot,
            UpdatePolicy = options.Model.UpdatePolicy switch
            {
                ModelUpdatePolicy.OnStartup => ArtifactUpdatePolicy.OnStartup,
                ModelUpdatePolicy.Manual => ArtifactUpdatePolicy.Manual,
                ModelUpdatePolicy.Never => ArtifactUpdatePolicy.Never,
                _ => throw new ArgumentOutOfRangeException(nameof(options.Model.UpdatePolicy))
            },
            DownloadRetries = 3,
            LockRetryDelay = options.Cache.LockRetryDelay,
            LockedFileDeleteRetries = options.Cache.LockedFileDeleteRetries,
            LockedFileDeleteRetryDelay = options.Cache.LockedFileDeleteRetryDelay
        };
    }

    private static (string? ModelId, int? MaxTokens, bool Normalize) ReadModelMetadata(string directory)
    {
        var manifest = Path.Combine(directory, "onnx-text-embeddings.json");
        if (File.Exists(manifest))
        {
            using var document = JsonDocument.Parse(File.ReadAllText(manifest));
            var root = document.RootElement;
            string? id = root.TryGetProperty("modelId", out var idElement) ? idElement.GetString() : null;
            int? max = null;
            if (root.TryGetProperty("model", out var model) && model.ValueKind == JsonValueKind.Object)
            {
                if (model.TryGetProperty("maxSequenceLength", out var maxElement) && maxElement.TryGetInt32(out var parsed))
                    max = parsed;
                var normalize = true;
                if (model.TryGetProperty("output", out var output) && output.ValueKind == JsonValueKind.Object &&
                    output.TryGetProperty("normalize", out var normalizeElement) && normalizeElement.ValueKind is JsonValueKind.True or JsonValueKind.False)
                    normalize = normalizeElement.GetBoolean();
                return (id, max, normalize);
            }
            return (id, max, true);
        }

        foreach (var filename in new[] { "tokenizer_config.json", "config.json" })
        {
            var path = Directory.EnumerateFiles(directory, filename, SearchOption.AllDirectories).FirstOrDefault();
            if (path is null)
                continue;
            try
            {
                using var document = JsonDocument.Parse(File.ReadAllText(path));
                foreach (var property in new[] { "model_max_length", "max_position_embeddings", "max_seq_length" })
                {
                    if (document.RootElement.TryGetProperty(property, out var value) &&
                        value.TryGetInt64(out var max) && max is > 0 and <= 1_000_000)
                        return (null, (int)max, true);
                }
            }
            catch (JsonException)
            {
            }
        }

        return (null, null, true);
    }

    public void Dispose() => manager.Dispose();
}
