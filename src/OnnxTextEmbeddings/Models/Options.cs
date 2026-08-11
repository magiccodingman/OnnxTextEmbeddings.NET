namespace OnnxTextEmbeddings;

public enum JasperModelPrecision
{
    Int8 = 1,
    Int4 = 2,
    Float32 = 3
}

public enum ModelSourceKind
{
    HuggingFace = 1,
    LocalDirectory = 2,
    HttpManifest = 3
}

public enum ModelUpdatePolicy
{
    OnStartup = 1,
    Manual = 2,
    Never = 3
}

public static class JasperModelPresets
{
    public const string Int8Repository = "magiccodingman/Jasper-Token-Compression-600M-ONNX-INT8";
    public const string Int4Repository = "magiccodingman/Jasper-Token-Compression-600M-ONNX-INT4";
    public const string Float32Repository = "magiccodingman/Jasper-Token-Compression-600M-ONNX-FP32";

    public static string GetRepository(JasperModelPrecision precision) => precision switch
    {
        JasperModelPrecision.Int8 => Int8Repository,
        JasperModelPrecision.Int4 => Int4Repository,
        JasperModelPrecision.Float32 => Float32Repository,
        _ => throw new ArgumentOutOfRangeException(nameof(precision))
    };
}

public sealed class ModelOptions
{
    public ModelSourceKind SourceKind { get; private set; } = ModelSourceKind.HuggingFace;
    public string RepositoryId { get; private set; } = JasperModelPresets.Int8Repository;
    public string Revision { get; set; } = "main";
    public string? AccessToken { get; set; }
    public string? LocalDirectory { get; private set; }
    public Uri? ManifestUri { get; private set; }
    public string? ModelFile { get; set; }
    public ModelUpdatePolicy UpdatePolicy { get; set; } = ModelUpdatePolicy.OnStartup;

    public void UseJasper(JasperModelPrecision precision)
    {
        UseHuggingFace(JasperModelPresets.GetRepository(precision));
    }

    public void UseHuggingFace(string repositoryId, string revision = "main")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repositoryId);
        SourceKind = ModelSourceKind.HuggingFace;
        RepositoryId = repositoryId;
        Revision = string.IsNullOrWhiteSpace(revision) ? "main" : revision;
        LocalDirectory = null;
        ManifestUri = null;
    }

    public void UseLocalDirectory(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        SourceKind = ModelSourceKind.LocalDirectory;
        LocalDirectory = path;
        ManifestUri = null;
    }

    public void UseHttpManifest(Uri manifestUri)
    {
        ArgumentNullException.ThrowIfNull(manifestUri);
        if (!manifestUri.IsAbsoluteUri)
            throw new ArgumentException("The model manifest URI must be absolute.", nameof(manifestUri));
        SourceKind = ModelSourceKind.HttpManifest;
        ManifestUri = manifestUri;
        LocalDirectory = null;
    }
}

public sealed class ModelCacheOptions
{
    public string? Directory { get; set; }
    public TimeSpan LockRetryDelay { get; set; } = TimeSpan.FromMilliseconds(250);
    public TimeSpan LockedFileDeleteRetryDelay { get; set; } = TimeSpan.FromMilliseconds(200);
    public int LockedFileDeleteRetries { get; set; } = 5;
}

public sealed class InferenceOptions
{
    public int WorkerCount { get; set; } = 1;
    /// <summary>Zero means automatic.</summary>
    public int ThreadsPerWorker { get; set; }
    public int MaximumAutoThreadsPerWorker { get; set; } = 12;
    public int QueueCapacity { get; set; } = 256;
}

public sealed class ChunkingOptions
{
    public int ChunkOverlapTokens { get; set; }
    public bool RepeatHeadingContext { get; set; } = true;
    public bool IncludeChunkText { get; set; } = true;
}

public sealed class VectorOptions
{
    public EmbeddingVectorFormat DocumentFormat { get; set; } = EmbeddingVectorFormat.Int8;
    public EmbeddingVectorFormat QueryFormat { get; set; } = EmbeddingVectorFormat.Float32;
}

public sealed class SemanticSearchOptions
{
    public string ScoringProfile { get; set; } = SemanticScoringProfiles.DefaultV1;
    public float MinimumLengthConfidence { get; set; } = 0.96f;
    public float SupportWindow { get; set; } = 0.12f;
    public float SecondSupportWeight { get; set; } = 0.25f;
    public float ThirdSupportWeight { get; set; } = 0.10f;
}

public sealed class InitializationOptions
{
    public bool WarmupOnStartup { get; set; } = true;
    public bool BlockHostStartupUntilReady { get; set; }
}

public sealed class OnnxTextEmbeddingsOptions
{
    public int DocumentChunkMaxTokens { get; set; } = 1024;
    public int QueryMaxTokens { get; set; } = 1024;
    public ModelOptions Model { get; } = new();
    public ModelCacheOptions Cache { get; } = new();
    public InferenceOptions Inference { get; } = new();
    public ChunkingOptions Chunking { get; } = new();
    public VectorOptions Vectors { get; } = new();
    public SemanticSearchOptions Search { get; } = new();
    public InitializationOptions Initialization { get; } = new();

    internal void Validate()
    {
        if (DocumentChunkMaxTokens <= 0)
            throw new ArgumentOutOfRangeException(nameof(DocumentChunkMaxTokens));
        if (QueryMaxTokens <= 0)
            throw new ArgumentOutOfRangeException(nameof(QueryMaxTokens));
        if (Inference.WorkerCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(Inference.WorkerCount));
        if (Inference.ThreadsPerWorker < 0)
            throw new ArgumentOutOfRangeException(nameof(Inference.ThreadsPerWorker));
        if (Inference.MaximumAutoThreadsPerWorker <= 0)
            throw new ArgumentOutOfRangeException(nameof(Inference.MaximumAutoThreadsPerWorker));
        if (Inference.QueueCapacity <= 0)
            throw new ArgumentOutOfRangeException(nameof(Inference.QueueCapacity));
        if (Chunking.ChunkOverlapTokens < 0)
            throw new ArgumentOutOfRangeException(nameof(Chunking.ChunkOverlapTokens));
        if (Search.MinimumLengthConfidence is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(Search.MinimumLengthConfidence));
        if (Search.SupportWindow <= 0)
            throw new ArgumentOutOfRangeException(nameof(Search.SupportWindow));
    }
}
