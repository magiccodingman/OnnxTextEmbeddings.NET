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

    /// <summary>Identifies the selected built-in Jasper preset when one is in use.</summary>
    public JasperModelPrecision? JasperPrecision { get; private set; } = JasperModelPrecision.Int8;

    public void UseJasper(JasperModelPrecision precision)
    {
        UseHuggingFace(JasperModelPresets.GetRepository(precision));
        JasperPrecision = precision;
    }

    public void UseHuggingFace(string repositoryId, string revision = "main")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repositoryId);
        SourceKind = ModelSourceKind.HuggingFace;
        RepositoryId = repositoryId;
        Revision = string.IsNullOrWhiteSpace(revision) ? "main" : revision;
        LocalDirectory = null;
        ManifestUri = null;
        JasperPrecision = null;
    }

    public void UseLocalDirectory(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        SourceKind = ModelSourceKind.LocalDirectory;
        LocalDirectory = path;
        ManifestUri = null;
        JasperPrecision = null;
    }

    public void UseHttpManifest(Uri manifestUri)
    {
        ArgumentNullException.ThrowIfNull(manifestUri);
        if (!manifestUri.IsAbsoluteUri)
            throw new ArgumentException("The model manifest URI must be absolute.", nameof(manifestUri));
        SourceKind = ModelSourceKind.HttpManifest;
        ManifestUri = manifestUri;
        LocalDirectory = null;
        JasperPrecision = null;
    }
}

public sealed class ModelCacheOptions
{
    public string? Directory { get; set; }
    public TimeSpan LockRetryDelay { get; set; } = TimeSpan.FromMilliseconds(250);
    public TimeSpan LockedFileDeleteRetryDelay { get; set; } = TimeSpan.FromMilliseconds(200);
    public int LockedFileDeleteRetries { get; set; } = 5;
}

public static class InferenceDefaults
{
    public const int ThreadsPerModel = global::OnnxModelRuntime.OnnxModelRuntimeDefaults.ThreadsPerModel;
    public const int AutomaticConcurrentRequestsPerModelCap = global::OnnxModelRuntime.OnnxModelRuntimeDefaults.AutomaticConcurrentRequestsPerModelCap;
}

internal sealed record ResolvedInferenceOptions(
    int ModelInstanceCount,
    int ThreadsPerModel,
    int ConcurrentRequestsPerModel,
    int QueueCapacity)
{
    public int TotalConcurrentRequests => ModelInstanceCount * ConcurrentRequestsPerModel;
}

public sealed class InferenceOptions
{
    private int _modelInstanceCount = 1;
    private int _threadsPerModel = InferenceDefaults.ThreadsPerModel;
    private int _maximumAutoThreadsPerModel = InferenceDefaults.ThreadsPerModel;

    /// <summary>Number of independent ONNX sessions/model copies kept in memory. Default is one.</summary>
    public int ModelInstanceCount
    {
        get => _modelInstanceCount;
        set => _modelInstanceCount = value;
    }

    /// <summary>ONNX Runtime intra-op threads per model instance. Zero enables hardware-based automatic resolution. Default is 16.</summary>
    public int ThreadsPerModel
    {
        get => _threadsPerModel;
        set => _threadsPerModel = value;
    }

    /// <summary>Maximum thread count used only when <see cref="ThreadsPerModel"/> is zero.</summary>
    public int MaximumAutoThreadsPerModel
    {
        get => _maximumAutoThreadsPerModel;
        set => _maximumAutoThreadsPerModel = value;
    }

    /// <summary>
    /// Simultaneous inference calls allowed per model instance. Zero means automatic: ThreadsPerModel / 2,
    /// capped at eight. Explicit positive values are honored as-is.
    /// </summary>
    public int ConcurrentRequestsPerModel { get; set; }

    public int QueueCapacity { get; set; } = 256;

    [Obsolete("Use ModelInstanceCount. A model instance may now execute multiple requests concurrently.")]
    public int WorkerCount
    {
        get => ModelInstanceCount;
        set => ModelInstanceCount = value;
    }

    [Obsolete("Use ThreadsPerModel.")]
    public int ThreadsPerWorker
    {
        get => ThreadsPerModel;
        set => ThreadsPerModel = value;
    }

    [Obsolete("Use MaximumAutoThreadsPerModel.")]
    public int MaximumAutoThreadsPerWorker
    {
        get => MaximumAutoThreadsPerModel;
        set => MaximumAutoThreadsPerModel = value;
    }

    internal global::OnnxModelRuntime.OnnxModelRuntimeOptions ToRuntimeOptions() => new()
    {
        ModelInstanceCount = ModelInstanceCount,
        ThreadsPerModel = ThreadsPerModel,
        MaximumAutoThreadsPerModel = MaximumAutoThreadsPerModel,
        ConcurrentRequestsPerModel = ConcurrentRequestsPerModel,
        QueueCapacity = QueueCapacity
    };

    internal ResolvedInferenceOptions Resolve(JasperModelPrecision? jasperPrecision = null)
    {
        _ = jasperPrecision;
        var resolved = ToRuntimeOptions().Resolve();
        return new ResolvedInferenceOptions(
            resolved.ModelInstanceCount,
            resolved.ThreadsPerModel,
            resolved.ConcurrentRequestsPerModel,
            resolved.QueueCapacity);
    }
}

public sealed class ChunkingOptions
{
    public int ChunkOverlapTokens { get; set; }
    public bool RepeatHeadingContext { get; set; } = true;
    public bool IncludeChunkText { get; set; } = true;
}

public sealed class VectorOptions
{
    /// <summary>Default vector format returned for document embeddings. Defaults to Float32 for maximum interoperability.</summary>
    public EmbeddingVectorFormat DocumentFormat { get; set; } = EmbeddingVectorFormat.Float32;

    /// <summary>Default vector format returned for query embeddings. Defaults to Float32.</summary>
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
        _ = Inference.ToRuntimeOptions().Resolve();
        if (Chunking.ChunkOverlapTokens < 0)
            throw new ArgumentOutOfRangeException(nameof(Chunking.ChunkOverlapTokens));
        if (Search.MinimumLengthConfidence is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(Search.MinimumLengthConfidence));
        if (Search.SupportWindow <= 0)
            throw new ArgumentOutOfRangeException(nameof(Search.SupportWindow));
    }
}
