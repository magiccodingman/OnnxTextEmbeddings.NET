namespace OnnxTextEmbeddings;

public static class EmbeddingAggregationProfiles
{
    public const string Passthrough = "Passthrough";
    public const string SemanticCoverageV1 = "SemanticCoverage-v1";
}

public static class EmbeddingDimensionReductionProfiles
{
    public const string SrhtV1 = "SRHT-v1";
}

public enum EmbeddingRepresentationKind
{
    Direct = 1,
    Aggregated = 2
}

public enum EmbeddingAggregationStrategy
{
    SemanticCoverage = 1
}

public enum EmbeddingDimensionReductionStrategy
{
    Auto = 0,
    SrhtV1 = 1
}

public enum EmbeddingSourceMassMethod
{
    SourceTokenCount = 1,
    TokenRangeCoverage = 2
}

/// <summary>Per-call options for combining one or more document embeddings into exactly one vector.</summary>
public sealed record SingleEmbeddingOptions
{
    /// <summary>Requested output dimensionality. Null preserves the supplied dimensionality.</summary>
    public int? OutputDimensions { get; init; }

    /// <summary>
    /// Requested output numeric representation. Null preserves a common source format, or uses Float32 for mixed formats.
    /// </summary>
    public EmbeddingVectorFormat? OutputFormat { get; init; }

    public EmbeddingAggregationStrategy AggregationStrategy { get; init; } = EmbeddingAggregationStrategy.SemanticCoverage;
    public EmbeddingDimensionReductionStrategy DimensionReductionStrategy { get; init; } = EmbeddingDimensionReductionStrategy.Auto;

    /// <summary>
    /// Optional embedding-space neutral cosine baseline. Null uses the registered profile value when available,
    /// otherwise the universal fallback of 0.
    /// </summary>
    public float? NeutralSimilarityBaseline { get; init; }
}

public sealed record EmbeddingAggregationInfo
{
    public required string ProfileId { get; init; }
    public required int ProfileVersion { get; init; }
    public required EmbeddingSourceMassMethod SourceMassMethod { get; init; }
    public required float NeutralSimilarityBaseline { get; init; }
    public required float AffinityExponent { get; init; }
    public required float RedundancyExponent { get; init; }
    public required float AggregationCoherence { get; init; }
    public float? MinimumSourceSimilarity { get; init; }
    public required bool FallbackUsed { get; init; }
}

public sealed record EmbeddingDimensionReductionInfo
{
    public required string ProfileId { get; init; }
    public required int ProfileVersion { get; init; }
    public required int SourceDimensions { get; init; }
    public required int OutputDimensions { get; init; }
}

/// <summary>
/// Exactly one semantic embedding representing one direct source embedding or a lossy mathematical aggregation of many.
/// </summary>
public sealed record SingleEmbedding
{
    public int SchemaVersion { get; init; } = EmbeddingProtocol.SchemaVersion;
    public required EmbeddingVector Vector { get; init; }
    public required EmbeddingIdentity Identity { get; init; }
    public required EmbeddingRepresentationKind RepresentationKind { get; init; }
    public required int SourceEmbeddingCount { get; init; }
    public required int SourceTokenCount { get; init; }
    public required int SourceDimensions { get; init; }
    public required EmbeddingAggregationInfo Aggregation { get; init; }
    public EmbeddingDimensionReductionInfo? DimensionReduction { get; init; }
}
