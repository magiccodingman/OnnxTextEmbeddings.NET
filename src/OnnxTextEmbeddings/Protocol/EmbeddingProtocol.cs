namespace OnnxTextEmbeddings;

/// <summary>The serialized record schema for document and query embeddings.</summary>
public static class EmbeddingProtocol
{
    public const int SchemaVersion = 1;
    public const int VectorEncodingVersion = 1;
}

public enum EmbeddingVectorFormat
{
    Unspecified = 0,
    Int4 = 1,
    Int8 = 2,
    Float16 = 3,
    Float32 = 4
}

public enum EmbeddingQuantizationScheme
{
    None = 0,
    SymmetricPerVectorInt4V1 = 1,
    SymmetricPerVectorInt8V1 = 2
}

public enum ChunkBoundaryKind
{
    Unspecified = 0,
    WholeDocument = 1,
    MarkdownSection = 2,
    ParagraphGroup = 3,
    Paragraph = 4,
    SentenceGroup = 5,
    Sentence = 6,
    WordGroup = 7,
    TokenWindow = 8
}

public enum EmbeddingPurpose
{
    Generic = 0,
    Document = 1,
    Query = 2
}

public readonly record struct Utf16TextRange(int Start, int Length)
{
    public int End => checked(Start + Length);
}

public readonly record struct TokenRange(int Start, int Length)
{
    public int End => checked(Start + Length);
}

public sealed record EmbeddingQuantizationInfo
{
    public required EmbeddingQuantizationScheme Scheme { get; init; }
    public required float Scale { get; init; }
    public required float InverseIntegerNorm { get; init; }
}

/// <summary>A storage-independent, versioned embedding-vector representation.</summary>
public sealed record EmbeddingVector
{
    public int EncodingVersion { get; init; } = EmbeddingProtocol.VectorEncodingVersion;
    public required EmbeddingVectorFormat Format { get; init; }
    public required int Dimensions { get; init; }
    public required byte[] Data { get; init; }
    public EmbeddingQuantizationInfo? Quantization { get; init; }

    /// <summary>
    /// Converts this vector to another representation. Expanding a lossy INT4, INT8, or Float16 vector to a
    /// higher-precision representation does not restore precision that was already discarded.
    /// </summary>
    public EmbeddingVector ConvertTo(EmbeddingVectorFormat format) =>
        EmbeddingVectorMath.Convert(this, format);

    /// <summary>
    /// Returns this vector as float32 values. For quantized/lower-precision vectors these values are reconstructed
    /// from the stored representation and are not the original pre-quantization float32 values.
    /// </summary>
    public float[] ToFloat32() => EmbeddingVectorMath.ToFloat32(this);

    /// <summary>
    /// Creates an embedding vector from float32 values. With no explicit format, the input is preserved as Float32.
    /// Specify Float16, Int8, or Int4 to intentionally create a smaller lossy representation.
    /// </summary>
    public static EmbeddingVector FromFloat32(
        ReadOnlySpan<float> values,
        EmbeddingVectorFormat format = EmbeddingVectorFormat.Float32) =>
        EmbeddingVectorMath.FromFloat32(values, format);
}

public sealed record EmbeddingIdentity
{
    public required string ModelId { get; init; }
    public required string SourceRevision { get; init; }
    public required string EmbeddingSpaceFingerprint { get; init; }
    public required bool IsNormalized { get; init; }
}

public sealed record EmbeddingSource
{
    public required int DocumentTokenCount { get; init; }
    public required Utf16TextRange CharacterRange { get; init; }
    public required TokenRange TokenRange { get; init; }
    public required int TokenCount { get; init; }
    public required int TokenCapacity { get; init; }
}

public sealed record EmbeddingChunkInfo
{
    public required int Index { get; init; }
    public required int Count { get; init; }
    public required ChunkBoundaryKind BoundaryKind { get; init; }
    public IReadOnlyList<string> HeadingPath { get; init; } = Array.Empty<string>();
    public int ContextTokenCount { get; init; }
    public int ModelPrefixTokenCount { get; init; }
    public int SpecialTokenCount { get; init; }
    public int InputTokenCount { get; init; }
}

/// <summary>A stable, persistable document-embedding record.</summary>
public sealed record TextEmbedding
{
    public int SchemaVersion { get; init; } = EmbeddingProtocol.SchemaVersion;
    public required EmbeddingVector Vector { get; init; }
    public required EmbeddingIdentity Identity { get; init; }
    public required EmbeddingSource Source { get; init; }
    public required EmbeddingChunkInfo Chunk { get; init; }
    public string? Text { get; init; }
    public string? Context { get; init; }
}

/// <summary>A single-vector semantic query. Query embeddings are never chunked implicitly.</summary>
public sealed record QueryEmbedding
{
    public int SchemaVersion { get; init; } = EmbeddingProtocol.SchemaVersion;
    public required EmbeddingVector Vector { get; init; }
    public required EmbeddingIdentity Identity { get; init; }
    public required int SourceTokenCount { get; init; }
    public required int InputTokenCount { get; init; }
}
