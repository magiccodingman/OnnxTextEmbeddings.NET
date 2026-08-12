namespace OnnxTextEmbeddings;

/// <summary>Per-call overrides for document/text embedding.</summary>
public sealed record EmbeddingRequestOptions
{
    /// <summary>
    /// Maximum finalized model-input tokens for each document chunk. Null uses the configured DocumentChunkMaxTokens.
    /// </summary>
    public int? MaxTokens { get; init; }

    /// <summary>Return vector format for this call. Null uses the configured document-vector format.</summary>
    public EmbeddingVectorFormat? VectorFormat { get; init; }
}

/// <summary>Per-call overrides for a single query embedding.</summary>
public sealed record QueryEmbeddingRequestOptions
{
    /// <summary>
    /// Query acceptance ceiling. Null uses the configured QueryMaxTokens. Queries remain a single embedding and are never chunked.
    /// </summary>
    public int? MaxTokens { get; init; }

    /// <summary>Return vector format for this call. Null uses the configured query-vector format.</summary>
    public EmbeddingVectorFormat? VectorFormat { get; init; }
}
