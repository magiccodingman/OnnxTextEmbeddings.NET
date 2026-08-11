namespace OnnxTextEmbeddings;

public class OnnxTextEmbeddingsException : Exception
{
    public OnnxTextEmbeddingsException(string message) : base(message) { }
    public OnnxTextEmbeddingsException(string message, Exception innerException) : base(message, innerException) { }
}

public sealed class ModelSourceException : OnnxTextEmbeddingsException
{
    public ModelSourceException(string message) : base(message) { }
    public ModelSourceException(string message, Exception innerException) : base(message, innerException) { }
}

public sealed class ModelDownloadException : OnnxTextEmbeddingsException
{
    public ModelDownloadException(string message) : base(message) { }
    public ModelDownloadException(string message, Exception innerException) : base(message, innerException) { }
}

public sealed class ModelValidationException : OnnxTextEmbeddingsException
{
    public ModelValidationException(string message) : base(message) { }
    public ModelValidationException(string message, Exception innerException) : base(message, innerException) { }
}

public sealed class TokenizerCompatibilityException : OnnxTextEmbeddingsException
{
    public TokenizerCompatibilityException(string message) : base(message) { }
    public TokenizerCompatibilityException(string message, Exception innerException) : base(message, innerException) { }
}

public sealed class EmbeddingSpaceMismatchException : OnnxTextEmbeddingsException
{
    public EmbeddingSpaceMismatchException(string message) : base(message) { }
}

public sealed class QueryTokenLimitExceededException : OnnxTextEmbeddingsException
{
    public QueryTokenLimitExceededException(int sourceTokenCount, int inputTokenCount, int queryMaxTokens, int? modelMaxTokens)
        : base($"Query input uses {inputTokenCount} tokens, exceeding the configured query limit of {queryMaxTokens} tokens.")
    {
        SourceTokenCount = sourceTokenCount;
        InputTokenCount = inputTokenCount;
        QueryMaxTokens = queryMaxTokens;
        ModelMaxTokens = modelMaxTokens;
    }

    public int SourceTokenCount { get; }
    public int InputTokenCount { get; }
    public int QueryMaxTokens { get; }
    public int? ModelMaxTokens { get; }
}

public sealed class InferenceException : OnnxTextEmbeddingsException
{
    public InferenceException(string message, Exception innerException) : base(message, innerException) { }
}

public sealed class EmbeddingVectorFormatException : OnnxTextEmbeddingsException
{
    public EmbeddingVectorFormatException(string message) : base(message) { }
}

public sealed class EmbeddingSerializationException : OnnxTextEmbeddingsException
{
    public EmbeddingSerializationException(string message) : base(message) { }
    public EmbeddingSerializationException(string message, Exception innerException) : base(message, innerException) { }
}
