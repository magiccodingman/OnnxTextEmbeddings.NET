namespace OnnxTextEmbeddings;

/// <summary>One database-selected direct chunk that remains eligible for canonical DefaultV1 ranking.</summary>
public sealed record SemanticCandidate<TKey> where TKey : notnull
{
    public required TKey ItemKey { get; init; }
    public required string FieldName { get; init; }
    public float FieldWeight { get; init; } = 1f;
    public required TextEmbedding Embedding { get; init; }
    public float? NativeSimilarity { get; init; }
}

/// <summary>Diagnostics describing the database-native preselection stage.</summary>
public sealed record SemanticCandidateRetrievalInfo
{
    public required string Provider { get; init; }
    public required string Mode { get; init; }
    public required int RequestedCandidateCount { get; init; }
    public required int ReturnedCandidateCount { get; init; }
    public required bool Approximate { get; init; }
}

public sealed record SemanticCandidateBatch<TKey> where TKey : notnull
{
    public required IReadOnlyList<SemanticCandidate<TKey>> Candidates { get; init; }
    public required SemanticCandidateRetrievalInfo Retrieval { get; init; }
}

public sealed class DatabaseSemanticSearchOptions
{
    public int Top { get; set; } = 10;
    public int? CandidateCount { get; set; }
    public bool IncludeAllChunkMatches { get; set; }

    public int ResolveCandidateCount()
    {
        if (Top <= 0)
            throw new ArgumentOutOfRangeException(nameof(Top), "Top must be greater than zero.");
        if (CandidateCount is { } explicitCount)
        {
            if (explicitCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(CandidateCount), "CandidateCount must be greater than zero.");
            if (explicitCount < Top)
                throw new ArgumentOutOfRangeException(nameof(CandidateCount), "CandidateCount cannot be smaller than Top.");
            return explicitCount;
        }

        var multiplied = Math.Min(int.MaxValue, (long)Top * 10L);
        return (int)Math.Max(100L, multiplied);
    }
}

public sealed record DatabaseSemanticSearchResult<TKey> where TKey : notnull
{
    public required IReadOnlyList<SemanticSearchResult<TKey>> Results { get; init; }
    public required SemanticCandidateRetrievalInfo Retrieval { get; init; }
}

/// <summary>
/// Applies the one canonical managed scoring implementation to a database-selected chunk candidate set.
/// Database adapters should retrieve candidates; they should not reimplement DefaultV1.
/// </summary>
public interface ISemanticCandidateReranker
{
    Task<DatabaseSemanticSearchResult<TKey>> RerankAsync<TKey>(
        QueryEmbedding query,
        SemanticCandidateBatch<TKey> candidates,
        DatabaseSemanticSearchOptions? options = null,
        CancellationToken cancellationToken = default)
        where TKey : notnull;
}
