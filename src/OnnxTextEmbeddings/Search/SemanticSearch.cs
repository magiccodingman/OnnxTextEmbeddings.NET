namespace OnnxTextEmbeddings;

public static class SemanticScoringProfiles
{
    public const string DefaultV1 = "DefaultV1";
}

public sealed record SemanticField(
    string Name,
    IReadOnlyList<TextEmbedding> Embeddings,
    float Weight = 1f)
{
    public static SemanticField Create(string name, IReadOnlyList<TextEmbedding> embeddings, float weight = 1f) =>
        new(name, embeddings, weight);
}

public sealed class SemanticSearchRequest
{
    public int Top { get; set; } = 10;
    public bool IncludeAllChunkMatches { get; set; }
}

public sealed record SemanticScoringInfo(string ProfileId, int ProfileVersion);

public sealed record SemanticChunkMatch
{
    public required TextEmbedding Embedding { get; init; }
    public required float RawSimilarity { get; init; }
    public required float LengthConfidence { get; init; }
    public required float AdjustedSimilarity { get; init; }
}

public sealed record SemanticFieldMatch
{
    public required string Name { get; init; }
    public required float Weight { get; init; }
    public required float Score { get; init; }
    public required float WeightedScore { get; init; }
    public required IReadOnlyList<SemanticChunkMatch> Matches { get; init; }
}

public sealed record SemanticSearchResult<T>
{
    public required T Item { get; init; }
    public required float Score { get; init; }
    public required SemanticChunkMatch BestMatch { get; init; }
    public required IReadOnlyList<SemanticFieldMatch> Fields { get; init; }
    public required SemanticScoringInfo Scoring { get; init; }
}

public interface ISemanticSearch
{
    Task<IReadOnlyList<SemanticSearchResult<T>>> SearchAsync<T>(
        string query,
        IEnumerable<T> items,
        Func<T, IReadOnlyList<TextEmbedding>> embeddings,
        SemanticSearchRequest? request = null,
        CancellationToken cancellationToken = default);

    Task<IReadOnlyList<SemanticSearchResult<T>>> SearchAsync<T>(
        QueryEmbedding query,
        IEnumerable<T> items,
        Func<T, IReadOnlyList<TextEmbedding>> embeddings,
        SemanticSearchRequest? request = null,
        CancellationToken cancellationToken = default);

    Task<IReadOnlyList<SemanticSearchResult<T>>> SearchFieldsAsync<T>(
        string query,
        IEnumerable<T> items,
        Func<T, IReadOnlyList<SemanticField>> fields,
        SemanticSearchRequest? request = null,
        CancellationToken cancellationToken = default);

    Task<IReadOnlyList<SemanticSearchResult<T>>> SearchFieldsAsync<T>(
        QueryEmbedding query,
        IEnumerable<T> items,
        Func<T, IReadOnlyList<SemanticField>> fields,
        SemanticSearchRequest? request = null,
        CancellationToken cancellationToken = default);
}

internal sealed class SemanticSearchService(
    ITextEmbeddingService embeddingService,
    OnnxTextEmbeddingsOptions options) : ISemanticSearch
{
    public async Task<IReadOnlyList<SemanticSearchResult<T>>> SearchAsync<T>(
        string query,
        IEnumerable<T> items,
        Func<T, IReadOnlyList<TextEmbedding>> embeddings,
        SemanticSearchRequest? request = null,
        CancellationToken cancellationToken = default)
    {
        var queryEmbedding = await embeddingService.EmbedQueryAsync(query, cancellationToken).ConfigureAwait(false);
        return await SearchAsync(queryEmbedding, items, embeddings, request, cancellationToken).ConfigureAwait(false);
    }

    public Task<IReadOnlyList<SemanticSearchResult<T>>> SearchAsync<T>(
        QueryEmbedding query,
        IEnumerable<T> items,
        Func<T, IReadOnlyList<TextEmbedding>> embeddings,
        SemanticSearchRequest? request = null,
        CancellationToken cancellationToken = default) =>
        SearchFieldsAsync(query, items, item => new[] { SemanticField.Create("content", embeddings(item)) }, request, cancellationToken);

    public async Task<IReadOnlyList<SemanticSearchResult<T>>> SearchFieldsAsync<T>(
        string query,
        IEnumerable<T> items,
        Func<T, IReadOnlyList<SemanticField>> fields,
        SemanticSearchRequest? request = null,
        CancellationToken cancellationToken = default)
    {
        var queryEmbedding = await embeddingService.EmbedQueryAsync(query, cancellationToken).ConfigureAwait(false);
        return await SearchFieldsAsync(queryEmbedding, items, fields, request, cancellationToken).ConfigureAwait(false);
    }

    public Task<IReadOnlyList<SemanticSearchResult<T>>> SearchFieldsAsync<T>(
        QueryEmbedding query,
        IEnumerable<T> items,
        Func<T, IReadOnlyList<SemanticField>> fields,
        SemanticSearchRequest? request = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(items);
        ArgumentNullException.ThrowIfNull(fields);
        request ??= new SemanticSearchRequest();
        if (request.Top <= 0)
            throw new ArgumentOutOfRangeException(nameof(request), "Top must be greater than zero.");

        var queue = new PriorityQueue<SemanticSearchResult<T>, float>();
        foreach (var item in items)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var scored = ScoreItem(query, item, fields(item), request.IncludeAllChunkMatches);
            if (scored is null) continue;
            queue.Enqueue(scored, scored.Score);
            if (queue.Count > request.Top)
                queue.Dequeue();
        }

        var results = new List<SemanticSearchResult<T>>(queue.Count);
        while (queue.TryDequeue(out var result, out _))
            results.Add(result);
        results.Sort((a, b) => b.Score.CompareTo(a.Score));
        return Task.FromResult<IReadOnlyList<SemanticSearchResult<T>>>(results);
    }

    private SemanticSearchResult<T>? ScoreItem<T>(
        QueryEmbedding query,
        T item,
        IReadOnlyList<SemanticField> fields,
        bool includeAllMatches)
    {
        var fieldMatches = new List<SemanticFieldMatch>();
        SemanticChunkMatch? bestOverall = null;
        foreach (var field in fields)
        {
            if (field.Weight < 0)
                throw new ArgumentOutOfRangeException(nameof(field.Weight), "Semantic field weight cannot be negative.");
            if (field.Weight == 0 || field.Embeddings.Count == 0)
                continue;

            var matches = field.Embeddings.Select(embedding => ScoreChunk(query, embedding))
                .OrderByDescending(x => x.AdjustedSimilarity)
                .ToArray();
            if (matches.Length == 0) continue;
            var fieldScore = AggregateEvidence(matches.Select(x => x.AdjustedSimilarity).ToArray());
            var weighted = ApplyFieldWeight(fieldScore, field.Weight);
            var returnedMatches = includeAllMatches ? matches : matches.Take(3).ToArray();
            fieldMatches.Add(new SemanticFieldMatch
            {
                Name = field.Name,
                Weight = field.Weight,
                Score = fieldScore,
                WeightedScore = weighted,
                Matches = returnedMatches
            });
            if (bestOverall is null || matches[0].AdjustedSimilarity > bestOverall.AdjustedSimilarity)
                bestOverall = matches[0];
        }

        if (fieldMatches.Count == 0 || bestOverall is null)
            return null;
        var itemScore = AggregateEvidence(fieldMatches.Select(x => x.WeightedScore).OrderByDescending(x => x).ToArray());
        return new SemanticSearchResult<T>
        {
            Item = item,
            Score = itemScore,
            BestMatch = bestOverall,
            Fields = fieldMatches.OrderByDescending(x => x.WeightedScore).ToArray(),
            Scoring = new SemanticScoringInfo(SemanticScoringProfiles.DefaultV1, 1)
        };
    }

    private SemanticChunkMatch ScoreChunk(QueryEmbedding query, TextEmbedding embedding)
    {
        if (!query.Identity.EmbeddingSpaceFingerprint.Equals(embedding.Identity.EmbeddingSpaceFingerprint, StringComparison.Ordinal))
            throw new EmbeddingSpaceMismatchException($"Query embedding space '{query.Identity.EmbeddingSpaceFingerprint}' does not match candidate space '{embedding.Identity.EmbeddingSpaceFingerprint}'.");
        var raw = EmbeddingVectorMath.CosineSimilarity(query.Vector, embedding.Vector);
        var coverage = embedding.Source.TokenCapacity <= 0
            ? 1f
            : Math.Clamp((float)embedding.Source.TokenCount / embedding.Source.TokenCapacity, 0f, 1f);
        var confidence = options.Search.MinimumLengthConfidence +
            (1f - options.Search.MinimumLengthConfidence) * MathF.Sqrt(coverage);
        var adjusted = Math.Max(0f, raw) * confidence;
        return new SemanticChunkMatch
        {
            Embedding = embedding,
            RawSimilarity = raw,
            LengthConfidence = confidence,
            AdjustedSimilarity = adjusted
        };
    }

    private float AggregateEvidence(IReadOnlyList<float> scores)
    {
        if (scores.Count == 0) return 0;
        var best = scores[0];
        var total = best;
        if (scores.Count > 1)
            total += SupportBonus(best, scores[1], options.Search.SecondSupportWeight);
        if (scores.Count > 2)
            total += SupportBonus(best, scores[2], options.Search.ThirdSupportWeight);
        return Math.Min(1f, total);
    }

    private float SupportBonus(float best, float support, float weight)
    {
        var strength = Math.Clamp(1f - ((best - support) / options.Search.SupportWindow), 0f, 1f);
        return (1f - best) * weight * strength * support;
    }

    private static float ApplyFieldWeight(float score, float weight)
    {
        if (weight == 0) return 0;
        return 1f - MathF.Pow(1f - Math.Clamp(score, 0f, 1f), weight);
    }
}
