namespace OnnxTextEmbeddings;

public enum SearchRetrievalKind
{
    Semantic = 1,
    Lexical = 2
}

public sealed record SearchFieldWeight(string Name, float Weight = 1f)
{
    public static SearchFieldWeight Create(string name, float weight = 1f) => new(name, weight);
}

public sealed record SearchRetrievalStage
{
    public required string Name { get; init; }
    public required SearchRetrievalKind Kind { get; init; }
    public IReadOnlyList<SearchFieldWeight> Fields { get; init; } = Array.Empty<SearchFieldWeight>();
    public SearchFilter? Filter { get; init; }
    public int? CandidateCount { get; init; }
    public float Weight { get; init; } = 1f;

    public static SearchRetrievalStage Semantic(params SearchFieldWeight[] fields) => Semantic("semantic", fields);
    public static SearchRetrievalStage Semantic(string name, params SearchFieldWeight[] fields) => new()
    {
        Name = name,
        Kind = SearchRetrievalKind.Semantic,
        Fields = fields
    };

    public static SearchRetrievalStage Lexical(params SearchFieldWeight[] fields) => Lexical("lexical", fields);
    public static SearchRetrievalStage Lexical(string name, params SearchFieldWeight[] fields) => new()
    {
        Name = name,
        Kind = SearchRetrievalKind.Lexical,
        Fields = fields
    };

    public SearchRetrievalStage Where(SearchFilter filter) => this with { Filter = SearchFilter.CombineAnd(Filter, filter) };
    public SearchRetrievalStage Candidates(int count) => this with { CandidateCount = count };
    public SearchRetrievalStage WithWeight(float weight) => this with { Weight = weight };
}

public enum SearchFusionKind
{
    ReciprocalRank = 1
}

public sealed record SearchFusionOptions
{
    public SearchFusionKind Kind { get; init; } = SearchFusionKind.ReciprocalRank;
    public int RankConstant { get; init; } = 60;
}

public sealed class SearchQuery
{
    private readonly List<SearchRetrievalStage> stages = new();

    public SearchQuery(string text)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(text);
        Text = text;
    }

    public string Text { get; }
    public int Top { get; private set; } = 10;
    public SearchFilter? Filter { get; private set; }
    public SearchFilter? PostFilter { get; private set; }
    public SearchFusionOptions Fusion { get; private set; } = new();
    public IReadOnlyList<SearchRetrievalStage> Retrievals => stages;

    public static SearchQuery Create(string text) => new(text);

    public SearchQuery Where(SearchFilter filter)
    {
        Filter = SearchFilter.CombineAnd(Filter, filter);
        return this;
    }

    public SearchQuery PostWhere(SearchFilter filter)
    {
        PostFilter = SearchFilter.CombineAnd(PostFilter, filter);
        return this;
    }

    public SearchQuery Add(SearchRetrievalStage stage)
    {
        ArgumentNullException.ThrowIfNull(stage);
        stages.Add(stage);
        return this;
    }

    public SearchQuery Semantic(params SearchFieldWeight[] fields) => Add(SearchRetrievalStage.Semantic(fields));
    public SearchQuery Semantic(string name, params SearchFieldWeight[] fields) => Add(SearchRetrievalStage.Semantic(name, fields));
    public SearchQuery Lexical(params SearchFieldWeight[] fields) => Add(SearchRetrievalStage.Lexical(fields));
    public SearchQuery Lexical(string name, params SearchFieldWeight[] fields) => Add(SearchRetrievalStage.Lexical(name, fields));

    public SearchQuery Take(int top)
    {
        if (top <= 0) throw new ArgumentOutOfRangeException(nameof(top));
        Top = top;
        return this;
    }

    public SearchQuery UseReciprocalRankFusion(int rankConstant = 60)
    {
        if (rankConstant < 0) throw new ArgumentOutOfRangeException(nameof(rankConstant));
        Fusion = new SearchFusionOptions { Kind = SearchFusionKind.ReciprocalRank, RankConstant = rankConstant };
        return this;
    }

    public void Validate()
    {
        if (stages.Count == 0)
            throw new InvalidOperationException("A SearchQuery requires at least one retrieval stage.");
        if (stages.Select(stage => stage.Name).Distinct(StringComparer.Ordinal).Count() != stages.Count)
            throw new InvalidOperationException("Search retrieval stage names must be unique.");
        foreach (var stage in stages)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(stage.Name);
            if (stage.Weight < 0 || !float.IsFinite(stage.Weight)) throw new ArgumentOutOfRangeException(nameof(stage.Weight));
            if (stage.CandidateCount is <= 0) throw new ArgumentOutOfRangeException(nameof(stage.CandidateCount));
            if (stage.Fields.Select(field => field.Name).Distinct(StringComparer.Ordinal).Count() != stage.Fields.Count)
                throw new InvalidOperationException($"Search retrieval stage '{stage.Name}' contains duplicate field names.");
            foreach (var field in stage.Fields)
            {
                ArgumentException.ThrowIfNullOrWhiteSpace(field.Name);
                if (field.Weight < 0 || !float.IsFinite(field.Weight)) throw new ArgumentOutOfRangeException(nameof(field.Weight));
            }
        }
        if (Fusion.RankConstant < 0) throw new ArgumentOutOfRangeException(nameof(Fusion.RankConstant));
    }

    public int ResolveCandidateCount(SearchRetrievalStage stage)
    {
        ArgumentNullException.ThrowIfNull(stage);
        if (stage.CandidateCount is { } explicitCount)
            return explicitCount;
        return (int)Math.Min(int.MaxValue, Math.Max(100L, (long)Top * 10L));
    }
}

public sealed class SearchDocument
{
    private readonly Dictionary<string, object?> values = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string> textFields = new(StringComparer.Ordinal);
    private readonly Dictionary<string, IReadOnlyList<TextEmbedding>> semanticFields = new(StringComparer.Ordinal);

    public IReadOnlyDictionary<string, object?> Values => values;
    public IReadOnlyDictionary<string, string> TextFields => textFields;
    public IReadOnlyDictionary<string, IReadOnlyList<TextEmbedding>> SemanticFields => semanticFields;

    public SearchDocument Value(string name, object? value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        values[name] = value;
        return this;
    }

    public SearchDocument Text(string name, string? text)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        textFields[name] = text ?? string.Empty;
        return this;
    }

    public SearchDocument Semantic(string name, IReadOnlyList<TextEmbedding> embeddings)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(embeddings);
        semanticFields[name] = embeddings;
        return this;
    }
}

public sealed record SearchStageCandidate<TKey> where TKey : notnull
{
    public required TKey Item { get; init; }
    public required float RawScore { get; init; }
    public SemanticChunkMatch? BestSemanticMatch { get; init; }
    public IReadOnlyList<SemanticFieldMatch>? SemanticFields { get; init; }
    public IReadOnlyList<LexicalFieldMatch>? LexicalFields { get; init; }
}

public sealed record SearchStageRanking<TKey> where TKey : notnull
{
    public required string StageName { get; init; }
    public required SearchRetrievalKind Kind { get; init; }
    public required float Weight { get; init; }
    public required IReadOnlyList<SearchStageCandidate<TKey>> Candidates { get; init; }
}

public sealed record SearchStageContribution
{
    public required string StageName { get; init; }
    public required SearchRetrievalKind Kind { get; init; }
    public required int Rank { get; init; }
    public required float RawScore { get; init; }
    public required float FusionContribution { get; init; }
    public SemanticChunkMatch? BestSemanticMatch { get; init; }
    public IReadOnlyList<SemanticFieldMatch>? SemanticFields { get; init; }
    public IReadOnlyList<LexicalFieldMatch>? LexicalFields { get; init; }
}

public sealed record SearchResult<T>
{
    public required T Item { get; init; }
    public required float Score { get; init; }
    public required IReadOnlyList<SearchStageContribution> Contributions { get; init; }
}

public static class SearchRankFusion
{
    public static IReadOnlyList<SearchResult<TKey>> Fuse<TKey>(
        IReadOnlyList<SearchStageRanking<TKey>> rankings,
        SearchFusionOptions? options = null)
        where TKey : notnull
    {
        ArgumentNullException.ThrowIfNull(rankings);
        options ??= new SearchFusionOptions();
        if (options.Kind != SearchFusionKind.ReciprocalRank)
            throw new NotSupportedException($"Unsupported search fusion kind '{options.Kind}'.");
        if (options.RankConstant < 0)
            throw new ArgumentOutOfRangeException(nameof(options.RankConstant));

        var aggregates = new Dictionary<TKey, Aggregate>();
        var firstSeen = 0;
        foreach (var ranking in rankings)
        {
            if (ranking.Weight < 0 || !float.IsFinite(ranking.Weight))
                throw new ArgumentOutOfRangeException(nameof(ranking.Weight));
            var seenInStage = new HashSet<TKey>();
            for (var index = 0; index < ranking.Candidates.Count; index++)
            {
                var candidate = ranking.Candidates[index];
                if (!seenInStage.Add(candidate.Item))
                    continue;
                if (!aggregates.TryGetValue(candidate.Item, out var aggregate))
                {
                    aggregate = new Aggregate(firstSeen++);
                    aggregates.Add(candidate.Item, aggregate);
                }

                var rank = index + 1;
                var contribution = ranking.Weight / (options.RankConstant + rank);
                aggregate.Score += contribution;
                aggregate.Contributions.Add(new SearchStageContribution
                {
                    StageName = ranking.StageName,
                    Kind = ranking.Kind,
                    Rank = rank,
                    RawScore = candidate.RawScore,
                    FusionContribution = contribution,
                    BestSemanticMatch = candidate.BestSemanticMatch,
                    SemanticFields = candidate.SemanticFields,
                    LexicalFields = candidate.LexicalFields
                });
            }
        }

        return aggregates
            .Select(pair => new { pair.Key, pair.Value })
            .OrderByDescending(pair => pair.Value.Score)
            .ThenBy(pair => pair.Value.FirstSeen)
            .Select(pair => new SearchResult<TKey>
            {
                Item = pair.Key,
                Score = pair.Value.Score,
                Contributions = pair.Value.Contributions.OrderByDescending(item => item.FusionContribution).ToArray()
            })
            .ToArray();
    }

    private sealed class Aggregate(int firstSeen)
    {
        public int FirstSeen { get; } = firstSeen;
        public float Score { get; set; }
        public List<SearchStageContribution> Contributions { get; } = new();
    }
}

public interface IAdvancedSearch
{
    Task<IReadOnlyList<SearchResult<T>>> SearchAsync<T>(
        SearchQuery query,
        IEnumerable<T> items,
        Func<T, SearchDocument> document,
        CancellationToken cancellationToken = default)
        where T : notnull;
}

internal sealed class AdvancedSearchService(
    ITextEmbeddingService embeddingService,
    ISemanticSearch semanticSearch,
    ILexicalSearch lexicalSearch) : IAdvancedSearch
{
    public async Task<IReadOnlyList<SearchResult<T>>> SearchAsync<T>(
        SearchQuery query,
        IEnumerable<T> items,
        Func<T, SearchDocument> document,
        CancellationToken cancellationToken = default)
        where T : notnull
    {
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(items);
        ArgumentNullException.ThrowIfNull(document);
        query.Validate();

        var prepared = items.Select((item, index) => new PreparedItem<T>(index, item, document(item)))
            .Where(item => SearchFilterEvaluator.Matches(query.Filter, item.Document.Values))
            .ToArray();
        if (prepared.Length == 0)
            return Array.Empty<SearchResult<T>>();

        QueryEmbedding? queryEmbedding = null;
        var rankings = new List<SearchStageRanking<int>>(query.Retrievals.Count);
        foreach (var stage in query.Retrievals)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var eligible = prepared.Where(item => SearchFilterEvaluator.Matches(stage.Filter, item.Document.Values)).ToArray();
            if (eligible.Length == 0 || stage.Weight == 0)
                continue;
            var candidateCount = query.ResolveCandidateCount(stage);

            if (stage.Kind == SearchRetrievalKind.Semantic)
            {
                queryEmbedding ??= await embeddingService.EmbedQueryAsync(query.Text, cancellationToken).ConfigureAwait(false);
                var ranked = await semanticSearch.SearchFieldsAsync(
                    queryEmbedding,
                    eligible,
                    item => CreateSemanticFields(item.Document, stage.Fields),
                    new SemanticSearchRequest { Top = candidateCount },
                    cancellationToken).ConfigureAwait(false);
                rankings.Add(new SearchStageRanking<int>
                {
                    StageName = stage.Name,
                    Kind = stage.Kind,
                    Weight = stage.Weight,
                    Candidates = ranked.Select(result => new SearchStageCandidate<int>
                    {
                        Item = result.Item.Index,
                        RawScore = result.Score,
                        BestSemanticMatch = result.BestMatch,
                        SemanticFields = result.Fields
                    }).ToArray()
                });
            }
            else if (stage.Kind == SearchRetrievalKind.Lexical)
            {
                var ranked = await lexicalSearch.SearchAsync(
                    query.Text,
                    eligible,
                    item => CreateLexicalFields(item.Document, stage.Fields),
                    new LexicalSearchRequest { Top = candidateCount },
                    cancellationToken).ConfigureAwait(false);
                rankings.Add(new SearchStageRanking<int>
                {
                    StageName = stage.Name,
                    Kind = stage.Kind,
                    Weight = stage.Weight,
                    Candidates = ranked.Select(result => new SearchStageCandidate<int>
                    {
                        Item = result.Item.Index,
                        RawScore = result.Score,
                        LexicalFields = result.Fields
                    }).ToArray()
                });
            }
            else
                throw new NotSupportedException($"Unsupported retrieval kind '{stage.Kind}'.");
        }

        var byIndex = prepared.ToDictionary(item => item.Index);
        return SearchRankFusion.Fuse(rankings, query.Fusion)
            .Where(result => byIndex.TryGetValue(result.Item, out var item) && SearchFilterEvaluator.Matches(query.PostFilter, item.Document.Values))
            .Take(query.Top)
            .Select(result => new SearchResult<T>
            {
                Item = byIndex[result.Item].Item,
                Score = result.Score,
                Contributions = result.Contributions
            })
            .ToArray();
    }

    private static IReadOnlyList<SemanticField> CreateSemanticFields(SearchDocument document, IReadOnlyList<SearchFieldWeight> selected)
    {
        if (selected.Count == 0)
            return document.SemanticFields.Select(pair => SemanticField.Create(pair.Key, pair.Value)).ToArray();
        return selected
            .Where(field => field.Weight > 0 && document.SemanticFields.ContainsKey(field.Name))
            .Select(field => SemanticField.Create(field.Name, document.SemanticFields[field.Name], field.Weight))
            .ToArray();
    }

    private static IReadOnlyList<LexicalField> CreateLexicalFields(SearchDocument document, IReadOnlyList<SearchFieldWeight> selected)
    {
        if (selected.Count == 0)
            return document.TextFields.Select(pair => LexicalField.Create(pair.Key, pair.Value)).ToArray();
        return selected
            .Where(field => field.Weight > 0 && document.TextFields.ContainsKey(field.Name))
            .Select(field => LexicalField.Create(field.Name, document.TextFields[field.Name], field.Weight))
            .ToArray();
    }

    private sealed record PreparedItem<T>(int Index, T Item, SearchDocument Document) where T : notnull;
}
