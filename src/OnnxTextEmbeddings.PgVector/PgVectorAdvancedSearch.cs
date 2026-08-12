using Npgsql;

namespace OnnxTextEmbeddings.PgVector;

public sealed record PgVectorSearchPlan
{
    public PgVectorCandidateQuery? Semantic { get; init; }
    public PgVectorLexicalQuery? Lexical { get; init; }
}

/// <summary>Executes composable semantic/lexical SearchQuery plans using PostgreSQL-native retrieval and core RRF.</summary>
public sealed class PgVectorAdvancedSearch(
    ITextEmbeddingService embeddingService,
    PgVectorSemanticSearch semanticSearch,
    PgVectorLexicalSearch lexicalSearch)
{
    public async Task<IReadOnlyList<SearchResult<TKey>>> SearchAsync<TKey>(
        NpgsqlConnection connection,
        SearchQuery query,
        PgVectorSearchPlan plan,
        QueryEmbedding? semanticQuery = null,
        Func<TKey, IReadOnlyDictionary<string, object?>>? postFilterValues = null,
        Action<NpgsqlCommand>? configureProviderParameters = null,
        NpgsqlTransaction? transaction = null,
        CancellationToken cancellationToken = default)
        where TKey : notnull
    {
        ArgumentNullException.ThrowIfNull(connection);
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(plan);
        query.Validate();
        if (query.PostFilter is not null && postFilterValues is null)
            throw new ArgumentException("A database SearchQuery with PostFilter requires postFilterValues so the late filter can be evaluated after fusion.", nameof(postFilterValues));

        QueryEmbedding? resolvedSemanticQuery = semanticQuery;
        var rankings = new List<SearchStageRanking<TKey>>(query.Retrievals.Count);
        foreach (var stage in query.Retrievals)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (stage.Weight == 0)
                continue;
            var stageCount = query.ResolveCandidateCount(stage);
            var stageFilter = SearchFilter.CombineAnd(query.Filter, stage.Filter);

            if (stage.Kind == SearchRetrievalKind.Semantic)
            {
                var configured = plan.Semantic ?? throw new InvalidOperationException("The SearchQuery contains a semantic stage but the PgVectorSearchPlan has no Semantic mapping.");
                resolvedSemanticQuery ??= await embeddingService.EmbedQueryAsync(query.Text, cancellationToken).ConfigureAwait(false);
                var selectedFields = stage.Fields.Count == 0 ? null : stage.Fields.Where(field => field.Weight > 0).Select(field => field.Name).ToArray();
                var fieldWeights = stage.Fields.Count == 0
                    ? null
                    : stage.Fields.Where(field => field.Weight > 0).ToDictionary(field => field.Name, field => field.Weight, StringComparer.Ordinal);
                var semanticConfig = configured with
                {
                    Filter = SearchFilter.CombineAnd(configured.Filter, stageFilter),
                    IncludeFields = selectedFields,
                    QueryFieldWeights = fieldWeights
                };
                var result = await semanticSearch.SearchAsync<TKey>(
                    connection,
                    resolvedSemanticQuery,
                    semanticConfig,
                    new DatabaseSemanticSearchOptions
                    {
                        Top = stageCount,
                        CandidateCount = (int)Math.Min(int.MaxValue, Math.Max(100L, (long)stageCount * 10L))
                    },
                    configureProviderParameters,
                    transaction,
                    cancellationToken).ConfigureAwait(false);
                rankings.Add(new SearchStageRanking<TKey>
                {
                    StageName = stage.Name,
                    Kind = stage.Kind,
                    Weight = stage.Weight,
                    Candidates = result.Results.Select(item => new SearchStageCandidate<TKey>
                    {
                        Item = item.Item,
                        RawScore = item.Score,
                        BestSemanticMatch = item.BestMatch,
                        SemanticFields = item.Fields
                    }).ToArray()
                });
            }
            else if (stage.Kind == SearchRetrievalKind.Lexical)
            {
                var configured = plan.Lexical ?? throw new InvalidOperationException("The SearchQuery contains a lexical stage but the PgVectorSearchPlan has no Lexical mapping.");
                var lexicalConfig = configured with { Filter = SearchFilter.CombineAnd(configured.Filter, stageFilter) };
                var result = await lexicalSearch.SearchAsync<TKey>(
                    connection,
                    query.Text,
                    lexicalConfig,
                    stage.Fields,
                    new DatabaseLexicalSearchOptions { Top = stageCount },
                    configureProviderParameters,
                    transaction,
                    cancellationToken).ConfigureAwait(false);
                rankings.Add(new SearchStageRanking<TKey>
                {
                    StageName = stage.Name,
                    Kind = stage.Kind,
                    Weight = stage.Weight,
                    Candidates = result.Results.Select(item => new SearchStageCandidate<TKey>
                    {
                        Item = item.Item,
                        RawScore = item.Score,
                        LexicalFields = item.Fields
                    }).ToArray()
                });
            }
            else
                throw new NotSupportedException($"Unsupported retrieval kind '{stage.Kind}'.");
        }

        var fused = SearchRankFusion.Fuse(rankings, query.Fusion);
        if (query.PostFilter is not null)
            fused = fused.Where(item => SearchFilterEvaluator.Matches(query.PostFilter, postFilterValues!(item.Item))).ToArray();
        return fused.Take(query.Top).ToArray();
    }
}
