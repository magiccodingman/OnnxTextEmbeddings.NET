namespace OnnxTextEmbeddings.Tests;

public sealed class LexicalAndAdvancedSearchTests
{
    [Fact]
    public async Task Bm25_FieldWeightsMakeExactTitleMatchWin()
    {
        var search = new InMemoryLexicalSearch();
        var items = new[]
        {
            new Item("title", "PostgreSQL Backup", "Routine database operations"),
            new Item("body", "Database Operations", "This document explains PostgreSQL backup PostgreSQL backup PostgreSQL backup")
        };

        var results = await search.SearchAsync(
            "postgresql backup",
            items,
            item =>
            [
                LexicalField.Create("title", item.Title, 8f),
                LexicalField.Create("body", item.Body, 1f)
            ],
            cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal("title", results[0].Item.Id);
        Assert.Equal(LexicalScoringProfiles.Bm25V1, results[0].Scoring.ProfileId);
    }

    [Fact]
    public void PortableFilters_EvaluateAndCompileWithoutInliningValues()
    {
        var filter = SearchFilter.And(
            SearchFilter.Equal("TenantId", 42),
            SearchFilter.In("Category", new[] { "Database", "Infrastructure" }),
            SearchFilter.GreaterThanOrEqual("Updated", 10),
            SearchFilter.Contains("Title", "100%_safe"));
        var values = new Dictionary<string, object?>
        {
            ["TenantId"] = 42,
            ["Category"] = "Database",
            ["Updated"] = 11,
            ["Title"] = "100%_SAFE restore"
        };

        Assert.True(SearchFilterEvaluator.Matches(filter, values));
        var compiled = SearchFilterSqlCompiler.Compile(filter, field => $"\"{field}\"");
        Assert.Contains("@ote_filter_0", compiled.Sql, StringComparison.Ordinal);
        Assert.DoesNotContain("Database", compiled.Sql, StringComparison.Ordinal);
        Assert.DoesNotContain("100%_safe", compiled.Sql, StringComparison.OrdinalIgnoreCase);
        Assert.Contains(compiled.Parameters, parameter => Equals(parameter.Value, 42));
        Assert.Contains(compiled.Parameters, parameter => Equals(parameter.Value, "%100\\%\\_safe%"));
    }

    [Fact]
    public void PortableFilters_CompileNullStableAtomicPredicatesForNotComposition()
    {
        var values = new Dictionary<string, object?>
        {
            ["Status"] = null,
            ["Category"] = null,
            ["Title"] = null
        };

        var notEqual = SearchFilter.NotEqual("Status", "Deleted");
        var notIn = SearchFilter.NotIn("Category", new[] { "Database", "Infrastructure" });
        var negatedEqual = SearchFilter.Not(SearchFilter.Equal("Status", "Deleted"));
        var negatedContains = SearchFilter.Not(SearchFilter.Contains("Title", "backup"));

        Assert.True(SearchFilterEvaluator.Matches(notEqual, values));
        Assert.True(SearchFilterEvaluator.Matches(notIn, values));
        Assert.True(SearchFilterEvaluator.Matches(negatedEqual, values));
        Assert.True(SearchFilterEvaluator.Matches(negatedContains, values));

        var compiledNotEqual = SearchFilterSqlCompiler.Compile(notEqual, field => $"\"{field}\"");
        var compiledNotIn = SearchFilterSqlCompiler.Compile(notIn, field => $"\"{field}\"");
        var compiledNegatedEqual = SearchFilterSqlCompiler.Compile(negatedEqual, field => $"\"{field}\"");
        var compiledNegatedContains = SearchFilterSqlCompiler.Compile(negatedContains, field => $"\"{field}\"");

        Assert.Contains("IS NULL OR", compiledNotEqual.Sql, StringComparison.Ordinal);
        Assert.Contains("IS NULL OR", compiledNotIn.Sql, StringComparison.Ordinal);
        Assert.Contains("IS NOT NULL AND", compiledNegatedEqual.Sql, StringComparison.Ordinal);
        Assert.StartsWith("NOT (", compiledNegatedEqual.Sql, StringComparison.Ordinal);
        Assert.Contains("IS NOT NULL AND", compiledNegatedContains.Sql, StringComparison.Ordinal);
        Assert.StartsWith("NOT (", compiledNegatedContains.Sql, StringComparison.Ordinal);
    }

    [Fact]
    public async Task LexicalOnlyAdvancedQuery_DoesNotRequestAnEmbedding()
    {
        var embedding = new ThrowingEmbeddingService();
        var semantic = new SemanticSearchService(embedding, new OnnxTextEmbeddingsOptions());
        var advanced = new AdvancedSearchService(embedding, semantic, new InMemoryLexicalSearch());
        var items = new[]
        {
            new Item("a", "PostgreSQL Backup", "restore instructions"),
            new Item("b", "Networking", "firewall")
        };
        var query = SearchQuery.Create("postgresql backup")
            .Where(SearchFilter.Equal("Tenant", 7))
            .Lexical(SearchFieldWeight.Create("title", 8), SearchFieldWeight.Create("body", 1))
            .Take(1);

        var results = await advanced.SearchAsync(
            query,
            items,
            item => new SearchDocument()
                .Value("Tenant", 7)
                .Text("title", item.Title)
                .Text("body", item.Body),
            TestContext.Current.CancellationToken);

        Assert.Single(results);
        Assert.Equal("a", results[0].Item.Id);
        Assert.Equal(0, embedding.QueryEmbeddingCalls);
    }

    [Fact]
    public void ReciprocalRankFusionRewardsItemsSupportedByBothRetrievers()
    {
        var semantic = new SearchStageRanking<string>
        {
            StageName = "semantic",
            Kind = SearchRetrievalKind.Semantic,
            Weight = 1,
            Candidates =
            [
                new SearchStageCandidate<string> { Item = "semantic-only", RawScore = .99f },
                new SearchStageCandidate<string> { Item = "both", RawScore = .90f }
            ]
        };
        var lexical = new SearchStageRanking<string>
        {
            StageName = "lexical",
            Kind = SearchRetrievalKind.Lexical,
            Weight = 1,
            Candidates =
            [
                new SearchStageCandidate<string> { Item = "both", RawScore = 8f },
                new SearchStageCandidate<string> { Item = "lexical-only", RawScore = 7f }
            ]
        };

        var fused = SearchRankFusion.Fuse([semantic, lexical]);

        Assert.Equal("both", fused[0].Item);
        Assert.Equal(2, fused[0].Contributions.Count);
    }

    private sealed record Item(string Id, string Title, string Body);

    private sealed class ThrowingEmbeddingService : ITextEmbeddingService
    {
        public int QueryEmbeddingCalls { get; private set; }
        public EmbeddingServiceStatus Status => new(EmbeddingServiceState.Ready);
        public ModelRuntimeInfo? ModelInfo => null;
        public Task WaitUntilReadyAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
        public Task<bool> UpdateModelAsync(CancellationToken cancellationToken = default) => Task.FromResult(false);
        public Task<int> CountTokensAsync(string text, CancellationToken cancellationToken = default) => throw new NotSupportedException();
        public Task<QueryTokenCount> CountQueryTokensAsync(string query, CancellationToken cancellationToken = default) => throw new NotSupportedException();
        public Task<IReadOnlyList<TextEmbedding>> EmbedAsync(string text, CancellationToken cancellationToken = default) => throw new NotSupportedException();
        public Task<IReadOnlyList<TextEmbedding>> EmbedDocumentAsync(string text, CancellationToken cancellationToken = default) => throw new NotSupportedException();
        public Task<QueryEmbedding> EmbedQueryAsync(string query, CancellationToken cancellationToken = default)
        {
            QueryEmbeddingCalls++;
            throw new InvalidOperationException("Lexical-only advanced search must not embed the query.");
        }
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }
}
