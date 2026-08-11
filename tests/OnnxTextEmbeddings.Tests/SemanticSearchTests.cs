namespace OnnxTextEmbeddings.Tests;

public sealed class SemanticSearchTests
{
    private const string Fingerprint = "same-space";
    private readonly QueryEmbedding _query = new()
    {
        Vector = EmbeddingVector.FromFloat32(new[] { 1f, 0f }, EmbeddingVectorFormat.Float32),
        Identity = Identity(),
        SourceTokenCount = 2,
        InputTokenCount = 2
    };

    [Fact]
    public async Task LengthConfidenceCanNarrowlyFavorLongerStrongEvidence()
    {
        var search = CreateSearch();
        var items = new[]
        {
            new Item("short", [Embedding(0.95f, 100, 1000)]),
            new Item("long", [Embedding(0.93f, 800, 1000)])
        };
        var results = await search.SearchAsync(_query, items, x => x.Embeddings);
        Assert.Equal("long", results[0].Item.Name);
    }

    [Fact]
    public async Task TrulyExcellentShortEvidenceStillWins()
    {
        var search = CreateSearch();
        var items = new[]
        {
            new Item("short", [Embedding(0.99f, 100, 1000)]),
            new Item("long", [Embedding(0.93f, 800, 1000)])
        };
        var results = await search.SearchAsync(_query, items, x => x.Embeddings);
        Assert.Equal("short", results[0].Item.Name);
    }

    [Fact]
    public async Task IrrelevantChunksDoNotAverageDownExcellentEvidence()
    {
        var search = CreateSearch();
        var items = new[]
        {
            new Item("large", [Embedding(0.94f), Embedding(0.20f), Embedding(0.18f), Embedding(0.10f)]),
            new Item("small", [Embedding(0.86f)])
        };
        var results = await search.SearchAsync(_query, items, x => x.Embeddings);
        Assert.Equal("large", results[0].Item.Name);
    }

    [Fact]
    public async Task StrongSupportingChunksBoostWithoutOverpoweringHigherBestMatch()
    {
        var search = CreateSearch();
        var items = new[]
        {
            new Item("isolated", [Embedding(0.94f)]),
            new Item("supported", [Embedding(0.91f), Embedding(0.90f), Embedding(0.89f)])
        };
        var results = await search.SearchAsync(_query, items, x => x.Embeddings);
        var supported = results.Single(x => x.Item.Name == "supported");
        Assert.True(supported.Score > supported.BestMatch.AdjustedSimilarity);
        Assert.Equal("isolated", results[0].Item.Name);
    }

    [Fact]
    public async Task ManyWeakChunksCannotBeatOneExcellentMatch()
    {
        var search = CreateSearch();
        var weak = Enumerable.Range(0, 20).Select(_ => Embedding(0.55f)).ToArray();
        var items = new[] { new Item("excellent", [Embedding(0.90f)]), new Item("weak", weak) };
        var results = await search.SearchAsync(_query, items, x => x.Embeddings);
        Assert.Equal("excellent", results[0].Item.Name);
    }

    [Fact]
    public async Task DifferentEmbeddingSpacesAreRejected()
    {
        var search = CreateSearch();
        var bad = Embedding(0.9f) with
        {
            Identity = Identity() with { EmbeddingSpaceFingerprint = "other" }
        };
        await Assert.ThrowsAsync<EmbeddingSpaceMismatchException>(async () =>
            await search.SearchAsync(_query, new[] { new Item("bad", [bad]) }, x => x.Embeddings));
    }

    private static ISemanticSearch CreateSearch()
    {
        var options = new OnnxTextEmbeddingsOptions();
        return new SemanticSearchService(new NoopEmbeddingService(), options);
    }

    private static TextEmbedding Embedding(float cosine, int tokenCount = 1000, int capacity = 1000)
    {
        var y = MathF.Sqrt(Math.Max(0, 1 - cosine * cosine));
        return new TextEmbedding
        {
            Vector = EmbeddingVector.FromFloat32(new[] { cosine, y }, EmbeddingVectorFormat.Float32),
            Identity = Identity(),
            Source = new EmbeddingSource
            {
                DocumentTokenCount = tokenCount,
                CharacterRange = new Utf16TextRange(0, 1),
                TokenRange = new TokenRange(0, tokenCount),
                TokenCount = tokenCount,
                TokenCapacity = capacity
            },
            Chunk = new EmbeddingChunkInfo { Index = 0, Count = 1, BoundaryKind = ChunkBoundaryKind.Paragraph }
        };
    }

    private static EmbeddingIdentity Identity() => new()
    {
        ModelId = "test",
        SourceRevision = "1",
        EmbeddingSpaceFingerprint = Fingerprint,
        IsNormalized = true
    };

    private sealed record Item(string Name, IReadOnlyList<TextEmbedding> Embeddings);

    private sealed class NoopEmbeddingService : ITextEmbeddingService
    {
        public EmbeddingServiceStatus Status => new(EmbeddingServiceState.Ready);
        public ModelRuntimeInfo? ModelInfo => null;
        public Task WaitUntilReadyAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
        public Task<bool> UpdateModelAsync(CancellationToken cancellationToken = default) => Task.FromResult(false);
        public Task<IReadOnlyList<TextEmbedding>> EmbedAsync(string text, CancellationToken cancellationToken = default) => throw new NotSupportedException();
        public Task<IReadOnlyList<TextEmbedding>> EmbedDocumentAsync(string text, CancellationToken cancellationToken = default) => EmbedAsync(text, cancellationToken);
        public Task<QueryEmbedding> EmbedQueryAsync(string query, CancellationToken cancellationToken = default) => throw new NotSupportedException();
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }
}
