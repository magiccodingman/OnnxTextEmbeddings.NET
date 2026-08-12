namespace OnnxTextEmbeddings.Tests;

public sealed class DatabaseCandidateSearchTests
{
    private const string Fingerprint = "database-space";

    [Fact]
    public void CandidateCountDefaultsToAtLeastOneHundredAndTenTimesTop()
    {
        Assert.Equal(100, new DatabaseSemanticSearchOptions { Top = 1 }.ResolveCandidateCount());
        Assert.Equal(100, new DatabaseSemanticSearchOptions { Top = 10 }.ResolveCandidateCount());
        Assert.Equal(250, new DatabaseSemanticSearchOptions { Top = 25 }.ResolveCandidateCount());
        Assert.Equal(42, new DatabaseSemanticSearchOptions { Top = 10, CandidateCount = 42 }.ResolveCandidateCount());
    }

    [Fact]
    public void TextEmbeddingReductionPreservesDirectMetadataAndMatchesQuerySpace()
    {
        var values = Enumerable.Range(0, 16).Select(i => (float)Math.Sin(i + 1)).ToArray();
        EmbeddingVectorMath.NormalizeInPlace(values);
        var source = Embedding(values, "doc", 0.95f);
        var query = Query(values);

        var reducedDocument = source.ReduceDimensions(7, EmbeddingVectorFormat.Float16);
        var reducedQuery = query.ReduceDimensions(7, EmbeddingVectorFormat.Float32);

        Assert.Equal(7, reducedDocument.Vector.Dimensions);
        Assert.Equal(EmbeddingVectorFormat.Float16, reducedDocument.Vector.Format);
        Assert.Equal(source.Source, reducedDocument.Source);
        Assert.Equal(source.Chunk, reducedDocument.Chunk);
        Assert.Equal(source.Text, reducedDocument.Text);
        Assert.Equal(source.Context, reducedDocument.Context);
        Assert.Equal(EmbeddingDimensionReductionProfiles.SrhtV1, reducedDocument.DimensionReduction!.ProfileId);
        Assert.Equal(reducedDocument.Identity.EmbeddingSpaceFingerprint, reducedQuery.Identity.EmbeddingSpaceFingerprint);

        var restored = EmbeddingSerializer.DeserializeJson(EmbeddingSerializer.SerializeJson(reducedDocument));
        Assert.Equal(reducedDocument.DimensionReduction, restored.DimensionReduction);
        Assert.Equal(reducedDocument.Identity, restored.Identity);
    }

    [Fact]
    public async Task DatabaseCandidateRerankingUsesCanonicalDefaultV1EvidenceScoring()
    {
        var service = new SemanticSearchService(new NoopEmbeddingService(), new OnnxTextEmbeddingsOptions());
        var query = Query([1f, 0f]);
        var candidates = new SemanticCandidateBatch<string>
        {
            Candidates =
            [
                Candidate("supported", Embedding([0.91f, MathF.Sqrt(1 - 0.91f * 0.91f)], "supported", 0.91f)),
                Candidate("supported", Embedding([0.90f, MathF.Sqrt(1 - 0.90f * 0.90f)], "supported", 0.90f)),
                Candidate("isolated", Embedding([0.94f, MathF.Sqrt(1 - 0.94f * 0.94f)], "isolated", 0.94f))
            ],
            Retrieval = new SemanticCandidateRetrievalInfo
            {
                Provider = "test-db",
                Mode = "Exact",
                RequestedCandidateCount = 100,
                ReturnedCandidateCount = 3,
                Approximate = false
            }
        };

        var reranked = await service.RerankAsync(
            query,
            candidates,
            new DatabaseSemanticSearchOptions { Top = 2, CandidateCount = 100 },
            TestContext.Current.CancellationToken);

        Assert.Equal("isolated", reranked.Results[0].Item);
        var supported = reranked.Results.Single(result => result.Item == "supported");
        Assert.True(supported.Score > supported.BestMatch.AdjustedSimilarity);
        Assert.Equal(SemanticScoringProfiles.DefaultV1, supported.Scoring.ProfileId);
        Assert.Equal("test-db", reranked.Retrieval.Provider);
    }

    private static SemanticCandidate<string> Candidate(string item, TextEmbedding embedding) => new()
    {
        ItemKey = item,
        FieldName = "content",
        FieldWeight = 1f,
        Embedding = embedding
    };

    private static QueryEmbedding Query(IReadOnlyList<float> values)
    {
        var copy = values.ToArray();
        EmbeddingVectorMath.NormalizeInPlace(copy);
        return new QueryEmbedding
        {
            Vector = EmbeddingVector.FromFloat32(copy),
            Identity = Identity(),
            SourceTokenCount = 2,
            InputTokenCount = 2
        };
    }

    private static TextEmbedding Embedding(IReadOnlyList<float> values, string text, float _)
    {
        var copy = values.ToArray();
        EmbeddingVectorMath.NormalizeInPlace(copy);
        return new TextEmbedding
        {
            Vector = EmbeddingVector.FromFloat32(copy),
            Identity = Identity(),
            Source = new EmbeddingSource
            {
                DocumentTokenCount = 100,
                CharacterRange = new Utf16TextRange(0, 10),
                TokenRange = new TokenRange(0, 10),
                TokenCount = 10,
                TokenCapacity = 10
            },
            Chunk = new EmbeddingChunkInfo
            {
                Index = 0,
                Count = 1,
                BoundaryKind = ChunkBoundaryKind.Paragraph,
                InputTokenCount = 10
            },
            Text = text,
            Context = "test"
        };
    }

    private static EmbeddingIdentity Identity() => new()
    {
        ModelId = "test-model",
        SourceRevision = "1",
        EmbeddingSpaceFingerprint = Fingerprint,
        IsNormalized = true
    };

    private sealed class NoopEmbeddingService : ITextEmbeddingService
    {
        public EmbeddingServiceStatus Status => new(EmbeddingServiceState.Ready);
        public ModelRuntimeInfo? ModelInfo => null;
        public Task WaitUntilReadyAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
        public Task<bool> UpdateModelAsync(CancellationToken cancellationToken = default) => Task.FromResult(false);
        public Task<int> CountTokensAsync(string text, CancellationToken cancellationToken = default) => Task.FromResult(0);
        public Task<QueryTokenCount> CountQueryTokensAsync(string query, CancellationToken cancellationToken = default) => Task.FromResult(new QueryTokenCount(0, 0, 1024, 1024));
        public Task<IReadOnlyList<TextEmbedding>> EmbedAsync(string text, CancellationToken cancellationToken = default) => throw new NotSupportedException();
        public Task<IReadOnlyList<TextEmbedding>> EmbedDocumentAsync(string text, CancellationToken cancellationToken = default) => throw new NotSupportedException();
        public Task<QueryEmbedding> EmbedQueryAsync(string query, CancellationToken cancellationToken = default) => throw new NotSupportedException();
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }
}
