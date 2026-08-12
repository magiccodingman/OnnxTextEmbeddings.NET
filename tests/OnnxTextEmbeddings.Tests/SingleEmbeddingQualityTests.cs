namespace OnnxTextEmbeddings.Tests;

public sealed class SingleEmbeddingQualityTests
{
    [Fact]
    public void SemanticCoverageProtectsMinorityTopicsBetterThanLinearMean()
    {
        var embeddings = new List<TextEmbedding>();
        for (var i = 0; i < 8; i++)
            embeddings.Add(Embedding([1f, 0f, 0f], i * 10, 10, 100));
        embeddings.Add(Embedding([0f, 1f, 0f], 80, 10, 100));
        embeddings.Add(Embedding([0f, 0f, 1f], 90, 10, 100));

        var semanticCoverage = embeddings.CombineToSingle().Vector.ToFloat32();
        var linearMean = NormalizedMean(embeddings.Select(x => x.Vector.ToFloat32()).ToArray());

        var dominant = new[] { 1f, 0f, 0f };
        var minority = new[] { 0f, 1f, 0f };
        var semanticMinority = Dot(semanticCoverage, minority);
        var linearMinority = Dot(linearMean, minority);
        var semanticDominant = Dot(semanticCoverage, dominant);

        Assert.True(semanticMinority > linearMinority * 2f);
        Assert.True(semanticDominant > 0.85f);
    }

    [Theory]
    [InlineData(2048)]
    [InlineData(1024)]
    [InlineData(768)]
    [InlineData(512)]
    [InlineData(256)]
    public void RepresentativeReductionTargetsAreDeterministicAndNormalized(int targetDimensions)
    {
        var source = Enumerable.Range(0, 2048)
            .Select(i => (float)(Math.Sin(i * 0.17) + Math.Cos(i * 0.031)))
            .ToArray();
        EmbeddingVectorMath.NormalizeInPlace(source);
        var embedding = Embedding(source, 0, 20, 20);

        var first = new[] { embedding }.CombineToSingle(new SingleEmbeddingOptions { OutputDimensions = targetDimensions });
        var second = new[] { embedding }.CombineToSingle(new SingleEmbeddingOptions { OutputDimensions = targetDimensions });

        Assert.Equal(targetDimensions, first.Vector.Dimensions);
        Assert.Equal(first.Vector.Data, second.Vector.Data);
        var values = first.Vector.ToFloat32();
        var norm = Math.Sqrt(values.Sum(x => x * x));
        Assert.InRange(norm, 0.999, 1.001);
    }

    private static float[] NormalizedMean(IReadOnlyList<float[]> vectors)
    {
        var result = new float[vectors[0].Length];
        foreach (var vector in vectors)
        {
            for (var i = 0; i < result.Length; i++)
                result[i] += vector[i];
        }
        EmbeddingVectorMath.NormalizeInPlace(result);
        return result;
    }

    private static float Dot(IReadOnlyList<float> left, IReadOnlyList<float> right)
    {
        var sum = 0f;
        for (var i = 0; i < left.Count; i++)
            sum += left[i] * right[i];
        return sum;
    }

    private static TextEmbedding Embedding(
        IReadOnlyList<float> values,
        int tokenStart,
        int tokenLength,
        int documentTokenCount)
    {
        var copy = values.ToArray();
        EmbeddingVectorMath.NormalizeInPlace(copy);
        return new TextEmbedding
        {
            Vector = EmbeddingVector.FromFloat32(copy),
            Identity = new EmbeddingIdentity
            {
                ModelId = "test-model",
                SourceRevision = "r1",
                EmbeddingSpaceFingerprint = "quality-space",
                IsNormalized = true
            },
            Source = new EmbeddingSource
            {
                DocumentTokenCount = documentTokenCount,
                CharacterRange = new Utf16TextRange(tokenStart, tokenLength),
                TokenRange = new TokenRange(tokenStart, tokenLength),
                TokenCount = tokenLength,
                TokenCapacity = tokenLength
            },
            Chunk = new EmbeddingChunkInfo
            {
                Index = 0,
                Count = 1,
                BoundaryKind = ChunkBoundaryKind.WholeDocument,
                InputTokenCount = tokenLength
            }
        };
    }
}
