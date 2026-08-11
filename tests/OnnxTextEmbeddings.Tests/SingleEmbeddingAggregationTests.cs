namespace OnnxTextEmbeddings.Tests;

public sealed class SingleEmbeddingAggregationTests
{
    [Fact]
    public void SingleInputWithoutTransformIsDirectPassthrough()
    {
        var input = Embedding([1f, 0f, 0f, 0f], 0, 10, 10, EmbeddingVectorFormat.Int8);

        var result = new[] { input }.CombineToSingle();

        Assert.Equal(EmbeddingRepresentationKind.Direct, result.RepresentationKind);
        Assert.Same(input.Vector, result.Vector);
        Assert.Equal(input.Identity, result.Identity);
        Assert.Equal(10, result.SourceTokenCount);
        Assert.Equal(EmbeddingAggregationProfiles.Passthrough, result.Aggregation.ProfileId);
        Assert.Equal(1f, result.Aggregation.AggregationCoherence);
    }

    [Fact]
    public void MixedInputFormatsDefaultToFloat32()
    {
        var inputs = new[]
        {
            Embedding([1f, 0f, 0f, 0f], 0, 10, 20, EmbeddingVectorFormat.Int8),
            Embedding([0.8f, 0.6f, 0f, 0f], 10, 10, 20, EmbeddingVectorFormat.Float16)
        };

        var result = inputs.CombineToSingle();

        Assert.Equal(EmbeddingVectorFormat.Float32, result.Vector.Format);
        Assert.Equal(EmbeddingRepresentationKind.Aggregated, result.RepresentationKind);
        Assert.Equal(EmbeddingAggregationProfiles.SemanticCoverageV1, result.Aggregation.ProfileId);
    }

    [Theory]
    [InlineData(EmbeddingVectorFormat.Int4)]
    [InlineData(EmbeddingVectorFormat.Int8)]
    [InlineData(EmbeddingVectorFormat.Float16)]
    [InlineData(EmbeddingVectorFormat.Float32)]
    public void ExplicitOutputFormatIsHonored(EmbeddingVectorFormat format)
    {
        var inputs = new[]
        {
            Embedding([1f, 0f, 0f, 0f], 0, 10, 20),
            Embedding([0f, 1f, 0f, 0f], 10, 10, 20)
        };

        var result = inputs.CombineToSingle(new SingleEmbeddingOptions { OutputFormat = format });

        Assert.Equal(format, result.Vector.Format);
        Assert.Equal(4, result.Vector.Dimensions);
    }

    [Fact]
    public void OverlapMassCountsUniqueSourceTokensOnce()
    {
        var inputs = new[]
        {
            Embedding([1f, 0f, 0f, 0f], 0, 100, 150),
            Embedding([0.9f, 0.1f, 0f, 0f], 50, 100, 150)
        };

        var result = inputs.CombineToSingle();

        Assert.Equal(150, result.SourceTokenCount);
        Assert.Equal(EmbeddingSourceMassMethod.TokenRangeCoverage, result.Aggregation.SourceMassMethod);
    }

    [Fact]
    public void IdenticalVectorsHaveNearPerfectCoherence()
    {
        var inputs = new[]
        {
            Embedding([1f, 0f, 0f, 0f], 0, 10, 20),
            Embedding([1f, 0f, 0f, 0f], 10, 10, 20)
        };

        var result = inputs.CombineToSingle();

        Assert.InRange(result.Aggregation.AggregationCoherence, 0.9999f, 1f);
        Assert.False(result.Aggregation.FallbackUsed);
        Assert.InRange(result.Aggregation.MinimumSourceSimilarity!.Value, 0.9999f, 1f);
    }

    [Fact]
    public void OpposingVectorsUseDeterministicMedoidFallback()
    {
        var inputs = new[]
        {
            Embedding([1f, 0f], 0, 10, 20),
            Embedding([-1f, 0f], 10, 10, 20)
        };

        var result = inputs.CombineToSingle();

        Assert.True(result.Aggregation.FallbackUsed);
        Assert.InRange(result.Aggregation.AggregationCoherence, 0f, 0.001f);
        var values = result.Vector.ToFloat32();
        Assert.InRange(MathF.Abs(values[0]), 0.9999f, 1f);
    }

    [Fact]
    public void RepeatedSemanticRegionGetsDiminishingInfluence()
    {
        var repeated = Enumerable.Range(0, 8)
            .Select(i => Embedding([1f, 0f], i * 10, 10, 100))
            .ToList();
        repeated.Add(Embedding([0f, 1f], 80, 10, 100));
        repeated.Add(Embedding([0f, 1f], 90, 10, 100));

        var result = repeated.CombineToSingle();
        var values = result.Vector.ToFloat32();

        // A linear mean would produce a 4:1 x/y ratio. SemanticCoverage deliberately compresses that gap.
        Assert.True(values[1] / values[0] > 0.25f);
        Assert.True(values[0] > values[1]);
    }

    [Fact]
    public void DimensionReductionChangesFingerprintAndMatchesReducedQuerySpace()
    {
        var values = Enumerable.Range(0, 16).Select(i => i % 3 == 0 ? 1f : -0.25f).ToArray();
        EmbeddingVectorMath.NormalizeInPlace(values);
        var input = Embedding(values, 0, 20, 20);
        var result = new[] { input }.CombineToSingle(new SingleEmbeddingOptions { OutputDimensions = 5 });
        var query = new QueryEmbedding
        {
            Vector = EmbeddingVector.FromFloat32(values),
            Identity = Identity(),
            SourceTokenCount = 3,
            InputTokenCount = 5
        };
        var reducedQuery = query.ReduceDimensions(5);

        Assert.Equal(5, result.Vector.Dimensions);
        Assert.NotEqual(input.Identity.EmbeddingSpaceFingerprint, result.Identity.EmbeddingSpaceFingerprint);
        Assert.Equal(result.Identity.EmbeddingSpaceFingerprint, reducedQuery.Identity.EmbeddingSpaceFingerprint);
        Assert.Equal(EmbeddingDimensionReductionProfiles.SrhtV1, result.DimensionReduction!.ProfileId);
        Assert.InRange(EmbeddingVectorMath.CosineSimilarity(result.Vector, reducedQuery.Vector), 0.999f, 1f);
    }

    [Fact]
    public void SrhtSupportsNonPowerOfTwoSourceDimensions()
    {
        var values = Enumerable.Range(0, 4095).Select(i => (float)Math.Sin(i + 1)).ToArray();
        EmbeddingVectorMath.NormalizeInPlace(values);
        var input = Embedding(values, 0, 10, 10);

        var result = new[] { input }.CombineToSingle(new SingleEmbeddingOptions
        {
            OutputDimensions = 2048,
            OutputFormat = EmbeddingVectorFormat.Float16
        });

        Assert.Equal(2048, result.Vector.Dimensions);
        Assert.Equal(EmbeddingVectorFormat.Float16, result.Vector.Format);
        var reconstructed = result.Vector.ToFloat32();
        var norm = Math.Sqrt(reconstructed.Sum(x => x * x));
        Assert.InRange(norm, 0.99, 1.01);
    }

    [Fact]
    public void SrhtIsDeterministic()
    {
        var values = Enumerable.Range(0, 32).Select(i => (float)Math.Cos(i)).ToArray();
        EmbeddingVectorMath.NormalizeInPlace(values);
        var input = Embedding(values, 0, 10, 10);
        var options = new SingleEmbeddingOptions { OutputDimensions = 7 };

        var first = new[] { input }.CombineToSingle(options);
        var second = new[] { input }.CombineToSingle(options);

        Assert.Equal(first.Vector.Data, second.Vector.Data);
        Assert.Equal(first.Identity.EmbeddingSpaceFingerprint, second.Identity.EmbeddingSpaceFingerprint);
    }

    [Fact]
    public void CannotExpandDimensions()
    {
        var input = Embedding([1f, 0f, 0f, 0f], 0, 10, 10);

        var exception = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new[] { input }.CombineToSingle(new SingleEmbeddingOptions { OutputDimensions = 8 }));

        Assert.Contains("exceeds the maximum available dimension", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void MismatchedEmbeddingSpacesAreRejected()
    {
        var first = Embedding([1f, 0f], 0, 10, 20);
        var second = Embedding([0f, 1f], 10, 10, 20) with
        {
            Identity = Identity() with { EmbeddingSpaceFingerprint = "other-space" }
        };

        Assert.Throws<EmbeddingSpaceMismatchException>(() => new[] { first, second }.CombineToSingle());
    }

    [Fact]
    public void MismatchedDimensionsAreRejected()
    {
        var first = Embedding([1f, 0f], 0, 10, 20);
        var second = Embedding([0f, 1f, 0f], 10, 10, 20);

        Assert.Throws<ArgumentException>(() => new[] { first, second }.CombineToSingle());
    }

    [Fact]
    public void SingleEmbeddingJsonRoundTrips()
    {
        var inputs = new[]
        {
            Embedding([1f, 0f, 0f, 0f], 0, 10, 20),
            Embedding([0f, 1f, 0f, 0f], 10, 10, 20)
        };
        var result = inputs.CombineToSingle(new SingleEmbeddingOptions
        {
            OutputDimensions = 2,
            OutputFormat = EmbeddingVectorFormat.Int8
        });

        var json = EmbeddingSerializer.SerializeJson(result);
        var restored = EmbeddingSerializer.DeserializeSingleJson(json);

        Assert.Equal(result.Identity, restored.Identity);
        Assert.Equal(result.Vector, restored.Vector);
        Assert.Equal(result.Aggregation, restored.Aggregation);
        Assert.Equal(result.DimensionReduction, restored.DimensionReduction);
    }

    private static TextEmbedding Embedding(
        IReadOnlyList<float> values,
        int tokenStart,
        int tokenLength,
        int documentTokenCount,
        EmbeddingVectorFormat format = EmbeddingVectorFormat.Float32)
    {
        var copy = values.ToArray();
        EmbeddingVectorMath.NormalizeInPlace(copy);
        return new TextEmbedding
        {
            Vector = EmbeddingVector.FromFloat32(copy, format),
            Identity = Identity(),
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

    private static EmbeddingIdentity Identity() => new()
    {
        ModelId = "test-model",
        SourceRevision = "r1",
        EmbeddingSpaceFingerprint = "test-space",
        IsNormalized = true
    };
}
