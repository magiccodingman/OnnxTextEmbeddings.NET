using OnnxTextEmbeddings.PgVector;

namespace OnnxTextEmbeddings.Tests;

public sealed class PgVectorExtensionsTests
{
    [Fact]
    public void Int8RoundTripsThroughPgVectorWithHighCosineAgreement()
    {
        var source = EmbeddingVector.FromFloat32(new[] { 0.2f, -0.4f, 0.8f, 0.1f }, EmbeddingVectorFormat.Int8);
        var native = source.ToPgVector();
        var restored = native.ToEmbeddingVector(EmbeddingVectorFormat.Int8);

        Assert.Equal(source.Dimensions, restored.Dimensions);
        Assert.True(EmbeddingVectorMath.CosineSimilarity(source, restored) > 0.999f);
    }

    [Fact]
    public void Int4CanBeExpandedToPgVectorWithoutLosingDimensions()
    {
        var source = EmbeddingVector.FromFloat32(new[] { 0.2f, -0.4f, 0.8f, 0.1f, -0.3f }, EmbeddingVectorFormat.Int4);
        var native = source.ToPgVector();
        var values = native.ToArray();

        Assert.Equal(5, values.Length);
        Assert.True(values.Any(x => x != 0));
    }

    [Fact]
    public void HalfVectorRoundTripUsesFloat16ByDefault()
    {
        var source = EmbeddingVector.FromFloat32(new[] { 0.25f, 0.5f, -0.75f }, EmbeddingVectorFormat.Float32);
        var native = source.ToPgHalfVector();
        var restored = native.ToEmbeddingVector();

        Assert.Equal(EmbeddingVectorFormat.Float16, restored.Format);
        Assert.Equal(3, restored.Dimensions);
        Assert.True(EmbeddingVectorMath.CosineSimilarity(source, restored) > 0.999f);
    }
}
