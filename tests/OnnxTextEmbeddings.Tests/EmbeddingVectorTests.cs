namespace OnnxTextEmbeddings.Tests;

public sealed class EmbeddingVectorTests
{
    private static readonly float[] Values = Normalize([0.32f, -0.18f, 0.55f, 0.05f, -0.72f, 0.14f, 0.09f]);

    [Theory]
    [InlineData(EmbeddingVectorFormat.Float32)]
    [InlineData(EmbeddingVectorFormat.Float16)]
    [InlineData(EmbeddingVectorFormat.Int8)]
    [InlineData(EmbeddingVectorFormat.Int4)]
    public void FormatsRoundTripWithHighCosine(EmbeddingVectorFormat format)
    {
        var vector = EmbeddingVector.FromFloat32(Values, format);
        var reconstructed = EmbeddingVector.FromFloat32(vector.ToFloat32(), EmbeddingVectorFormat.Float32);
        var original = EmbeddingVector.FromFloat32(Values, EmbeddingVectorFormat.Float32);
        var cosine = EmbeddingVectorMath.CosineSimilarity(original, reconstructed);
        Assert.True(cosine > (format == EmbeddingVectorFormat.Int4 ? 0.98f : 0.999f), $"Cosine was {cosine}");
    }

    [Fact]
    public void FromFloat32WithoutFormatPreservesFloat32()
    {
        var vector = EmbeddingVector.FromFloat32(Values);

        Assert.Equal(EmbeddingVectorFormat.Float32, vector.Format);
        Assert.Equal(Values.Length * sizeof(float), vector.Data.Length);
    }

    [Fact]
    public void ExplicitDownConversionSupportsEveryCompactFormat()
    {
        var fp32 = EmbeddingVector.FromFloat32(Values);

        Assert.Equal(EmbeddingVectorFormat.Float16, fp32.ConvertTo(EmbeddingVectorFormat.Float16).Format);
        Assert.Equal(EmbeddingVectorFormat.Int8, fp32.ConvertTo(EmbeddingVectorFormat.Int8).Format);
        Assert.Equal(EmbeddingVectorFormat.Int4, fp32.ConvertTo(EmbeddingVectorFormat.Int4).Format);
    }

    [Fact]
    public void Int4PacksTwoDimensionsPerByte()
    {
        var vector = EmbeddingVector.FromFloat32(Values, EmbeddingVectorFormat.Int4);
        Assert.Equal((Values.Length + 1) / 2, vector.Data.Length);
        Assert.Equal(EmbeddingQuantizationScheme.SymmetricPerVectorInt4V1, vector.Quantization!.Scheme);
    }

    [Fact]
    public void Int8UsesOneBytePerDimension()
    {
        var vector = EmbeddingVector.FromFloat32(Values, EmbeddingVectorFormat.Int8);
        Assert.Equal(Values.Length, vector.Data.Length);
        Assert.Equal(EmbeddingQuantizationScheme.SymmetricPerVectorInt8V1, vector.Quantization!.Scheme);
    }

    [Fact]
    public void MixedPrecisionSimilarityDoesNotRequireMatchingFormats()
    {
        var query = EmbeddingVector.FromFloat32(Values, EmbeddingVectorFormat.Float32);
        var int8 = EmbeddingVector.FromFloat32(Values, EmbeddingVectorFormat.Int8);
        var int4 = EmbeddingVector.FromFloat32(Values, EmbeddingVectorFormat.Int4);
        Assert.True(EmbeddingVectorMath.CosineSimilarity(query, int8) > 0.999f);
        Assert.True(EmbeddingVectorMath.CosineSimilarity(query, int4) > 0.98f);
    }

    private static float[] Normalize(float[] values)
    {
        EmbeddingVectorMath.NormalizeInPlace(values);
        return values;
    }
}
