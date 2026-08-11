namespace OnnxTextEmbeddings.Tests;

public sealed class OptionsTests
{
    [Fact]
    public void Defaults_AreOpinionatedForLightweightCpuUsage()
    {
        var options = new OnnxTextEmbeddingsOptions();

        Assert.Equal(1024, options.DocumentChunkMaxTokens);
        Assert.Equal(1024, options.QueryMaxTokens);
        Assert.Equal(1, options.Inference.WorkerCount);
        Assert.Equal(12, options.Inference.MaximumAutoThreadsPerWorker);
        Assert.Equal(EmbeddingVectorFormat.Int8, options.Vectors.DocumentFormat);
        Assert.Equal(EmbeddingVectorFormat.Float32, options.Vectors.QueryFormat);
        Assert.Equal(JasperModelPresets.Int8Repository, options.Model.RepositoryId);
    }

    [Theory]
    [InlineData(JasperModelPrecision.Int8, JasperModelPresets.Int8Repository)]
    [InlineData(JasperModelPrecision.Int4, JasperModelPresets.Int4Repository)]
    [InlineData(JasperModelPrecision.Float32, JasperModelPresets.Float32Repository)]
    public void JasperPreset_SelectsExpectedRepository(JasperModelPrecision precision, string expected)
    {
        var options = new OnnxTextEmbeddingsOptions();
        options.Model.UseJasper(precision);
        Assert.Equal(expected, options.Model.RepositoryId);
    }

    [Fact]
    public void InvalidConfiguration_IsRejected()
    {
        var options = new OnnxTextEmbeddingsOptions { QueryMaxTokens = 0 };
        Assert.Throws<ArgumentOutOfRangeException>(() => options.Validate());
    }
}
