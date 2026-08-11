namespace OnnxTextEmbeddings.Tests;

public sealed class OptionsTests
{
    [Fact]
    public void Defaults_AreOpinionatedForLightweightCpuUsage()
    {
        var options = new OnnxTextEmbeddingsOptions();
        var resolved = options.Inference.Resolve();

        Assert.Equal(1024, options.DocumentChunkMaxTokens);
        Assert.Equal(1024, options.QueryMaxTokens);
        Assert.Equal(1, options.Inference.ModelInstanceCount);
        Assert.Equal(16, options.Inference.ThreadsPerModel);
        Assert.Equal(0, options.Inference.ConcurrentRequestsPerModel);
        Assert.Equal(8, resolved.ConcurrentRequestsPerModel);
        Assert.Equal(8, resolved.TotalConcurrentRequests);
        Assert.Equal(EmbeddingVectorFormat.Int8, options.Vectors.DocumentFormat);
        Assert.Equal(EmbeddingVectorFormat.Float32, options.Vectors.QueryFormat);
        Assert.Equal(JasperModelPresets.Int8Repository, options.Model.RepositoryId);
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 1)]
    [InlineData(4, 2)]
    [InlineData(8, 4)]
    [InlineData(12, 6)]
    [InlineData(16, 8)]
    [InlineData(24, 8)]
    [InlineData(32, 8)]
    public void AutomaticConcurrency_IsHalfThreadsCappedAtEight(int threads, int expectedConcurrency)
    {
        var options = new InferenceOptions { ThreadsPerModel = threads };
        Assert.Equal(expectedConcurrency, options.Resolve().ConcurrentRequestsPerModel);
    }

    [Fact]
    public void ExplicitConcurrency_IsNotSilentlyCapped()
    {
        var options = new InferenceOptions
        {
            ThreadsPerModel = 16,
            ConcurrentRequestsPerModel = 12
        };

        Assert.Equal(12, options.Resolve().ConcurrentRequestsPerModel);
    }

    [Fact]
    public void MultipleModelInstancesMultiplyTotalConcurrency()
    {
        var options = new InferenceOptions
        {
            ModelInstanceCount = 2,
            ThreadsPerModel = 16
        };
        var resolved = options.Resolve();

        Assert.Equal(8, resolved.ConcurrentRequestsPerModel);
        Assert.Equal(16, resolved.TotalConcurrentRequests);
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
