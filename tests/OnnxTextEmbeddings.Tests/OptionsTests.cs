namespace OnnxTextEmbeddings.Tests;

public sealed class OptionsTests
{
    [Fact]
    public void Defaults_AreOpinionatedForJasperInt8CpuUsage()
    {
        var options = new OnnxTextEmbeddingsOptions();
        var resolved = options.Inference.Resolve(options.Model.JasperPrecision);

        Assert.Equal(1024, options.DocumentChunkMaxTokens);
        Assert.Equal(1024, options.QueryMaxTokens);
        Assert.Equal(1, options.Inference.ModelInstanceCount);
        Assert.Equal(16, options.Inference.ThreadsPerModel);
        Assert.Equal(0, options.Inference.ConcurrentRequestsPerModel);
        Assert.Equal(5, resolved.ConcurrentRequestsPerModel);
        Assert.Equal(5, resolved.TotalConcurrentRequests);
        Assert.Equal(EmbeddingVectorFormat.Float32, options.Vectors.DocumentFormat);
        Assert.Equal(EmbeddingVectorFormat.Float32, options.Vectors.QueryFormat);
        Assert.Equal(JasperModelPresets.Int8Repository, options.Model.RepositoryId);
        Assert.Equal(JasperModelPrecision.Int8, options.Model.JasperPrecision);
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 1)]
    [InlineData(4, 2)]
    [InlineData(8, 4)]
    [InlineData(12, 4)]
    [InlineData(16, 4)]
    [InlineData(24, 4)]
    [InlineData(32, 4)]
    public void AutomaticConcurrency_UsesGlobalCapOfFour(int threads, int expectedConcurrency)
    {
        var options = new InferenceOptions { ThreadsPerModel = threads };
        Assert.Equal(expectedConcurrency, options.Resolve().ConcurrentRequestsPerModel);
    }

    [Fact]
    public void JasperInt8_AutomaticConcurrencyCapsAtFive()
    {
        var options = new InferenceOptions { ThreadsPerModel = 16 };
        Assert.Equal(5, options.Resolve(JasperModelPrecision.Int8).ConcurrentRequestsPerModel);
    }

    [Theory]
    [InlineData(JasperModelPrecision.Int4)]
    [InlineData(JasperModelPrecision.Float32)]
    public void OtherJasperPrecisions_UseGlobalConcurrencyCap(JasperModelPrecision precision)
    {
        var options = new InferenceOptions { ThreadsPerModel = 16 };
        Assert.Equal(4, options.Resolve(precision).ConcurrentRequestsPerModel);
    }

    [Fact]
    public void ExplicitConcurrency_IsNotSilentlyCapped()
    {
        var options = new InferenceOptions
        {
            ThreadsPerModel = 16,
            ConcurrentRequestsPerModel = 12
        };

        Assert.Equal(12, options.Resolve(JasperModelPrecision.Int8).ConcurrentRequestsPerModel);
    }

    [Fact]
    public void MultipleModelInstancesMultiplyTotalConcurrency()
    {
        var options = new InferenceOptions
        {
            ModelInstanceCount = 2,
            ThreadsPerModel = 16
        };
        var resolved = options.Resolve(JasperModelPrecision.Int8);

        Assert.Equal(5, resolved.ConcurrentRequestsPerModel);
        Assert.Equal(10, resolved.TotalConcurrentRequests);
    }

    [Theory]
    [InlineData(JasperModelPrecision.Int8, JasperModelPresets.Int8Repository)]
    [InlineData(JasperModelPrecision.Int4, JasperModelPresets.Int4Repository)]
    [InlineData(JasperModelPrecision.Float32, JasperModelPresets.Float32Repository)]
    public void JasperPreset_SelectsExpectedRepositoryAndTuningProfile(JasperModelPrecision precision, string expected)
    {
        var options = new OnnxTextEmbeddingsOptions();
        options.Model.UseJasper(precision);
        Assert.Equal(expected, options.Model.RepositoryId);
        Assert.Equal(precision, options.Model.JasperPrecision);
    }

    [Fact]
    public void CustomModel_ClearsJasperTuningProfile()
    {
        var options = new OnnxTextEmbeddingsOptions();
        options.Model.UseHuggingFace("owner/custom-model");

        Assert.Null(options.Model.JasperPrecision);
        Assert.Equal(4, options.Inference.Resolve(options.Model.JasperPrecision).ConcurrentRequestsPerModel);
    }

    [Fact]
    public void InvalidConfiguration_IsRejected()
    {
        var options = new OnnxTextEmbeddingsOptions { QueryMaxTokens = 0 };
        Assert.Throws<ArgumentOutOfRangeException>(() => options.Validate());
    }
}
