using OnnxModelRuntime;

namespace OnnxTextEmbeddings.Tests;

public sealed class InferenceWorkerPoolTests
{
    [Fact]
    public void EmbeddingInferenceOptions_MapDirectlyToSharedRuntimeOptions()
    {
        var options = new InferenceOptions
        {
            ModelInstanceCount = 3,
            ThreadsPerModel = 12,
            MaximumAutoThreadsPerModel = 24,
            ConcurrentRequestsPerModel = 5,
            QueueCapacity = 77
        };

        var mapped = options.ToRuntimeOptions();
        Assert.Equal(3, mapped.ModelInstanceCount);
        Assert.Equal(12, mapped.ThreadsPerModel);
        Assert.Equal(24, mapped.MaximumAutoThreadsPerModel);
        Assert.Equal(5, mapped.ConcurrentRequestsPerModel);
        Assert.Equal(77, mapped.QueueCapacity);
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(4, 2)]
    [InlineData(16, 8)]
    [InlineData(32, 8)]
    public void EmbeddingResolution_DelegatesAutomaticConcurrencyPolicyToOnnxModelRuntime(int threads, int expected)
    {
        var embeddingOptions = new InferenceOptions { ThreadsPerModel = threads };
        var runtimeOptions = new OnnxModelRuntimeOptions { ThreadsPerModel = threads };

        Assert.Equal(runtimeOptions.Resolve().ConcurrentRequestsPerModel, embeddingOptions.Resolve().ConcurrentRequestsPerModel);
        Assert.Equal(expected, embeddingOptions.Resolve().ConcurrentRequestsPerModel);
    }
}
