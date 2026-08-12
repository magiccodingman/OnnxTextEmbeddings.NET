namespace OnnxTextEmbeddings.Tests;

public sealed class TokenCountInfoTests
{
    [Fact]
    public void QueryTokenCount_ReportsConfiguredAndModelLimitsIndependently()
    {
        var count = new QueryTokenCount(
            SourceTokenCount: 1100,
            InputTokenCount: 1102,
            QueryMaxTokens: 1024,
            ModelMaxTokens: 4096);

        Assert.False(count.FitsConfiguredLimit);
        Assert.True(count.FitsModelLimit);
        Assert.False(count.Fits);
    }

    [Fact]
    public void QueryTokenCount_FitsWhenBothLimitsAllowInput()
    {
        var count = new QueryTokenCount(100, 102, 1024, 4096);

        Assert.True(count.FitsConfiguredLimit);
        Assert.True(count.FitsModelLimit);
        Assert.True(count.Fits);
    }
}
