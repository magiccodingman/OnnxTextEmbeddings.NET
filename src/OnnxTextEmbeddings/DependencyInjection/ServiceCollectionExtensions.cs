using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace OnnxTextEmbeddings;

public static class ServiceCollectionExtensions
{
    public static IServiceCollection AddOnnxTextEmbeddings(
        this IServiceCollection services,
        Action<OnnxTextEmbeddingsOptions>? configure = null)
    {
        ArgumentNullException.ThrowIfNull(services);
        var options = new OnnxTextEmbeddingsOptions();
        configure?.Invoke(options);
        options.Validate();

        services.AddSingleton(options);
        services.AddSingleton(new HttpClient
        {
            Timeout = Timeout.InfiniteTimeSpan
        });
        services.AddSingleton<HuggingFaceModelSource>();
        services.AddSingleton<HttpManifestModelSource>();
        services.AddSingleton<ModelCacheManager>();
        services.AddSingleton<TextEmbeddingService>();
        services.AddSingleton<ITextEmbeddingService>(sp => sp.GetRequiredService<TextEmbeddingService>());
        services.AddSingleton<SemanticSearchService>();
        services.AddSingleton<ISemanticSearch>(sp => sp.GetRequiredService<SemanticSearchService>());
        services.AddSingleton<ISemanticCandidateReranker>(sp => sp.GetRequiredService<SemanticSearchService>());
        services.AddSingleton<IHostedService, EmbeddingWarmupHostedService>();
        return services;
    }
}

internal sealed class EmbeddingWarmupHostedService(
    ITextEmbeddingService embeddingService,
    OnnxTextEmbeddingsOptions options,
    ILogger<EmbeddingWarmupHostedService> logger) : IHostedService
{
    private Task? _warmup;

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        if (!options.Initialization.WarmupOnStartup)
            return;
        if (options.Initialization.BlockHostStartupUntilReady)
        {
            await embeddingService.WaitUntilReadyAsync(cancellationToken).ConfigureAwait(false);
            return;
        }

        _warmup = WarmupAsync();
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        if (_warmup is null) return;
        try { await _warmup.WaitAsync(cancellationToken).ConfigureAwait(false); }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested) { }
    }

    private async Task WarmupAsync()
    {
        try
        {
            await embeddingService.WaitUntilReadyAsync(CancellationToken.None).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "ONNX text embedding warmup failed. Future requests will surface the initialization error.");
        }
    }
}
