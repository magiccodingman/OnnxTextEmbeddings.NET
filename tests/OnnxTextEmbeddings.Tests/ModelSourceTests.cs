using System.Net;
using System.Text;
using Microsoft.Extensions.Logging.Abstractions;

namespace OnnxTextEmbeddings.Tests;

public sealed class ModelSourceTests
{
    [Fact]
    public async Task HuggingFaceResolverSelectsRuntimeAssetsAndResolvedSha()
    {
        const string json = """
        {
          "sha": "abc123",
          "siblings": [
            { "rfilename": "model.onnx", "size": 12 },
            { "rfilename": "tokenizer.json", "size": 34 },
            { "rfilename": "config.json", "size": 56 },
            { "rfilename": "README.md", "size": 78 }
          ]
        }
        """;
        using var http = new HttpClient(new StaticHandler(_ => Json(json)));
        var options = new OnnxTextEmbeddingsOptions();
        options.Model.UseHuggingFace("owner/model");
        var source = new HuggingFaceModelSource(http, options, NullLogger<HuggingFaceModelSource>.Instance);

        var resolved = await source.ResolveAsync(TestContext.Current.CancellationToken);

        Assert.Equal("owner/model", resolved.ModelId);
        Assert.Equal("abc123", resolved.Revision);
        Assert.Equal(3, resolved.Assets.Count);
        Assert.DoesNotContain(resolved.Assets, asset => asset.Path == "README.md");
        Assert.All(resolved.Assets, asset => Assert.Contains("/resolve/abc123/", asset.Uri.AbsoluteUri, StringComparison.Ordinal));
    }

    [Fact]
    public async Task HttpManifestSupportsRelativeAssetsHashesAndSizes()
    {
        const string json = """
        {
          "modelId": "custom/embed",
          "revision": "v7",
          "assets": [
            { "path": "model.onnx", "url": "files/model.onnx", "size": 123, "sha256": "abcd" },
            "tokenizer.json"
          ]
        }
        """;
        using var http = new HttpClient(new StaticHandler(_ => Json(json)));
        var options = new OnnxTextEmbeddingsOptions();
        options.Model.UseHttpManifest(new Uri("https://models.example/embed/manifest.json"));
        var source = new HttpManifestModelSource(http, options);

        var resolved = await source.ResolveAsync(TestContext.Current.CancellationToken);

        Assert.Equal("custom/embed", resolved.ModelId);
        Assert.Equal("v7", resolved.Revision);
        Assert.Equal(new Uri("https://models.example/embed/files/model.onnx"), resolved.Assets[0].Uri);
        Assert.Equal(123, resolved.Assets[0].Size);
        Assert.Equal("abcd", resolved.Assets[0].Sha256);
        Assert.Equal(new Uri("https://models.example/embed/tokenizer.json"), resolved.Assets[1].Uri);
    }

    [Fact]
    public async Task CacheRejectsManifestPathTraversalBeforeActivation()
    {
        var temp = Path.Combine(Path.GetTempPath(), "onnx-text-embeddings-tests", Guid.NewGuid().ToString("N"));
        try
        {
            const string manifest = """
            {
              "modelId": "bad/model",
              "revision": "r1",
              "assets": ["../escape.onnx", "tokenizer.json"]
            }
            """;
            using var http = new HttpClient(new StaticHandler(request =>
                request.RequestUri?.AbsolutePath.EndsWith("manifest.json", StringComparison.Ordinal) == true
                    ? Json(manifest)
                    : new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent([1, 2, 3]) }));
            var options = new OnnxTextEmbeddingsOptions();
            options.Cache.Directory = temp;
            options.Model.UseHttpManifest(new Uri("https://models.example/manifest.json"));
            var hf = new HuggingFaceModelSource(http, options, NullLogger<HuggingFaceModelSource>.Instance);
            var manifestSource = new HttpManifestModelSource(http, options);
            var cache = new ModelCacheManager(http, options, hf, manifestSource, NullLogger<ModelCacheManager>.Instance);

            await Assert.ThrowsAsync<ModelDownloadException>(async () =>
                await cache.ResolveCandidateAsync(TestContext.Current.CancellationToken));

            Assert.False(File.Exists(Path.Combine(temp, "escape.onnx")));
        }
        finally
        {
            if (Directory.Exists(temp))
                Directory.Delete(temp, recursive: true);
        }
    }

    private static HttpResponseMessage Json(string json) => new(HttpStatusCode.OK)
    {
        Content = new StringContent(json, Encoding.UTF8, "application/json")
    };

    private sealed class StaticHandler(Func<HttpRequestMessage, HttpResponseMessage> responseFactory) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            Task.FromResult(responseFactory(request));
    }
}
