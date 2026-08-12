using System.Net;
using System.Security.Cryptography;
using System.Text;

namespace OnnxTextEmbeddings.Tests;

public sealed class ModelSourceTests
{
    [Fact]
    public async Task LocalArtifactSnapshot_PreservesEmbeddingFingerprintAndModelMetadata()
    {
        var temp = Path.Combine(Path.GetTempPath(), "onnx-text-embeddings-artifacts", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(temp);
        try
        {
            await File.WriteAllBytesAsync(Path.Combine(temp, "model.onnx"), [1, 2, 3, 4], TestContext.Current.CancellationToken);
            await File.WriteAllTextAsync(Path.Combine(temp, "tokenizer.json"), "{}", TestContext.Current.CancellationToken);
            await File.WriteAllTextAsync(Path.Combine(temp, "config.json"), "{\"max_position_embeddings\":2048}", TestContext.Current.CancellationToken);
            await File.WriteAllTextAsync(Path.Combine(temp, "onnx-text-embeddings.json"), "{\"modelId\":\"local/test\",\"model\":{\"maxSequenceLength\":1024,\"output\":{\"normalize\":true}}}", TestContext.Current.CancellationToken);

            var options = new OnnxTextEmbeddingsOptions();
            options.Model.UseLocalDirectory(temp);
            using var http = new HttpClient();
            using var artifacts = new EmbeddingArtifactManager(http, options);

            var candidate = await artifacts.ResolveCandidateAsync(TestContext.Current.CancellationToken);

            Assert.False(candidate.RequiresPromotion);
            Assert.Equal("local/test", candidate.Snapshot.ModelId);
            Assert.Equal("local", candidate.Snapshot.SourceRevision);
            Assert.Equal(1024, candidate.Snapshot.ModelMaxTokens);
            Assert.Equal(Path.Combine(temp, "model.onnx"), candidate.Snapshot.ModelPath);
            Assert.Equal(Path.Combine(temp, "tokenizer.json"), candidate.Snapshot.TokenizerPath);
            Assert.Equal(await ComputeLegacyFingerprintAsync(temp), candidate.Snapshot.EmbeddingSpaceFingerprint);
        }
        finally
        {
            if (Directory.Exists(temp))
                Directory.Delete(temp, recursive: true);
        }
    }

    [Fact]
    public async Task ModelArtifactsPathTraversal_IsTranslatedToExistingPublicDownloadException()
    {
        var temp = Path.Combine(Path.GetTempPath(), "onnx-text-embeddings-artifacts", Guid.NewGuid().ToString("N"));
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
            using var artifacts = new EmbeddingArtifactManager(http, options);

            await Assert.ThrowsAsync<ModelDownloadException>(async () =>
                await artifacts.ResolveCandidateAsync(TestContext.Current.CancellationToken));

            Assert.False(File.Exists(Path.Combine(temp, "escape.onnx")));
        }
        finally
        {
            if (Directory.Exists(temp))
                Directory.Delete(temp, recursive: true);
        }
    }

    private static async Task<string> ComputeLegacyFingerprintAsync(string directory)
    {
        using var aggregate = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var files = Directory.EnumerateFiles(directory, "*", SearchOption.AllDirectories)
            .Where(path => new[] { ".onnx", ".json", ".txt", ".model", ".data", ".onnx_data" }
                .Contains(Path.GetExtension(path), StringComparer.OrdinalIgnoreCase))
            .OrderBy(path => Path.GetRelativePath(directory, path), StringComparer.Ordinal)
            .ToArray();
        foreach (var file in files)
        {
            var relative = Path.GetRelativePath(directory, file).Replace('\\', '/');
            aggregate.AppendData(Encoding.UTF8.GetBytes(relative));
            aggregate.AppendData([0]);
            await using var stream = File.OpenRead(file);
            aggregate.AppendData(await SHA256.HashDataAsync(stream, TestContext.Current.CancellationToken));
        }
        return Convert.ToHexString(aggregate.GetHashAndReset()).ToLowerInvariant();
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
