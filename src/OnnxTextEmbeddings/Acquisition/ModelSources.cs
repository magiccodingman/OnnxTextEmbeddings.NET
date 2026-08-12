using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace OnnxTextEmbeddings;

internal sealed record RemoteModelAsset(
    string Path,
    Uri Uri,
    long? Size = null,
    string? Sha256 = null);

internal sealed record ResolvedRemoteModel(
    string ModelId,
    string Revision,
    IReadOnlyList<RemoteModelAsset> Assets);

internal sealed record ModelSnapshot(
    string ModelId,
    string SourceRevision,
    string EmbeddingSpaceFingerprint,
    string Directory,
    string ModelPath,
    string TokenizerPath,
    int? ModelMaxTokens,
    bool NormalizeOutput);

internal sealed record ModelCandidate(ModelSnapshot Snapshot, bool RequiresPromotion, string? CacheRoot);

internal interface IModelSource
{
    Task<ResolvedRemoteModel> ResolveAsync(CancellationToken cancellationToken);
}

internal sealed class HuggingFaceModelSource(
    HttpClient httpClient,
    OnnxTextEmbeddingsOptions options,
    ILogger<HuggingFaceModelSource> logger) : IModelSource
{
    public async Task<ResolvedRemoteModel> ResolveAsync(CancellationToken cancellationToken)
    {
        var model = options.Model;
        var repositoryId = model.RepositoryId;
        var revision = string.IsNullOrWhiteSpace(model.Revision) ? "main" : model.Revision;
        var url = $"https://huggingface.co/api/models/{Uri.EscapeDataString(repositoryId).Replace("%2F", "/", StringComparison.OrdinalIgnoreCase)}?revision={Uri.EscapeDataString(revision)}";

        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        AddAuthorization(request, model.AccessToken);
        using var response = await httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
            throw new ModelSourceException($"Hugging Face returned {(int)response.StatusCode} while resolving '{repositoryId}'.");

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        using var document = await JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken).ConfigureAwait(false);
        var root = document.RootElement;
        var sourceRevision = root.TryGetProperty("sha", out var sha) && sha.ValueKind == JsonValueKind.String
            ? sha.GetString() ?? revision
            : revision;

        if (!root.TryGetProperty("siblings", out var siblings) || siblings.ValueKind != JsonValueKind.Array)
            throw new ModelSourceException($"Hugging Face repository '{repositoryId}' did not expose a file list.");

        var filenames = new List<(string Path, long? Size)>();
        foreach (var sibling in siblings.EnumerateArray())
        {
            if (!sibling.TryGetProperty("rfilename", out var filenameElement) || filenameElement.ValueKind != JsonValueKind.String)
                continue;
            var path = filenameElement.GetString();
            if (string.IsNullOrWhiteSpace(path) || !ShouldDownload(path))
                continue;
            long? size = null;
            if (sibling.TryGetProperty("size", out var sizeElement) && sizeElement.TryGetInt64(out var parsedSize))
                size = parsedSize;
            filenames.Add((path, size));
        }

        if (!filenames.Any(x => x.Path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase)))
            throw new ModelSourceException($"Hugging Face repository '{repositoryId}' contains no downloadable ONNX model.");
        if (!filenames.Any(x => x.Path.EndsWith("tokenizer.json", StringComparison.OrdinalIgnoreCase)))
            throw new ModelSourceException($"Hugging Face repository '{repositoryId}' contains no tokenizer.json.");

        var escapedRevision = Uri.EscapeDataString(sourceRevision);
        var assets = filenames.Select(x => new RemoteModelAsset(
            x.Path,
            new Uri($"https://huggingface.co/{repositoryId}/resolve/{escapedRevision}/{EscapePath(x.Path)}?download=true"),
            x.Size)).ToArray();

        logger.LogInformation("Resolved Hugging Face model {Repository} at revision {Revision} with {AssetCount} runtime assets.", repositoryId, sourceRevision, assets.Length);
        return new ResolvedRemoteModel(repositoryId, sourceRevision, assets);
    }

    private static bool ShouldDownload(string path)
    {
        var name = Path.GetFileName(path);
        if (name.Equals("README.md", StringComparison.OrdinalIgnoreCase) || name.Equals(".gitattributes", StringComparison.OrdinalIgnoreCase))
            return false;
        var extension = Path.GetExtension(path);
        return extension.Equals(".onnx", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".json", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".txt", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".model", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".data", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".onnx_data", StringComparison.OrdinalIgnoreCase);
    }

    private static string EscapePath(string path) => string.Join('/', path.Split('/').Select(Uri.EscapeDataString));

    internal static void AddAuthorization(HttpRequestMessage request, string? accessToken)
    {
        if (!string.IsNullOrWhiteSpace(accessToken))
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", accessToken);
    }
}

