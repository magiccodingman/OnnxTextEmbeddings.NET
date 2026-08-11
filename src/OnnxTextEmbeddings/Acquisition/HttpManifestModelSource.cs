using System.Text.Json;

namespace OnnxTextEmbeddings;

internal sealed class HttpManifestModelSource(
    HttpClient httpClient,
    OnnxTextEmbeddingsOptions options) : IModelSource
{
    public async Task<ResolvedRemoteModel> ResolveAsync(CancellationToken cancellationToken)
    {
        var manifestUri = options.Model.ManifestUri ?? throw new ModelSourceException("No HTTP model manifest URI was configured.");
        using var request = new HttpRequestMessage(HttpMethod.Get, manifestUri);
        using var response = await httpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        var json = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;
        var modelId = root.TryGetProperty("modelId", out var modelIdElement) ? modelIdElement.GetString() : null;
        var revision = root.TryGetProperty("revision", out var revisionElement) ? revisionElement.GetString() : null;
        if (!root.TryGetProperty("assets", out var assetsElement) || assetsElement.ValueKind != JsonValueKind.Array)
            throw new ModelSourceException("HTTP model manifest must contain an assets array.");

        var assets = new List<RemoteModelAsset>();
        foreach (var item in assetsElement.EnumerateArray())
        {
            if (item.ValueKind == JsonValueKind.String)
            {
                var path = item.GetString()!;
                assets.Add(new RemoteModelAsset(path, new Uri(manifestUri, path)));
                continue;
            }
            var pathValue = item.GetProperty("path").GetString() ?? throw new ModelSourceException("Manifest asset path is empty.");
            var urlValue = item.TryGetProperty("url", out var urlElement) ? urlElement.GetString() : pathValue;
            var hash = item.TryGetProperty("sha256", out var hashElement) ? hashElement.GetString() : null;
            long? size = item.TryGetProperty("size", out var sizeElement) && sizeElement.TryGetInt64(out var parsedSize) ? parsedSize : null;
            assets.Add(new RemoteModelAsset(pathValue, new Uri(manifestUri, urlValue!), size, hash));
        }

        return new ResolvedRemoteModel(modelId ?? manifestUri.Host, revision ?? "manifest", assets);
    }
}

