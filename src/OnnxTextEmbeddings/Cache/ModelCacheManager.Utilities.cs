using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace OnnxTextEmbeddings;

internal sealed partial class ModelCacheManager
{
    private static (string? ModelId, int? MaxTokens, bool Normalize) ReadModelMetadata(string directory)
    {
        var manifest = Path.Combine(directory, "onnx-text-embeddings.json");
        if (File.Exists(manifest))
        {
            using var document = JsonDocument.Parse(File.ReadAllText(manifest));
            var root = document.RootElement;
            string? id = root.TryGetProperty("modelId", out var idElement) ? idElement.GetString() : null;
            int? max = null;
            if (root.TryGetProperty("model", out var model) && model.ValueKind == JsonValueKind.Object)
            {
                if (model.TryGetProperty("maxSequenceLength", out var maxElement) && maxElement.TryGetInt32(out var parsed))
                    max = parsed;
                var normalize = true;
                if (model.TryGetProperty("output", out var output) && output.ValueKind == JsonValueKind.Object &&
                    output.TryGetProperty("normalize", out var normalizeElement) && normalizeElement.ValueKind is JsonValueKind.True or JsonValueKind.False)
                    normalize = normalizeElement.GetBoolean();
                return (id, max, normalize);
            }
            return (id, max, true);
        }

        foreach (var filename in new[] { "tokenizer_config.json", "config.json" })
        {
            var path = Directory.EnumerateFiles(directory, filename, SearchOption.AllDirectories).FirstOrDefault();
            if (path is null) continue;
            try
            {
                using var document = JsonDocument.Parse(File.ReadAllText(path));
                foreach (var property in new[] { "model_max_length", "max_position_embeddings", "max_seq_length" })
                {
                    if (document.RootElement.TryGetProperty(property, out var value) && value.TryGetInt64(out var max) && max is > 0 and <= 1_000_000)
                        return (null, (int)max, true);
                }
            }
            catch (JsonException) { }
        }
        return (null, null, true);
    }

    private string GetModelCacheRoot()
    {
        var root = options.Cache.Directory;
        if (string.IsNullOrWhiteSpace(root))
            root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "OnnxTextEmbeddings", "models");
        var sourceKey = options.Model.SourceKind == ModelSourceKind.HuggingFace ? options.Model.RepositoryId : options.Model.ManifestUri?.ToString() ?? "http";
        return Path.Combine(root, Sanitize(sourceKey));
    }

    private async Task<FileStream> AcquireLockAsync(string cacheRoot, CancellationToken cancellationToken)
    {
        var path = Path.Combine(cacheRoot, ".lock");
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                return new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
            }
            catch (IOException)
            {
                await Task.Delay(options.Cache.LockRetryDelay, cancellationToken).ConfigureAwait(false);
            }
        }
    }

    private void CleanupStaging(string cacheRoot)
    {
        var path = Path.Combine(cacheRoot, "staging");
        if (!Directory.Exists(path)) return;
        foreach (var directory in Directory.EnumerateDirectories(path))
        {
            try { Directory.Delete(directory, true); }
            catch (IOException ex) { logger.LogDebug(ex, "Unable to remove abandoned staging directory {Directory}.", directory); }
        }
    }

    private async Task DeleteOtherSnapshotsAsync(string cacheRoot, string activeDirectory, CancellationToken cancellationToken)
    {
        var snapshots = Path.Combine(cacheRoot, "snapshots");
        if (!Directory.Exists(snapshots)) return;
        foreach (var directory in Directory.EnumerateDirectories(snapshots))
        {
            if (Path.GetFullPath(directory).Equals(Path.GetFullPath(activeDirectory), StringComparison.Ordinal))
                continue;
            await DeleteDirectoryWithRetriesAsync(directory, cancellationToken).ConfigureAwait(false);
        }
    }

    private async Task DeleteDirectoryWithRetriesAsync(string directory, CancellationToken cancellationToken)
    {
        for (var attempt = 0; attempt <= options.Cache.LockedFileDeleteRetries; attempt++)
        {
            try
            {
                Directory.Delete(directory, true);
                return;
            }
            catch (IOException) when (attempt < options.Cache.LockedFileDeleteRetries)
            {
                await Task.Delay(options.Cache.LockedFileDeleteRetryDelay, cancellationToken).ConfigureAwait(false);
            }
            catch (UnauthorizedAccessException) when (attempt < options.Cache.LockedFileDeleteRetries)
            {
                await Task.Delay(options.Cache.LockedFileDeleteRetryDelay, cancellationToken).ConfigureAwait(false);
            }
        }
    }

    private static async Task<string> ComputeFingerprintAsync(string directory, CancellationToken cancellationToken)
    {
        using var aggregate = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var files = Directory.EnumerateFiles(directory, "*", SearchOption.AllDirectories)
            .Where(IsFingerprintAsset)
            .OrderBy(x => Path.GetRelativePath(directory, x), StringComparer.Ordinal)
            .ToArray();
        foreach (var file in files)
        {
            var relative = Path.GetRelativePath(directory, file).Replace('\\', '/');
            aggregate.AppendData(Encoding.UTF8.GetBytes(relative));
            aggregate.AppendData(new byte[] { 0 });
            var hash = Convert.FromHexString(await Sha256FileAsync(file, cancellationToken).ConfigureAwait(false));
            aggregate.AppendData(hash);
        }
        return Convert.ToHexString(aggregate.GetHashAndReset()).ToLowerInvariant();
    }

    private static bool IsFingerprintAsset(string path)
    {
        var ext = Path.GetExtension(path);
        return ext.Equals(".onnx", StringComparison.OrdinalIgnoreCase)
            || ext.Equals(".json", StringComparison.OrdinalIgnoreCase)
            || ext.Equals(".txt", StringComparison.OrdinalIgnoreCase)
            || ext.Equals(".model", StringComparison.OrdinalIgnoreCase)
            || ext.Equals(".data", StringComparison.OrdinalIgnoreCase)
            || ext.Equals(".onnx_data", StringComparison.OrdinalIgnoreCase);
    }

    private static async Task<string> Sha256FileAsync(string path, CancellationToken cancellationToken)
    {
        await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 128, true);
        var hash = await SHA256.HashDataAsync(stream, cancellationToken).ConfigureAwait(false);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private static string ValidateRelativePath(string path)
    {
        if (string.IsNullOrWhiteSpace(path) || Path.IsPathRooted(path))
            throw new ModelDownloadException($"Invalid model asset path '{path}'.");
        var normalized = path.Replace('/', Path.DirectorySeparatorChar);
        if (normalized.Split(Path.DirectorySeparatorChar).Any(x => x == ".."))
            throw new ModelDownloadException($"Invalid model asset path '{path}'.");
        return normalized;
    }

    private static string Sanitize(string value)
    {
        var chars = value.Select(c => char.IsLetterOrDigit(c) || c is '-' or '_' or '.' ? c : '_').ToArray();
        return new string(chars);
    }
}
