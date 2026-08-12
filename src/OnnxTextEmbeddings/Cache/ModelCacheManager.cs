using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace OnnxTextEmbeddings;

internal sealed partial class ModelCacheManager
{
    private readonly HttpClient httpClient;
    private readonly OnnxTextEmbeddingsOptions options;
    private readonly HuggingFaceModelSource huggingFaceSource;
    private readonly HttpManifestModelSource httpManifestSource;
    private readonly ILogger<ModelCacheManager> logger;

    public ModelCacheManager(
        HttpClient httpClient,
        OnnxTextEmbeddingsOptions options,
        HuggingFaceModelSource huggingFaceSource,
        HttpManifestModelSource httpManifestSource,
        ILogger<ModelCacheManager> logger)
    {
        this.httpClient = httpClient;
        this.options = options;
        this.huggingFaceSource = huggingFaceSource;
        this.httpManifestSource = httpManifestSource;
        this.logger = logger;
    }

    public async Task<ModelCandidate> ResolveCandidateAsync(CancellationToken cancellationToken, bool forceRemoteCheck = false)
    {
        if (options.Model.SourceKind == ModelSourceKind.LocalDirectory)
            return new ModelCandidate(await LoadLocalAsync(cancellationToken).ConfigureAwait(false), false, null);

        var cacheRoot = GetModelCacheRoot();
        Directory.CreateDirectory(cacheRoot);
        await using var cacheLock = await AcquireLockAsync(cacheRoot, cancellationToken).ConfigureAwait(false);
        CleanupStaging(cacheRoot);
        var current = await TryLoadCurrentAsync(cacheRoot, cancellationToken).ConfigureAwait(false);

        if (!forceRemoteCheck && current is not null && options.Model.UpdatePolicy is ModelUpdatePolicy.Never or ModelUpdatePolicy.Manual)
            return new ModelCandidate(current, false, cacheRoot);

        ResolvedRemoteModel remote;
        try
        {
            remote = options.Model.SourceKind switch
            {
                ModelSourceKind.HuggingFace => await huggingFaceSource.ResolveAsync(cancellationToken).ConfigureAwait(false),
                ModelSourceKind.HttpManifest => await httpManifestSource.ResolveAsync(cancellationToken).ConfigureAwait(false),
                _ => throw new ModelSourceException($"Unsupported model source: {options.Model.SourceKind}.")
            };
        }
        catch (Exception ex) when (current is not null && ex is not OperationCanceledException)
        {
            logger.LogWarning(ex, "Unable to check the remote model. Using the existing cached snapshot.");
            return new ModelCandidate(current, false, cacheRoot);
        }

        if (current is not null && current.SourceRevision == remote.Revision)
            return new ModelCandidate(current, false, cacheRoot);

        var candidate = await DownloadCandidateAsync(cacheRoot, remote, cancellationToken).ConfigureAwait(false);
        return new ModelCandidate(candidate, true, cacheRoot);
    }

    public async Task DiscardAsync(ModelCandidate candidate, CancellationToken cancellationToken)
    {
        if (!candidate.RequiresPromotion || candidate.CacheRoot is null)
            return;

        await using var cacheLock = await AcquireLockAsync(candidate.CacheRoot, cancellationToken).ConfigureAwait(false);
        if (Directory.Exists(candidate.Snapshot.Directory))
            await DeleteDirectoryWithRetriesAsync(candidate.Snapshot.Directory, cancellationToken).ConfigureAwait(false);
    }

    public async Task PromoteAsync(ModelCandidate candidate, CancellationToken cancellationToken)
    {
        if (!candidate.RequiresPromotion || candidate.CacheRoot is null)
            return;

        var cacheRoot = candidate.CacheRoot;
        await using var cacheLock = await AcquireLockAsync(cacheRoot, cancellationToken).ConfigureAwait(false);
        var relative = Path.GetRelativePath(Path.Combine(cacheRoot, "snapshots"), candidate.Snapshot.Directory);
        var record = new CurrentCacheRecord(relative, candidate.Snapshot.SourceRevision, candidate.Snapshot.EmbeddingSpaceFingerprint);
        var currentPath = Path.Combine(cacheRoot, "current.json");
        var tempPath = currentPath + ".tmp";
        var json = JsonSerializer.Serialize(record, EmbeddingJsonContext.Compact.CurrentCacheRecord);
        await File.WriteAllTextAsync(tempPath, json, cancellationToken).ConfigureAwait(false);
        File.Move(tempPath, currentPath, true);
    }

    public async Task CleanupOldSnapshotsAsync(ModelCandidate candidate, CancellationToken cancellationToken)
    {
        if (candidate.CacheRoot is null)
            return;

        await using var cacheLock = await AcquireLockAsync(candidate.CacheRoot, cancellationToken).ConfigureAwait(false);
        await DeleteOtherSnapshotsAsync(candidate.CacheRoot, candidate.Snapshot.Directory, cancellationToken).ConfigureAwait(false);
    }

    private async Task<ModelSnapshot> DownloadCandidateAsync(string cacheRoot, ResolvedRemoteModel remote, CancellationToken cancellationToken)
    {
        var stagingRoot = Path.Combine(cacheRoot, "staging", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(stagingRoot);
        try
        {
            foreach (var asset in remote.Assets)
            {
                var relativePath = ValidateRelativePath(asset.Path);
                var destination = Path.GetFullPath(Path.Combine(stagingRoot, relativePath));
                if (!destination.StartsWith(Path.GetFullPath(stagingRoot) + Path.DirectorySeparatorChar, StringComparison.Ordinal))
                    throw new ModelDownloadException($"Asset path '{asset.Path}' escapes the model staging directory.");
                Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
                await DownloadAssetAsync(asset, destination, cancellationToken).ConfigureAwait(false);
            }

            var fingerprint = await ComputeFingerprintAsync(stagingRoot, cancellationToken).ConfigureAwait(false);
            var snapshotName = $"{Sanitize(remote.Revision)}-{fingerprint[..12]}";
            var snapshotsRoot = Path.Combine(cacheRoot, "snapshots");
            Directory.CreateDirectory(snapshotsRoot);
            var finalDirectory = Path.Combine(snapshotsRoot, snapshotName);
            if (Directory.Exists(finalDirectory))
                Directory.Delete(finalDirectory, true);
            Directory.Move(stagingRoot, finalDirectory);
            return LoadSnapshot(finalDirectory, remote.ModelId, remote.Revision, fingerprint);
        }
        catch
        {
            if (Directory.Exists(stagingRoot))
                Directory.Delete(stagingRoot, true);
            throw;
        }
    }

    private async Task DownloadAssetAsync(RemoteModelAsset asset, string destination, CancellationToken cancellationToken)
    {
        Exception? lastError = null;
        for (var attempt = 1; attempt <= 3; attempt++)
        {
            try
            {
                using var request = new HttpRequestMessage(HttpMethod.Get, asset.Uri);
                HuggingFaceModelSource.AddAuthorization(request, options.Model.AccessToken);
                using var response = await httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
                if ((int)response.StatusCode is 408 or 429 or 500 or 502 or 503 or 504)
                {
                    if (attempt < 3)
                    {
                        var delay = response.Headers.RetryAfter?.Delta ?? TimeSpan.FromSeconds(attempt);
                        await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
                        continue;
                    }
                }
                response.EnsureSuccessStatusCode();
                var partial = destination + ".partial";
                await using (var source = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false))
                await using (var target = new FileStream(partial, FileMode.Create, FileAccess.Write, FileShare.None, 1024 * 128, true))
                    await source.CopyToAsync(target, cancellationToken).ConfigureAwait(false);

                var length = new FileInfo(partial).Length;
                if (asset.Size is { } expectedSize && expectedSize > 0 && length != expectedSize)
                    throw new ModelDownloadException($"Downloaded '{asset.Path}' is {length} bytes; expected {expectedSize}.");
                if (!string.IsNullOrWhiteSpace(asset.Sha256))
                {
                    var actual = await Sha256FileAsync(partial, cancellationToken).ConfigureAwait(false);
                    if (!actual.Equals(asset.Sha256, StringComparison.OrdinalIgnoreCase))
                        throw new ModelDownloadException($"SHA-256 mismatch for '{asset.Path}'.");
                }
                File.Move(partial, destination, true);
                return;
            }
            catch (Exception ex) when (ex is HttpRequestException or IOException or ModelDownloadException)
            {
                lastError = ex;
                if (attempt < 3)
                    await Task.Delay(TimeSpan.FromSeconds(attempt), cancellationToken).ConfigureAwait(false);
            }
        }
        throw new ModelDownloadException($"Failed to download '{asset.Path}'.", lastError!);
    }

    private async Task<ModelSnapshot?> TryLoadCurrentAsync(string cacheRoot, CancellationToken cancellationToken)
    {
        var currentPath = Path.Combine(cacheRoot, "current.json");
        if (!File.Exists(currentPath))
            return null;
        try
        {
            var json = await File.ReadAllTextAsync(currentPath, cancellationToken).ConfigureAwait(false);
            var record = JsonSerializer.Deserialize(json, EmbeddingJsonContext.Compact.CurrentCacheRecord);
            if (record is null)
                return null;
            var directory = Path.Combine(cacheRoot, "snapshots", record.DirectoryName);
            if (!Directory.Exists(directory))
                return null;
            return LoadSnapshot(directory, options.Model.RepositoryId, record.SourceRevision, record.Fingerprint);
        }
        catch (Exception ex) when (ex is IOException or JsonException or ModelValidationException)
        {
            logger.LogWarning(ex, "Cached model metadata is invalid and will be ignored.");
            return null;
        }
    }

    private async Task<ModelSnapshot> LoadLocalAsync(CancellationToken cancellationToken)
    {
        var directory = Path.GetFullPath(options.Model.LocalDirectory ?? throw new ModelSourceException("No local model directory was configured."));
        if (!Directory.Exists(directory))
            throw new ModelSourceException($"Local model directory '{directory}' does not exist.");
        var fingerprint = await ComputeFingerprintAsync(directory, cancellationToken).ConfigureAwait(false);
        return LoadSnapshot(directory, Path.GetFileName(directory), "local", fingerprint);
    }

    private ModelSnapshot LoadSnapshot(string directory, string modelId, string revision, string fingerprint)
    {
        var onnxFiles = Directory.EnumerateFiles(directory, "*.onnx", SearchOption.AllDirectories).ToArray();
        string modelPath;
        if (!string.IsNullOrWhiteSpace(options.Model.ModelFile))
        {
            modelPath = Path.GetFullPath(Path.Combine(directory, options.Model.ModelFile));
            if (!File.Exists(modelPath))
                throw new ModelValidationException($"Configured ONNX model '{options.Model.ModelFile}' does not exist in the snapshot.");
        }
        else
        {
            modelPath = onnxFiles.FirstOrDefault(x => Path.GetFileName(x).Equals("model.onnx", StringComparison.OrdinalIgnoreCase))
                ?? (onnxFiles.Length == 1 ? onnxFiles[0] : throw new ModelValidationException("The model snapshot contains multiple ONNX files. Configure Model.ModelFile explicitly."));
        }

        var tokenizerPath = Directory.EnumerateFiles(directory, "tokenizer.json", SearchOption.AllDirectories).FirstOrDefault()
            ?? throw new ModelValidationException("The model snapshot does not contain tokenizer.json.");
        var (manifestModelId, maxTokens, normalize) = ReadModelMetadata(directory);
        return new ModelSnapshot(manifestModelId ?? modelId, revision, fingerprint, directory, modelPath, tokenizerPath, maxTokens, normalize);
    }
}
