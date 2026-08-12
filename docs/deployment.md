# Deployment

## Online/default deployment

Ship the small NuGet package. On first initialization the configured Hugging Face model is downloaded to the local cache. Subsequent starts reuse the valid snapshot and apply the configured update policy.

## Offline deployment

Provision model assets as part of the application image/host and use:

```csharp
options.Model.UseLocalDirectory("/opt/models/jasper");
options.Model.UpdatePolicy = ModelUpdatePolicy.Never;
```

No Python, Git, Git LFS, Hugging Face CLI, GPU driver, or model server is required.

## Containers

Mount the cache on a persistent volume when container recreation should not redownload hundreds of megabytes. For immutable images, bake the model directory separately and use the local source.

## Multi-instance hosts

The cache uses cross-process locking so multiple application instances sharing a cache do not activate partial downloads concurrently. If instances do not share a filesystem, give each instance its own cache or provision a common local model directory.

## Windows

Model swaps dispose previous ONNX sessions before deleting their snapshot directory. Cleanup retries locked files because Windows prevents deletion of open model files more aggressively than Unix-like systems.

## Release publishing

Repository releases are produced from the `release` branch by `.github/workflows/publish-nuget.yml`. The publish job uses the GitHub environment `release` and NuGet Trusted Publishing via OIDC; it expects the environment secret `NUGET_USER`.
