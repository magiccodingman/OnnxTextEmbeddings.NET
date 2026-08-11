# Troubleshooting

## First request takes a long time

The default model is downloaded and ONNX Runtime is initialized on first use. Reuse the cache between runs. In hosted applications, keep `WarmupOnStartup = true` so initialization begins with the host instead of the first user request.

## `QueryTokenLimitExceededException`

Queries are intentionally never silently truncated/chunked. Use `CountQueryTokensAsync` to inspect `InputTokenCount`, limits, and `Fits` before embedding when you want validation without exception-driven control flow.

## `EmbeddingSpaceMismatchException`

The query fingerprint and stored document fingerprint differ. Regenerate persisted embeddings with the active model space; do not disable the check merely because dimensions happen to match.

## Repository has multiple `.onnx` files

Set:

```csharp
options.Model.ModelFile = "model-int8.onnx";
```

## Download fails but old model exists

During an update, the active runtime remains in service if the candidate fails. Inspect `ITextEmbeddingService.Status.LastError` and logs for the download/validation failure.

## Too much CPU contention

Start by reducing `ConcurrentRequestsPerModel` or `ThreadsPerModel`. The default is 16 threads and automatic concurrency of 8. If multiple model instances were explicitly configured, reducing `ModelInstanceCount` also reduces CPU and model-memory pressure.

Do not add model instances merely to gain ordinary request concurrency; one session already supports concurrent calls.

## Search looks wrong after a deployment

Confirm model fingerprint/revision, field weights, stored token metadata, and whether application-side filtering changed. Inspect `SemanticSearchResult.Fields` and `SemanticChunkMatch` raw/adjusted scores rather than debugging only the final item score.
