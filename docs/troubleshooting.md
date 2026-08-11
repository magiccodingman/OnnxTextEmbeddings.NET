# Troubleshooting

## First request takes a long time

The default model is downloaded and ONNX Runtime is initialized on first use. Reuse the cache between runs. In hosted applications, keep `WarmupOnStartup = true` so initialization begins with the host instead of the first user request.

## `QueryTokenLimitExceededException`

Queries are intentionally never silently truncated/chunked. Shorten the query or deliberately preprocess it in application code. `QueryMaxTokens` can be increased only up to the model's supported maximum.

## `EmbeddingSpaceMismatchException`

The query fingerprint and stored document fingerprint differ. Regenerate persisted embeddings with the active model space; do not disable the check merely because dimensions happen to match.

## Repository has multiple `.onnx` files

Set:

```csharp
options.Model.ModelFile = "model-int8.onnx";
```

so selection is explicit.

## Download fails but old model exists

During an update, the active runtime remains in service if the candidate fails. Inspect `ITextEmbeddingService.Status.LastError` and logs for the download/validation failure.

## Too much CPU contention

Reduce `ThreadsPerWorker`, `MaximumAutoThreadsPerWorker`, or `WorkerCount`. Multiple workers each own an ONNX session, so `workers × threads` can oversubscribe a CPU quickly.

## Search looks wrong after a deployment

Confirm model fingerprint/revision, field weights, stored token metadata, and whether application-side filtering changed. Inspect `SemanticSearchResult.Fields` and `SemanticChunkMatch` raw/adjusted scores rather than debugging only the final item score.
