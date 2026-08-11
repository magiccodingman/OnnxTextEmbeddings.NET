# Troubleshooting

## First request takes a long time

The default model is downloaded and ONNX Runtime is initialized on first use. Reuse the cache between runs. In hosted applications, keep `WarmupOnStartup = true` so initialization begins with the host instead of the first user request.

## `QueryTokenLimitExceededException`

Queries are never silently truncated or chunked. Use `CountQueryTokensAsync` to inspect `InputTokenCount`, limits, and `Fits` before embedding.

If a particular operation legitimately needs a different ceiling and the model supports it:

```csharp
var request = new QueryEmbeddingRequestOptions { MaxTokens = 2048 };
var count = await embeddingService.CountQueryTokensAsync(text, request);
if (count.Fits)
    query = await embeddingService.EmbedQueryAsync(text, request);
```

## Need different document chunk sizes for different data

Use a per-call request instead of changing the singleton's global options:

```csharp
var chunks = await embeddingService.EmbedDocumentAsync(
    text,
    new EmbeddingRequestOptions { MaxTokens = 512 });
```

The override is reflected in the chunk's stored historical token capacity.

## A model instance is recovering

Inspect:

```csharp
embeddingService.ModelInfo?.Instances
```

A recoverable runtime/session fault puts only the affected instance into `Draining`/`Recovering`/`Faulted` state. Other healthy instances continue serving traffic. The old session is disposed only after active calls drain, and the instance returns to `Healthy` only after a fresh session loads successfully.

Repeated replacement-load failures use backoff; the library does not assume the session repaired itself merely because time passed.

## All model instances are temporarily unavailable

The global bounded queue waits for at least one instance to recover. Once its configured queue capacity is reached, new producers asynchronously wait for queue space. This is intentional backpressure rather than an unbounded memory queue.

## The whole process was OOM-killed

In-process model recovery works only while the .NET process remains alive. If Linux OOM killer, a container/cgroup, Windows, or a service host terminates the whole process, use process-level supervision (systemd, Kubernetes, Windows Service recovery, etc.) to restart it.

## `EmbeddingSpaceMismatchException`

The query fingerprint and stored document fingerprint differ. Regenerate persisted embeddings with the active model space; equal dimensions alone do not make vectors compatible.

## Repository has multiple `.onnx` files

```csharp
options.Model.ModelFile = "model-int8.onnx";
```

## Download fails but old model exists

During an update, the active runtime remains in service if the candidate fails. Inspect `ITextEmbeddingService.Status.LastError` and logs for the download/validation failure.

## Too much CPU contention

Start by reducing `ConcurrentRequestsPerModel` or `ThreadsPerModel`. Automatic defaults are 5 concurrent calls/model for Jasper INT8 and 4 for the global/custom profile at the normal 16 threads.

If multiple model instances were explicitly configured, reducing `ModelInstanceCount` also reduces CPU and model-memory pressure. Do not add model instances merely to gain ordinary request concurrency; one session already supports concurrent calls.

## Search looks wrong after a deployment

Confirm model fingerprint/revision, field weights, stored token metadata, and application-side filtering. Inspect `SemanticSearchResult.Fields` and `SemanticChunkMatch` raw/adjusted scores rather than debugging only the final item score.
