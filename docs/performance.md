# Performance

The package is CPU-first and optimized around small local embedding models.

## Shared-session concurrency

A single ONNX Runtime session can execute multiple inference requests concurrently. The package therefore separates request concurrency from model replication.

Default topology:

```text
ModelInstanceCount             1
ThreadsPerModel               16
ConcurrentRequestsPerModel    Auto
```

At 16 threads/model, automatic concurrency resolves to:

```text
Jasper INT8    5
Jasper INT4    4
Jasper FP32    4
custom model   4
```

These are benchmark-derived default caps, not ONNX Runtime hard limits.

## Automatic concurrency

Automatic mode starts from approximately half the configured model thread count, minimum one, then applies the selected model profile cap:

```text
min(max(ThreadsPerModel / 2, 1), profileCap)
```

For the global/custom profile, 4 threads resolves to 2 requests and 8 or more threads reaches the cap of 4. Jasper INT8 has a separate cap of 5 because benchmarking showed that fifth concurrent request still provided useful aggregate throughput, while the deal degraded materially beyond it.

Explicit positive `ConcurrentRequestsPerModel` values remain allowed and are not silently clamped. Benchmark your own CPU/model combination before overriding the defaults.

## Multiple model instances

`ModelInstanceCount > 1` creates additional independent ONNX sessions/model copies. This increases memory substantially but can increase aggregate throughput after a single session reaches its practical scaling point.

Multiple instances use explicit least-loaded routing:

```text
A 3/5
B 2/5  ← next request
```

When equally loaded, routing rotates between instances. This means two idle instances and two new requests are intentionally spread one request to each rather than filling the first instance.

A failed instance is removed from rotation while it drains and rebuilds; healthy instances continue receiving traffic. See [concurrency.md](concurrency.md).

## Token limits

Concurrency never changes request budgets. `DocumentChunkMaxTokens = 1024` and `QueryMaxTokens = 1024` are independent per-request defaults.

Those defaults can be overridden for individual calls:

```csharp
var chunks = await embeddingService.EmbedDocumentAsync(
    text,
    new EmbeddingRequestOptions { MaxTokens = 512 });

var query = await embeddingService.EmbedQueryAsync(
    queryText,
    new QueryEmbeddingRequestOptions { MaxTokens = 2048 });
```

Document overrides change chunk size. Query overrides change the acceptance ceiling only; queries still never chunk and can never exceed the loaded model's hard token limit.

## Bounded queue

Inference requests use one bounded global channel (`QueueCapacity = 256` by default). If all healthy capacity is busy—or every model instance is temporarily recovering—work waits in that bounded queue. Once the queue fills, producers asynchronously wait instead of causing unbounded allocations.

## Vector size

For a 2048-dimensional embedding, raw vector payloads are approximately:

```text
INT4   1 KiB
INT8   2 KiB
FP16   4 KiB
FP32   8 KiB
```

## Search

Search over a precomputed `QueryEmbedding` performs no ONNX inference. For larger PostgreSQL-backed datasets, use SQL-side pgvector candidate preselection, then DefaultV1 final ranking on a bounded candidate set.

## Measure your machine

CPU architecture, core count, memory bandwidth, model precision, thread budgets, and request concurrency materially affect throughput. Treat the built-in profiles as strong starting points and benchmark before adding model instances or overriding concurrency.
