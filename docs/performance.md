# Performance

The package is CPU-first and optimized around small local embedding models.

## Shared-session concurrency

A single ONNX Runtime session can execute multiple inference requests concurrently. The package therefore separates request concurrency from model replication.

Defaults:

```text
ModelInstanceCount             1
ThreadsPerModel               16
ConcurrentRequestsPerModel    Auto → 8
```

## Automatic concurrency

When concurrency is left at zero, it resolves to half the model's thread count with a cap of eight and a minimum of one:

```text
max(1, min(ThreadsPerModel / 2, 8))
```

This gives 2 concurrent requests at 4 threads, 4 at 8 threads, 6 at 12 threads, and 8 at 16 or more threads.

Explicit values are allowed above eight; the automatic cap is simply the package's conservative default boundary.

## Multiple model instances

`ModelInstanceCount > 1` creates additional independent ONNX sessions/model copies. It is supported, load-balanced, and self-healing—but it is **not expected to increase throughput in the normal case**.

The common CPU bottleneck after a session is sufficiently busy is shared platform throughput: memory bandwidth, caches, memory controllers/interconnects, or another part of the CPU-to-RAM path. Adding another full model copy can therefore consume substantially more RAM while competing for the same limiting resource.

Use multiple model instances for experimental benchmarking, unusual CPU/NUMA/memory layouts, or machines where measurements prove a benefit. A future/experimental optimization could isolate model sessions by CPU topology (for example NUMA nodes, CPU groups, or AMD CCD-related layouts), bind their threads, and preserve memory locality so they interfere less with one another. Whether that works is hardware-specific and should be measured rather than assumed.

When multiple copies are enabled, least-loaded routing makes the best use of the capacity you chose to provision:

```text
A 3/8
B 2/8  ← next request
```

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

CPU architecture, core count, memory bandwidth, model precision, thread budgets, and request concurrency materially affect throughput. Treat the defaults as a strong starting point and benchmark before adding model instances or overriding concurrency.
