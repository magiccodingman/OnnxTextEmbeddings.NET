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

The default uses one model copy in memory and allows eight simultaneous requests against it.

## Automatic concurrency

When concurrency is left at zero, it resolves to half the model's thread count with a cap of eight and a minimum of one. This gives 2 concurrent requests at 4 threads, 4 at 8 threads, 6 at 12 threads, and 8 at 16 or more threads.

Explicit values are allowed above eight, but **8 concurrent requests per model is the recommended maximum**. Personal benchmark results showed little or no additional throughput benefit beyond that point, so the library does not automatically push higher.

## Multiple model instances

`ModelInstanceCount > 1` creates additional independent ONNX sessions/model copies. This can increase memory substantially and is no longer required simply to support concurrent callers. Increase it only when a target machine's benchmarks show a benefit.

## Token limits

Concurrency never changes the request budget. `DocumentChunkMaxTokens = 1024` and `QueryMaxTokens = 1024` are independent per-request defaults. Eight concurrent queries may each independently use up to their configured query maximum.

## Bounded queue

Inference requests use a bounded channel (`QueueCapacity = 256` by default). This prevents burst traffic from becoming unbounded allocations.

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

CPU architecture, core count, memory bandwidth, model precision, thread budgets, and request concurrency materially affect throughput. Treat the defaults as a strong general-purpose starting point and benchmark before adding model instances or pushing concurrency above eight.
