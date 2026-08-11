# Performance

The package is CPU-first and optimized around small local embedding models.

## Worker model

Each worker owns an ONNX Runtime session. One worker is the default because a single CPU session can already use multiple intra-op threads. Increase workers only when measured concurrency improves throughput on the target machine.

## Threading

`ThreadsPerWorker = 0` selects an automatic budget, capped by `MaximumAutoThreadsPerWorker` (default 12). A useful tuning rule is to increase either workers or per-worker threads first, not both simultaneously.

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

INT4 and INT8 include small per-vector quantization metadata in addition to these payload bytes.

## Search

Search over a precomputed `QueryEmbedding` performs no ONNX inference. For larger PostgreSQL-backed datasets, use SQL-side pgvector candidate preselection, then DefaultV1 final ranking on a bounded candidate set.

## Measure your machine

Wall-clock thresholds are intentionally not baked into unit tests. CPU architecture, core count, memory bandwidth, ONNX Runtime version, model precision, and thread budgets all materially affect inference latency.
