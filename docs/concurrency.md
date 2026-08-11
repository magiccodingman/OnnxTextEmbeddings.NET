# Concurrency and threading

ONNX Runtime can execute multiple inference calls concurrently against the same `InferenceSession`. OnnxTextEmbeddings.NET uses that capability so ordinary request concurrency does not require another copy of the model in memory.

## Three separate controls

```csharp
options.Inference.ModelInstanceCount = 1;
options.Inference.ThreadsPerModel = 16;
options.Inference.ConcurrentRequestsPerModel = 0; // automatic
```

These mean different things:

- `ModelInstanceCount` — independent ONNX sessions/model copies in memory.
- `ThreadsPerModel` — ONNX Runtime intra-op thread count for each session.
- `ConcurrentRequestsPerModel` — simultaneous `Run()` calls allowed against each session.

## Defaults

The package defaults to one model instance and 16 threads per model.

When `ConcurrentRequestsPerModel` is zero, the effective concurrency is:

```text
max(1, min(ThreadsPerModel / 2, 8))
```

Examples:

| Threads/model | Automatic concurrent requests/model |
|---:|---:|
| 1 | 1 |
| 2 | 1 |
| 4 | 2 |
| 8 | 4 |
| 12 | 6 |
| 16 | 8 |
| 24 | 8 |
| 32 | 8 |

The automatic cap is 8. Explicit positive values are honored rather than silently capped, but **8 is the documented recommended maximum** because practical benchmarks showed little or no additional gain beyond eight simultaneous requests per model instance.

## Memory behavior

Default:

```text
1 model copy
16 intra-op threads
8 concurrent requests
```

Setting only `ConcurrentRequestsPerModel` does not load another model.

Setting:

```csharp
options.Inference.ModelInstanceCount = 2;
options.Inference.ConcurrentRequestsPerModel = 8;
```

creates two independent ONNX sessions/model instances and permits up to 16 total in-flight inference calls. This intentionally consumes more memory and should be used only when benchmarks show that another session helps.

## Token limits remain per request

Concurrency does not change token limits. With the defaults, eight requests may run concurrently, and **each request independently has a 1024-token document/query ceiling** unless the application configures a different supported limit.

## `ORT_SEQUENTIAL` is retained

The session uses ONNX Runtime's sequential graph execution mode plus the configured intra-op thread count. This does not prohibit concurrent callers from invoking `Run()` on the same session; graph scheduling mode and request-level concurrency are different concerns.

## Queueing

Requests enter one bounded channel. For every model instance, the package starts the resolved number of concurrent dispatchers, all sharing that model's session. The shared queue naturally load-balances work across available inference slots and model instances.

## Compatibility names

The early `WorkerCount`, `ThreadsPerWorker`, and `MaximumAutoThreadsPerWorker` properties remain as obsolete aliases for source compatibility. New code should use the model-oriented names above because a "worker" no longer maps one-to-one to a model instance or a single concurrent request.
