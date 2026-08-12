# Concurrency, load balancing, and recovery

OnnxTextEmbeddings.NET keeps its public embedding-oriented inference options and diagnostics, but the generic session-hosting machinery is now supplied by the reusable `OnnxModelRuntime.NET` NuGet package.

The embedding package maps its existing options directly into `OnnxModelRuntimeOptions`, so this extraction is an ownership refactor rather than a new concurrency model.

## Three separate controls

```csharp
options.Inference.ModelInstanceCount = 1;
options.Inference.ThreadsPerModel = 16;
options.Inference.ConcurrentRequestsPerModel = 0; // automatic
```

- `ModelInstanceCount` — independent ONNX sessions/model copies in memory.
- `ThreadsPerModel` — ONNX Runtime intra-op thread count for each session.
- `ConcurrentRequestsPerModel` — simultaneous inference calls allowed against each session.

## What OnnxTextEmbeddings still owns

The local `EmbeddingOnnxExecutor` owns the model-specific tensor contract, output-shape validation, mean pooling when required, normalization, and embedding dimension observation.

It does **not** own queueing, scheduling, session lifecycle, or recovery.

## What OnnxModelRuntime owns

`OnnxModelRuntime.NET` owns:

```text
InferenceSession creation/disposal
bounded global queue + backpressure
per-instance request limits
least-loaded scheduling + fair tie rotation
health/draining/recovery/generation tracking
recoverable-failure retry policy
memory-pressure isolation
runtime diagnostics
async shutdown
```

## Automatic concurrency

Automatic mode remains:

```text
max(1, min(ThreadsPerModel / 2, 8))
```

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

The cap is a package policy rather than an ONNX Runtime limitation. Explicit positive values are honored.

When `ThreadsPerModel = 0`, the shared runtime also resolves the hardware-based thread count using `MaximumAutoThreadsPerModel` and configured model-instance count.

## Least-loaded routing

```text
global bounded queue
        ↓
least-loaded healthy scheduler
        ↓
A 3/8
B 2/8  ← next request
C recovering (not eligible)
```

A single scheduler owns reservations; native execution runs independently after reservation. Equal loads use a rotating tie cursor.

## Why multiple copies are not the default performance strategy

Loading another model copy costs RAM and usually does not create a proportional throughput gain on an already-busy CPU. Shared memory bandwidth/cache/interconnect/platform throughput frequently becomes limiting before model-instance count does.

`ModelInstanceCount > 1` therefore remains useful for experiments, unusual CPU/NUMA topologies, redundancy/recovery behavior, and systems where benchmarks prove another copy helps.

## Failure lifecycle

A recoverable model-instance failure removes only that instance from scheduling:

```text
Healthy
  ↓ recoverable runtime/session failure
Draining
  ↓ active calls complete
Recovering
  ↓ old instance disposed / fresh instance created
Healthy (generation + 1)
```

Failed recreations remain unavailable and retry with bounded backoff. Other healthy instances keep serving. With no healthy instance, the bounded queue waits for recovery instead of inventing capacity.

A normal recoverable runtime failure can retry the affected request at most once. Memory-pressure failures rebuild/quarantine the affected instance but do not immediately send the same request to another loaded copy, reducing cascading OOM risk.

Model-specific validation/application exceptions are not classified as infrastructure failures and therefore do not cause expensive session reconstruction.

## Public diagnostics compatibility

`ITextEmbeddingService.ModelInfo` still exposes the embedding package's existing `ModelRuntimeInfo` / `ModelInstanceRuntimeInfo` records. They are projections of the generic runtime diagnostics so consumers are not forced to reference OnnxModelRuntime.NET types in their application API.

```csharp
var runtime = embeddingService.ModelInfo;
Console.WriteLine(runtime?.HealthyModelInstanceCount);
Console.WriteLine(runtime?.RecoveringModelInstanceCount);
Console.WriteLine(runtime?.ActiveRequests);
```

## OOM/process boundary

The runtime can recover only while the .NET process remains alive. An OS/container/cgroup process kill still requires systemd, Kubernetes, or another process supervisor to restart the application.

## Compatibility aliases

The early `WorkerCount`, `ThreadsPerWorker`, and `MaximumAutoThreadsPerWorker` properties remain obsolete aliases. New code should use the model-oriented names.
