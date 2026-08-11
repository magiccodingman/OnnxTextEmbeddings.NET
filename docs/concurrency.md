# Concurrency, load balancing, and recovery

ONNX Runtime can execute multiple inference calls concurrently against the same `InferenceSession`. OnnxTextEmbeddings.NET uses that capability so ordinary request concurrency does not require another copy of the model in memory.

## Three separate controls

```csharp
options.Inference.ModelInstanceCount = 1;
options.Inference.ThreadsPerModel = 16;
options.Inference.ConcurrentRequestsPerModel = 0; // automatic
```

- `ModelInstanceCount` — independent ONNX sessions/model copies in memory.
- `ThreadsPerModel` — ONNX Runtime intra-op thread count for each session.
- `ConcurrentRequestsPerModel` — simultaneous `Run()` calls allowed against each session.

## Automatic concurrency profiles

Automatic mode first computes approximately half the configured thread count, with a minimum of one, then applies the model profile cap.

```text
automatic = min(max(ThreadsPerModel / 2, 1), profileCap)
```

Profile caps:

| Model selection | Automatic cap |
|---|---:|
| Jasper INT8 | **5** |
| Jasper INT4 | 4 |
| Jasper FP32 | 4 |
| Custom Hugging Face/local/HTTP model | 4 |

At 16 threads/model the defaults therefore resolve to 5 for the built-in Jasper INT8 preset and 4 for the others. These values come from observed CPU throughput/value behavior rather than an ONNX Runtime hard limit.

Explicit positive values remain honored:

```csharp
options.Inference.ConcurrentRequestsPerModel = 7;
```

## Least-loaded routing

When several model instances are in memory, the package does not fill one model and then move to the next. A single scheduler tracks `ActiveRequests` for every healthy instance and reserves the healthy instance with the lowest active count.

```text
global bounded queue
        ↓
least-loaded healthy scheduler
        ↓
A 3/5
B 2/5  ← next request
C recovering (not eligible)
```

Ties use a rotating cursor. Therefore:

```text
A 0/5    B 0/5
request 1 → A
A 1/5    B 0/5
request 2 → B
```

This preserves even utilization when instances are equally available while still preferring whichever instance becomes less loaded under uneven execution times.

## Slot accounting

A request slot is reserved before inference and released in a `finally` path. Cancellation and failures therefore cannot permanently consume one of the instance's concurrency slots.

The public `ModelInfo.Instances` diagnostics expose each instance's current active/max request count.

## Instance health lifecycle

Each session/model copy has an explicit lifecycle:

```text
Healthy
   ↓ runtime/session failure
Draining
   ↓ active requests reach zero
Recovering
   ↓ old session disposed
create fresh InferenceSession
   ↓ success
Healthy (new generation)
```

If replacement creation fails, the instance stays out of rotation and retries with bounded exponential backoff. It is never marked healthy merely because time passed.

`Generation` increases only after a fresh session is successfully created. `TotalRecoveries`, `RecoveryAttempts`, and `LastFailure` are exposed for diagnostics.

## Traffic during recovery

With multiple instances:

```text
A recovering
B healthy
```

new work routes only to B, subject to B's normal concurrency limit.

With only one instance:

```text
global queue
   ↓
A recovering
   ↓
queue waits
   ↓
A generation 2 healthy
   ↓
queue resumes
```

The queue remains bounded. Once its capacity is full, producers asynchronously wait instead of causing unbounded memory growth.

## Request retry policy

A recoverable ONNX session/runtime failure may transparently retry the affected request **once** through the global scheduler. Because the bad instance is already out of rotation, the retry goes to another healthy instance when one exists or waits for recovery when it was the only instance.

Memory-pressure failures are different. They quarantine and rebuild the affected instance, but the failed request is not immediately retried on another model copy. This avoids turning one allocation failure into a cascading OOM across every loaded session.

## OOM boundary

The package can recover from failures that leave the .NET process alive, including model/session allocation failures surfaced to managed code. It cannot self-heal after the operating system, container runtime, or cgroup kills the entire process. Configure systemd, Kubernetes, your service manager, or another process supervisor for process-level restart.

## Runtime diagnostics

```csharp
var runtime = embeddingService.ModelInfo;

Console.WriteLine(runtime?.HealthyModelInstanceCount);
Console.WriteLine(runtime?.RecoveringModelInstanceCount);
Console.WriteLine(runtime?.ActiveRequests);

foreach (var instance in runtime?.Instances ?? [])
{
    Console.WriteLine($"{instance.Index}: {instance.Health} " +
                      $"{instance.ActiveRequests}/{instance.MaxConcurrentRequests} " +
                      $"generation {instance.Generation}");
}
```

## `ORT_SEQUENTIAL` is retained

The session uses ONNX Runtime's sequential graph execution mode plus the configured intra-op thread count. That graph scheduling choice does not prohibit concurrent callers from invoking `Run()` against the same session.

## Compatibility names

The early `WorkerCount`, `ThreadsPerWorker`, and `MaximumAutoThreadsPerWorker` properties remain obsolete aliases for source compatibility. New code should use the model-oriented names.
