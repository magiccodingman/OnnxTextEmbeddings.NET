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

## Automatic concurrency

Automatic mode resolves to approximately half the configured model thread count, minimum one, capped at eight:

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

The cap is a package default, not an ONNX Runtime hard limit. Explicit positive values remain honored.

## Least-loaded routing

When several model instances are in memory, the package does not fill one model and then move to the next. A single scheduler tracks `ActiveRequests` for every healthy instance and reserves the healthy instance with the lowest active count.

```text
global bounded queue
        ↓
least-loaded healthy scheduler
        ↓
A 3/8
B 2/8  ← next request
C recovering (not eligible)
```

Ties use a rotating cursor. Two idle instances receiving two requests therefore receive one request each.

## Why multiple model instances are not the default performance strategy

Loading another model copy usually costs a lot of RAM and **usually does not produce another proportional throughput gain**. On typical CPU systems, once one session is sufficiently busy, the limiting resource tends to be shared hardware such as memory bandwidth, caches, memory controllers/interconnects, or broader motherboard/platform throughput. The exact limiter varies by CPU and system, but it is often not "we need another model in RAM."

`ModelInstanceCount > 1` therefore exists primarily for:

- experimentation and benchmarking;
- unusual CPU/memory/NUMA topologies;
- future architecture-specific scheduling/affinity work;
- machines where measurements actually show a second session helps.

A possible experimental optimization is to isolate model instances by CPU topology—such as NUMA nodes, CPU groups, or AMD CCD-related layouts—and preserve memory locality so sessions contend less for shared bandwidth/cache. That needs per-platform benchmarking and is not assumed to improve every machine.

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

With multiple instances, new work routes only to healthy copies. With one instance, the global queue waits for the replacement session to become healthy, then resumes. The queue remains bounded; once it is full, producers asynchronously wait instead of creating unbounded memory growth.

## Request retry policy

A recoverable ONNX session/runtime failure may transparently retry the affected request **once** through the global scheduler. Because the bad instance is already out of rotation, the retry goes to another healthy instance when one exists or waits for recovery when it was the only instance.

Memory-pressure failures are different. They quarantine and rebuild the affected instance, but the failed request is not immediately retried on another model copy. This avoids turning one allocation failure into a cascading OOM across every loaded session.

## OOM boundary

The package can recover from failures that leave the .NET process alive. It cannot self-heal after the operating system, container runtime, or cgroup kills the entire process. Configure systemd, Kubernetes, your service manager, or another process supervisor for process-level restart.

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
