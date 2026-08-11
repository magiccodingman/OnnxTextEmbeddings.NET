# Configuration

Configure application-wide defaults during registration:

```csharp
builder.Services.AddOnnxTextEmbeddings(options =>
{
    options.DocumentChunkMaxTokens = 1024;
    options.QueryMaxTokens = 1024;

    options.Inference.ModelInstanceCount = 1;
    options.Inference.ThreadsPerModel = 16;
    options.Inference.ConcurrentRequestsPerModel = 0; // auto: 8 at 16 threads
    options.Inference.QueueCapacity = 256;

    options.Chunking.ChunkOverlapTokens = 0;
    options.Chunking.RepeatHeadingContext = true;

    options.Vectors.DocumentFormat = EmbeddingVectorFormat.Float32;
    options.Vectors.QueryFormat = EmbeddingVectorFormat.Float32;
});
```

## Application defaults vs per-call overrides

`DocumentChunkMaxTokens`, `QueryMaxTokens`, and the vector formats are defaults. Individual calls can override them without creating a second service registration.

```csharp
var chunks = await embeddingService.EmbedDocumentAsync(
    text,
    new EmbeddingRequestOptions
    {
        MaxTokens = 512,
        VectorFormat = EmbeddingVectorFormat.Int8
    });
```

For documents, `MaxTokens` changes the chunk/model-input ceiling for that request. It must be positive and cannot exceed the loaded model's hard token limit.

Queries remain one embedding:

```csharp
var request = new QueryEmbeddingRequestOptions
{
    MaxTokens = 2048,
    VectorFormat = EmbeddingVectorFormat.Float16
};

var count = await embeddingService.CountQueryTokensAsync(queryText, request);
if (count.Fits)
    query = await embeddingService.EmbedQueryAsync(queryText, request);
```

A query override changes only its acceptance ceiling; it never enables query chunking or silent truncation. `FitsModelLimit` still protects the model's actual hard maximum.

## Model instances, threads, and concurrency

`ModelInstanceCount` controls independent ONNX sessions/model copies. Default: one.

`ThreadsPerModel` controls ONNX Runtime intra-op threads for each model instance. Default: 16.

`ConcurrentRequestsPerModel = 0` means automatic:

```text
max(1, min(ThreadsPerModel / 2, 8))
```

At the default 16 threads/model this resolves to eight concurrent calls. Explicit positive values are not silently capped.

Multiple model instances are **not** expected to be a general throughput multiplier. They normally increase RAM use far more reliably than they increase throughput because CPU embedding workloads often become constrained by the shared memory/cache/interconnect/platform subsystem first. Keep one instance unless benchmarks on the target hardware show otherwise; multiple copies mainly exist for experimentation, unusual memory/NUMA layouts, and future CPU-topology-aware affinity work.

See [concurrency.md](concurrency.md).

## Queue capacity

Embedding work enters one bounded global queue. The scheduler dispatches only to healthy model instances with free capacity and uses least-active routing across multiple instances.

If every instance is recovering, queued work waits. Once the queue is full, producers asynchronously wait for queue capacity.

## Chunking

Overlap defaults to zero. If enabled, continuation chunks reuse up to the configured number of source tokens while reserving enough model-input capacity for that overlap.

## Vector formats

Both document and query return formats default to FP32 for maximum interoperability. Applications persisting many vectors should consider INT8 for its roughly 4x smaller payload.

Per-call request options can combine vector and token overrides:

```csharp
var compact = await embeddingService.EmbedDocumentAsync(
    text,
    new EmbeddingRequestOptions
    {
        MaxTokens = 768,
        VectorFormat = EmbeddingVectorFormat.Int8
    });
```

## Runtime health

`ITextEmbeddingService.ModelInfo` includes aggregate health plus `Instances`, a point-in-time snapshot containing health state, active/max requests, generation, total recoveries, current recovery attempt, and last failure text.

## Search

DefaultV1 values:

```text
MinimumLengthConfidence = 0.96
SupportWindow            = 0.12
SecondSupportWeight      = 0.25
ThirdSupportWeight       = 0.10
```

## Initialization

`WarmupOnStartup` defaults to true. `BlockHostStartupUntilReady` defaults to false. Call `WaitUntilReadyAsync` explicitly when a non-hosted application needs a readiness barrier.
