# Configuration

Configure the singleton during registration:

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

    // Float32 is the interoperability-first default. INT8 is recommended for compact storage.
    options.Vectors.DocumentFormat = EmbeddingVectorFormat.Float32;
    options.Vectors.QueryFormat = EmbeddingVectorFormat.Float32;
});
```

## Token ceilings

`DocumentChunkMaxTokens` controls each finalized document chunk input. `QueryMaxTokens` controls the single query vector. Both default to 1024 and remain **per-request limits**, regardless of how many requests execute concurrently.

## Token counting without limit exceptions

```csharp
int sourceTokens = await embeddingService.CountTokensAsync(text);
QueryTokenCount count = await embeddingService.CountQueryTokensAsync(text);
```

`CountTokensAsync` returns the source tokenizer count. `CountQueryTokensAsync` additionally evaluates the final model input and returns `InputTokenCount`, `QueryMaxTokens`, `ModelMaxTokens`, `FitsConfiguredLimit`, `FitsModelLimit`, and `Fits`. Counting an oversized query does not throw merely because it exceeds the configured maximum.

`EmbedQueryAsync` intentionally continues to throw `QueryTokenLimitExceededException` when `Fits` would be false.

## Model instances, threads, and concurrency

`ModelInstanceCount` controls independent ONNX sessions/model copies. The default is one.

`ThreadsPerModel` controls ONNX Runtime intra-op threads for each model instance. The default is 16. Set it to zero only when hardware-based automatic resolution is desired; `MaximumAutoThreadsPerModel` bounds that automatic value.

`ConcurrentRequestsPerModel = 0` means automatic:

```text
min(ThreadsPerModel / 2, 8), minimum 1
```

With defaults, one model instance services up to eight concurrent inference calls. Explicit positive concurrency values are honored, though 8 is the recommended practical maximum.

See [concurrency.md](concurrency.md).

## Queue capacity

Embedding work enters a bounded channel. A finite queue provides backpressure rather than allowing an unbounded burst of strings to become unbounded memory growth.

## Chunking

Overlap defaults to zero. If enabled, continuation chunks reuse up to the configured number of source tokens while reserving enough model-input capacity for that overlap.

## Vector formats

Both document and query return formats default to FP32 for maximum interoperability. Applications that persist many vectors should strongly consider INT8 for its roughly 4x smaller vector payload.

The configured values are only defaults. Any embedding call can select a format dynamically:

```csharp
var compact = await embeddingService.EmbedDocumentAsync(text, EmbeddingVectorFormat.Int8);
var tiny = await embeddingService.EmbedDocumentAsync(text, EmbeddingVectorFormat.Int4);
var full = await embeddingService.EmbedDocumentAsync(text, EmbeddingVectorFormat.Float32);
```

See [vector-formats.md](vector-formats.md).

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
