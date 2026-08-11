# Configuration

Configure the singleton during registration:

```csharp
builder.Services.AddOnnxTextEmbeddings(options =>
{
    options.DocumentChunkMaxTokens = 1024;
    options.QueryMaxTokens = 1024;
    options.Inference.WorkerCount = 1;
    options.Inference.ThreadsPerWorker = 0; // auto
    options.Inference.MaximumAutoThreadsPerWorker = 12;
    options.Inference.QueueCapacity = 256;
    options.Chunking.ChunkOverlapTokens = 0;
    options.Chunking.RepeatHeadingContext = true;
    options.Vectors.DocumentFormat = EmbeddingVectorFormat.Int8;
    options.Vectors.QueryFormat = EmbeddingVectorFormat.Float32;
});
```

## Token ceilings

`DocumentChunkMaxTokens` controls finalized document model input. `QueryMaxTokens` controls the single query vector. Both are validated against a model maximum when the snapshot exposes one.

## Workers and threads

`WorkerCount` controls independent ONNX sessions. `ThreadsPerWorker = 0` means automatic. The automatic thread budget is capped by `MaximumAutoThreadsPerWorker`. Avoid multiplying workers and per-worker thread counts blindly: throughput can fall once sessions compete for the same physical cores.

## Queue capacity

Embedding work enters a bounded channel. A finite queue provides backpressure rather than allowing an unbounded burst of strings to become unbounded memory growth.

## Chunking

Overlap defaults to zero. If enabled, continuation chunks reuse up to the configured number of source tokens while reserving enough model-input capacity for that overlap. Markdown overlap stays inside the same structural section so text from one heading is not mislabeled as context for another.

## Vector formats

Model precision and stored-vector format are independent. `EmbeddingVectorFormat.Int4` is packed symmetric per-vector quantization; INT8 is the default document representation. Queries default to FP32 because only one query vector is normally live at a time.

## Search

DefaultV1 values:

```text
MinimumLengthConfidence = 0.96
SupportWindow            = 0.12
SecondSupportWeight      = 0.25
ThirdSupportWeight       = 0.10
```

See [semantic-scoring.md](semantic-scoring.md) for the formula.

## Initialization

`WarmupOnStartup` defaults to true. `BlockHostStartupUntilReady` defaults to false. Call `WaitUntilReadyAsync` explicitly when a non-hosted application needs a readiness barrier.
