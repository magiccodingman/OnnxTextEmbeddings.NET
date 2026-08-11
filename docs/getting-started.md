# Getting started

## 1. Install and register

```bash
dotnet add package OnnxTextEmbeddings.NET
```

```csharp
builder.Services.AddOnnxTextEmbeddings();
```

The default registration uses Jasper INT8 on CPU, **one model instance with 16 threads and up to 8 automatic concurrent inference calls**, 1024-token document chunks, a 1024-token query ceiling, and FP32 returned vectors for maximum interoperability.

For applications that persist many embeddings, INT8 is the recommended compact default:

```csharp
builder.Services.AddOnnxTextEmbeddings(options =>
{
    options.Vectors.DocumentFormat = EmbeddingVectorFormat.Int8;
});
```

## 2. First model download

The model is not bundled in the NuGet package. The first initialization resolves `magiccodingman/Jasper-Token-Compression-600M-ONNX-INT8`, downloads the ONNX/tokenizer assets, creates a cache snapshot, and loads ONNX Runtime.

## 3. Embed a document

```csharp
var embeddings = await embeddingService.EmbedDocumentAsync(markdown); // FP32 by default
```

A short document yields one embedding. Larger documents yield several `TextEmbedding` records with exact source ranges, token ranges, chunk index/count, heading path, historical token capacity, model revision, and embedding-space fingerprint.

Choose token ceiling and return representation for one call without changing the global defaults:

```csharp
var compact = await embeddingService.EmbedDocumentAsync(
    markdown,
    new EmbeddingRequestOptions
    {
        MaxTokens = 512,
        VectorFormat = EmbeddingVectorFormat.Int8
    });
```

The built-in service chunks using the requested limit and encodes the requested format directly from the original FP32 ONNX output. A per-call document limit cannot exceed the loaded model's hard maximum.

## 4. Count and embed a query

A query is always one vector. Check its token budget without exception-driven validation:

```csharp
var request = new QueryEmbeddingRequestOptions { MaxTokens = 2048 };
var count = await embeddingService.CountQueryTokensAsync(userQuery, request);
if (!count.Fits) return;

var query = await embeddingService.EmbedQueryAsync(userQuery, request);
```

A query request can raise or lower the application default ceiling, but never bypasses the model's actual maximum. It never silently truncates or chunks the query.

## 5. Convert existing FP32 vectors

If you already have float32 values, the vector helper can preserve or compress them:

```csharp
EmbeddingVector fp32 = EmbeddingVector.FromFloat32(values); // stays FP32
EmbeddingVector fp16 = EmbeddingVector.FromFloat32(values, EmbeddingVectorFormat.Float16);
EmbeddingVector int8 = EmbeddingVector.FromFloat32(values, EmbeddingVectorFormat.Int8);
EmbeddingVector int4 = EmbeddingVector.FromFloat32(values, EmbeddingVectorFormat.Int4);
```

`ConvertTo(...)` can also change an existing `EmbeddingVector`. Converting a previously quantized vector back to a higher-precision representation does **not** restore information that quantization already discarded.

## 6. Search in memory

```csharp
var results = await semanticSearch.SearchAsync(
    query,
    documents,
    document => document.Embeddings,
    new SemanticSearchRequest { Top = 10 });
```

## 7. Tune concurrency only if needed

Automatic request concurrency is `ThreadsPerModel / 2`, capped at eight, so the normal 16-thread configuration exposes eight request slots on one model copy.

Only increase `ModelInstanceCount` if you are deliberately experimenting or benchmarks on the target hardware show a benefit. Extra model copies usually consume more RAM without increasing aggregate CPU throughput because they still contend for the same memory/cache/platform resources. The multi-instance scheduler is there for unusual setups, recovery, and future topology-aware experiments—not because more copies are normally faster.

See [concurrency.md](concurrency.md).

## 8. Persist embeddings

Store the complete `TextEmbedding`, not only raw vector bytes. The fingerprint and source metadata are intentionally part of the persistence contract. See [vector-formats.md](vector-formats.md) for size/fidelity choices.
