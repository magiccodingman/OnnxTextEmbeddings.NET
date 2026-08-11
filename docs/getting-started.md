# Getting started

## 1. Install and register

```bash
dotnet add package OnnxTextEmbeddings.NET
```

```csharp
builder.Services.AddOnnxTextEmbeddings();
```

The default registration uses Jasper INT8 on CPU, **one model instance with 16 threads and up to 8 concurrent inference calls**, 1024-token document chunks, a 1024-token query ceiling, INT8 document-vector storage, and FP32 query-vector storage.

## 2. First model download

The model is not bundled in the NuGet package. The first initialization resolves `magiccodingman/Jasper-Token-Compression-600M-ONNX-INT8`, downloads the ONNX/tokenizer assets, creates a cache snapshot, and loads ONNX Runtime.

## 3. Embed a document

```csharp
var embeddings = await embeddingService.EmbedDocumentAsync(markdown);
```

A short document yields one embedding. Larger documents yield several `TextEmbedding` records with exact source ranges, token ranges, chunk index/count, heading path, historical token capacity, model revision, and embedding-space fingerprint.

## 4. Count and embed a query

A query is always one vector. Check its token budget without exception-driven validation:

```csharp
var count = await embeddingService.CountQueryTokensAsync(userQuery);

if (!count.Fits)
{
    Console.WriteLine($"Query uses {count.InputTokenCount} of {count.QueryMaxTokens} configured tokens.");
    return;
}

var query = await embeddingService.EmbedQueryAsync(userQuery);
```

For a plain source count:

```csharp
int tokens = await embeddingService.CountTokensAsync(text);
```

`EmbedQueryAsync` still throws `QueryTokenLimitExceededException` if an oversized query is submitted directly; it never silently truncates or chunks it.

## 5. Search in memory

```csharp
var results = await semanticSearch.SearchAsync(
    query,
    documents,
    document => document.Embeddings,
    new SemanticSearchRequest { Top = 10 });
```

## 6. Tune concurrency only if needed

Defaults already allow eight simultaneous inference calls against the one model instance:

```csharp
builder.Services.AddOnnxTextEmbeddings(options =>
{
    options.Inference.ThreadsPerModel = 16;
    options.Inference.ConcurrentRequestsPerModel = 8; // explicit equivalent of default auto result
});
```

See [concurrency.md](concurrency.md) before increasing `ModelInstanceCount`; an extra instance means an extra ONNX model/session in memory.

## 7. Persist embeddings

Store the complete `TextEmbedding`, not only raw vector bytes. The fingerprint and source metadata are intentionally part of the persistence contract.
