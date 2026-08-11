# Getting started

## 1. Install and register

```bash
dotnet add package OnnxTextEmbeddings.NET
```

```csharp
builder.Services.AddOnnxTextEmbeddings();
```

The default registration uses Jasper INT8 on CPU, one inference worker, 1024-token document chunks, a 1024-token query ceiling, INT8 document-vector storage, and FP32 query-vector storage.

## 2. First model download

The model is not bundled in the NuGet package. The first initialization resolves `magiccodingman/Jasper-Token-Compression-600M-ONNX-INT8` through the Hugging Face HTTP API, downloads the ONNX/tokenizer assets, creates a cache snapshot, and then loads ONNX Runtime.

ASP.NET Core/Generic Host registrations include a warmup hosted service. By default warmup starts in the background; set `Initialization.BlockHostStartupUntilReady = true` when host startup must not complete until embeddings are ready.

## 3. Embed a document

```csharp
var embeddings = await embeddingService.EmbedDocumentAsync(markdown);
```

`EmbedAsync` is the shorter equivalent. A short document yields one embedding. Larger documents yield several `TextEmbedding` records with exact UTF-16 source ranges, token ranges, chunk index/count, heading path, historical token capacity, model revision, and embedding-space fingerprint.

## 4. Embed a query

```csharp
var query = await embeddingService.EmbedQueryAsync("how do backups work?");
```

A query is always one vector. Oversized queries throw `QueryTokenLimitExceededException`; the library never silently truncates or chunks a semantic query.

## 5. Search in memory

```csharp
var results = await semanticSearch.SearchAsync(
    query,
    documents,
    document => document.Embeddings,
    new SemanticSearchRequest { Top = 10 });
```

No database integration is required. `SemanticSearchResult<T>` exposes final score, best chunk, per-field scores, raw cosine values, length confidence, and adjusted similarity.

## 6. Persist embeddings

Store the complete `TextEmbedding`, not only raw vector bytes. The fingerprint and source metadata are intentionally part of the persistence contract. See [persistence.md](persistence.md).

## Cache location

Set an explicit cache directory when deployment needs a predictable path:

```csharp
builder.Services.AddOnnxTextEmbeddings(options =>
{
    options.Cache.Directory = Path.Combine(appData, "onnx-embeddings");
});
```

When unset, the package uses its platform-appropriate default cache root.
