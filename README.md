# OnnxTextEmbeddings.NET

Local, CPU-friendly text embeddings and lightweight semantic search for .NET 10.

**No Python. No embedding server. No Hugging Face CLI. No vector database required.** Install a NuGet package, register one service, and the default Jasper ONNX model is downloaded and cached automatically on first use.

OnnxTextEmbeddings.NET is aimed at wikis, documentation, note apps, games, local tools, and other small-to-medium semantic-search workloads where running a separate AI stack would be ridiculous overhead.

## Install

```bash
dotnet add package OnnxTextEmbeddings.NET
```

Optional PostgreSQL/pgvector helpers live in a separate package so core users do not inherit PostgreSQL dependencies:

```bash
dotnet add package OnnxTextEmbeddings.NET.PgVector
```

## Five-minute start

Register the defaults:

```csharp
builder.Services.AddOnnxTextEmbeddings();
```

Inject `ITextEmbeddingService` and embed text:

```csharp
var embeddings = await embeddingService.EmbedAsync("Hello world");
```

Short text normally returns one `TextEmbedding`. Long text is chunked automatically using Markdown structure, paragraphs, sentences, words, and finally token windows when necessary.

Search application records without a vector database:

```csharp
var results = await semanticSearch.SearchAsync(
    "How do I restore PostgreSQL?",
    pages,
    page => page.Embeddings,
    new SemanticSearchRequest { Top = 10 });
```

The default search score is intentionally evidence-oriented: one highly relevant section can make a long document rank well; unrelated sections do not average the item downward; strong secondary matches provide only bounded support.

## Queries and precomputed queries

Queries are deliberately a **single embedding vector**. They are never silently truncated or chunked.

```csharp
QueryEmbedding query = await embeddingService.EmbedQueryAsync("postgres backup restore");

var results = await semanticSearch.SearchAsync(
    query,
    pages,
    page => page.Embeddings);
```

If a query exceeds `QueryMaxTokens`, `QueryTokenLimitExceededException` is thrown so the application can make an explicit decision.

## Default model

The default is the CPU-friendly Jasper dynamic INT8 ONNX model:

- `magiccodingman/Jasper-Token-Compression-600M-ONNX-INT8`
- `magiccodingman/Jasper-Token-Compression-600M-ONNX-INT4`
- `magiccodingman/Jasper-Token-Compression-600M-ONNX-FP32`

Switch precision without changing the rest of the application:

```csharp
builder.Services.AddOnnxTextEmbeddings(options =>
{
    options.Model.UseJasper(JasperModelPrecision.Int4);
});
```

Model precision and **stored-vector precision are independent**. You can run the INT8 model while storing packed INT4 vectors, or run FP32 while persisting INT8 vectors.

## Compact vector storage

Document embeddings default to INT8 storage. Supported formats are:

| Format | Approx. payload per 2048-d vector | Notes |
|---|---:|---|
| INT4 | 1 KiB | Packed two signed values per byte |
| INT8 | 2 KiB | Default document storage |
| FP16 | 4 KiB | Standard IEEE half precision |
| FP32 | 8 KiB | Maximum fidelity |

```csharp
builder.Services.AddOnnxTextEmbeddings(options =>
{
    options.Vectors.DocumentFormat = EmbeddingVectorFormat.Int4;
    options.Vectors.QueryFormat = EmbeddingVectorFormat.Float32;
});
```

Each persisted vector carries encoding version, dimensions, format, and quantization metadata. `TextEmbedding` also carries the embedding-space fingerprint, source revision, source ranges, token counts, capacity, and chunk metadata.

## Weighted semantic fields

Applications often want title, tags, description, and body to contribute differently:

```csharp
var results = await semanticSearch.SearchFieldsAsync(
    query,
    pages,
    page =>
    [
        SemanticField.Create("title", page.TitleEmbeddings, 1.4f),
        SemanticField.Create("tags", page.TagEmbeddings, 1.2f),
        SemanticField.Create("content", page.ContentEmbeddings, 1.0f)
    ]);
```

## Configuration defaults

```text
Jasper model             INT8
DocumentChunkMaxTokens   1024
QueryMaxTokens           1024
WorkerCount              1
ThreadsPerWorker         Auto
MaximumAutoThreads       12
QueueCapacity            256
ChunkOverlapTokens       0
RepeatHeadingContext     true
Document vector format   INT8
Query vector format      FP32
Scoring profile          DefaultV1
```

The default queue is bounded and every configured worker owns an independent ONNX Runtime session. CPU thread budgets are configurable; the package does not create a model/session per request.

## Model cache and updates

The first request (or hosted-service warmup) resolves the model, downloads runtime assets into a local cache, validates them, creates the tokenizer and ONNX worker pool, and atomically activates the snapshot.

Updates are transactional. A failed candidate does not replace a working runtime. When a successful update is activated, old sessions are disposed before old snapshot files are removed, which matters on Windows.

```csharp
bool changed = await embeddingService.UpdateModelAsync();
```

Persisted embeddings from a different embedding-space fingerprint are rejected during search rather than producing meaningless cosine scores.

## Persistence

The core package owns no database. Store `TextEmbedding` records wherever the application already stores data:

- memory
- SQLite BLOBs
- SQL Server `VARBINARY`
- PostgreSQL `BYTEA`
- JSON/files
- pgvector `vector`/`halfvec` through the optional adapter

For compact portable persistence, `EmbeddingSerializer` provides JSON and a versioned binary vector representation.

## Documentation

- [Getting started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Model sources](docs/model-sources.md)
- [HTTP model manifest](docs/model-manifest.md)
- [Model cache and updates](docs/model-cache.md)
- [Chunking](docs/chunking.md)
- [Semantic search](docs/semantic-search.md)
- [DefaultV1 scoring](docs/semantic-scoring.md)
- [Persistence](docs/persistence.md)
- [SQLite](docs/sqlite.md)
- [PostgreSQL/pgvector](docs/postgres.md)
- [Performance](docs/performance.md)
- [Deployment](docs/deployment.md)
- [Troubleshooting](docs/troubleshooting.md)

## Scope

This library is intentionally not a RAG framework, vector database, ingestion platform, PDF parser, crawler, GPU framework, or distributed embedding service. It is a focused way to add high-quality local text embeddings and good small-scale semantic search to an ordinary .NET application.

If an application grows into millions of vectors, distributed ingestion, dedicated rerankers, or massive high-throughput search, use dedicated vector/search infrastructure. Keeping that boundary explicit is what lets this package stay small and easy.

## License

Apache-2.0.
