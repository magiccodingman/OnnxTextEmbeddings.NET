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

## Queries and token counting

Queries deliberately produce **one embedding vector**. They are never silently truncated or chunked.

```csharp
QueryEmbedding query = await embeddingService.EmbedQueryAsync("postgres backup restore");
```

If the final query input exceeds `QueryMaxTokens`, `EmbedQueryAsync` throws `QueryTokenLimitExceededException`. Applications that want to validate first can count without triggering that limit error:

```csharp
int sourceTokens = await embeddingService.CountTokensAsync(userText);
QueryTokenCount queryCount = await embeddingService.CountQueryTokensAsync(userText);

if (queryCount.Fits)
{
    var query = await embeddingService.EmbedQueryAsync(userText);
}
```

`QueryTokenCount` reports source tokens, actual model-input tokens (including tokenizer-added special tokens), the configured query maximum, the model maximum when known, and `Fits`.

## Concurrent CPU inference without duplicate model copies

A single ONNX Runtime session can execute multiple `Run()` calls concurrently, so request concurrency does **not** require loading another copy of the model.

Defaults:

```text
ModelInstanceCount             1
ThreadsPerModel               16
ConcurrentRequestsPerModel    Auto → min(ThreadsPerModel / 2, 8)
Resolved default concurrency   8 requests
QueueCapacity                256
```

So the normal configuration keeps **one Jasper model in memory** while allowing up to **8 inference requests in flight against that model instance**. Each request still has its own normal 1024-token default limit; concurrency does not combine or divide token budgets.

Tune explicitly when desired:

```csharp
builder.Services.AddOnnxTextEmbeddings(options =>
{
    options.Inference.ModelInstanceCount = 1;
    options.Inference.ThreadsPerModel = 16;
    options.Inference.ConcurrentRequestsPerModel = 8;
});
```

Automatic concurrency is capped at 8. Explicit positive values are honored, but **8 concurrent requests per model is the recommended practical maximum**; benchmarks showed little or no additional benefit beyond that point. Only increase `ModelInstanceCount` when you intentionally want another independent ONNX session/model copy in memory.

See [Concurrency and threading](docs/concurrency.md).

## Default model

The default is the CPU-friendly Jasper dynamic INT8 ONNX model:

- `magiccodingman/Jasper-Token-Compression-600M-ONNX-INT8`
- `magiccodingman/Jasper-Token-Compression-600M-ONNX-INT4`
- `magiccodingman/Jasper-Token-Compression-600M-ONNX-FP32`

Switch model precision without changing the rest of the application:

```csharp
builder.Services.AddOnnxTextEmbeddings(options =>
{
    options.Model.UseJasper(JasperModelPrecision.Int4);
});
```

Model precision and **stored-vector precision are independent**.

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
Jasper model                  INT8
DocumentChunkMaxTokens        1024
QueryMaxTokens                1024
ModelInstanceCount            1
ThreadsPerModel               16
ConcurrentRequestsPerModel    Auto (8 with default threads)
QueueCapacity                 256
ChunkOverlapTokens            0
RepeatHeadingContext          true
Document vector format        INT8
Query vector format           FP32
Scoring profile               DefaultV1
```

## Model cache and updates

The first request (or hosted-service warmup) resolves the model, downloads runtime assets into a local cache, validates them, creates the tokenizer and ONNX runtime, and atomically activates the snapshot.

Updates are transactional. A failed candidate does not replace a working runtime. When a successful update is activated, old sessions are disposed before old snapshot files are removed.

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

## Documentation

- [Getting started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Concurrency and threading](docs/concurrency.md)
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

## License

Apache-2.0.
