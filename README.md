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
var queryCount = await embeddingService.CountQueryTokensAsync(userText);
if (queryCount.Fits)
    query = await embeddingService.EmbedQueryAsync(userText);
```

## Concurrent CPU inference without duplicate model copies

Defaults keep **one model copy in memory**, use 16 threads, and allow up to 8 simultaneous inference calls against that one ONNX session. See [Concurrency and threading](docs/concurrency.md).

## Default model

The default is the CPU-friendly Jasper dynamic INT8 ONNX model:

- `magiccodingman/Jasper-Token-Compression-600M-ONNX-INT8`
- `magiccodingman/Jasper-Token-Compression-600M-ONNX-INT4`
- `magiccodingman/Jasper-Token-Compression-600M-ONNX-FP32`

Model precision and returned-vector precision are independent.

## Vector formats: FP32 by default, INT8 recommended for compact storage

Both document and query embeddings now return **FP32 by default** for maximum compatibility with databases, vector libraries, and external integrations.

Supported return/storage representations:

| Format | Approx. payload per 2048-d vector | Notes |
|---|---:|---|
| INT4 | 1 KiB | Packed aggressive quantization |
| INT8 | 2 KiB | Recommended compact storage option |
| FP16 | 4 KiB | Half precision |
| FP32 | 8 KiB | Default; maximum interoperability |

Make INT8 your application-wide document default when compact storage matters:

```csharp
builder.Services.AddOnnxTextEmbeddings(options =>
{
    options.Vectors.DocumentFormat = EmbeddingVectorFormat.Int8;
});
```

Or choose dynamically for a single call without changing the global default:

```csharp
var tiny = await embeddingService.EmbedDocumentAsync(text, EmbeddingVectorFormat.Int4);
var compact = await embeddingService.EmbedDocumentAsync(text, EmbeddingVectorFormat.Int8);
var half = await embeddingService.EmbedDocumentAsync(text, EmbeddingVectorFormat.Float16);
var full = await embeddingService.EmbedDocumentAsync(text, EmbeddingVectorFormat.Float32);

var compactQuery = await embeddingService.EmbedQueryAsync(queryText, EmbeddingVectorFormat.Int8);
```

Per-call selection is applied directly to the original FP32 ONNX result, so requesting FP32 never reconstructs an already-quantized default.

### Convert vectors you already have

```csharp
EmbeddingVector fp32 = EmbeddingVector.FromFloat32(values); // preserves FP32
EmbeddingVector fp16 = EmbeddingVector.FromFloat32(values, EmbeddingVectorFormat.Float16);
EmbeddingVector int8 = EmbeddingVector.FromFloat32(values, EmbeddingVectorFormat.Int8);
EmbeddingVector int4 = EmbeddingVector.FromFloat32(values, EmbeddingVectorFormat.Int4);

EmbeddingVector smaller = fp32.ConvertTo(EmbeddingVectorFormat.Int8);
```

A lower-precision vector can be dequantized into an FP32 representation for compatibility, but doing so cannot restore fidelity that was already discarded during quantization.

See [Vector formats and conversion](docs/vector-formats.md).

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
Document vector format        FP32
Query vector format           FP32
Scoring profile               DefaultV1
```

## Model cache and updates

The first request (or hosted-service warmup) resolves the model, downloads runtime assets into a local cache, validates them, creates the tokenizer and ONNX runtime, and atomically activates the snapshot.

Updates are transactional. A failed candidate does not replace a working runtime. When a successful update is activated, old sessions are disposed before old snapshot files are removed.

## Persistence

The core package owns no database. Store `TextEmbedding` records wherever the application already stores data: memory, SQLite BLOBs, SQL Server `VARBINARY`, PostgreSQL `BYTEA`, JSON/files, or pgvector through the optional adapter.

## Documentation

- [Getting started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Concurrency and threading](docs/concurrency.md)
- [Vector formats and conversion](docs/vector-formats.md)
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
