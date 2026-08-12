# OnnxTextEmbeddings.NET

Local, CPU-friendly text embeddings and lightweight semantic, lexical, and hybrid search for .NET 10.

**No Python. No embedding server. No Hugging Face CLI. No vector database required.** Install a NuGet package, register one service, and the default Jasper ONNX model is downloaded and cached automatically on first use.

OnnxTextEmbeddings.NET is aimed at wikis, documentation, note apps, games, local tools, and other small-to-medium search workloads where running a separate AI/search stack would be ridiculous overhead.

## Install

```bash
dotnet add package OnnxTextEmbeddings.NET
```

Optional database-native search integrations are separate packages, so core never drags in database dependencies you do not use:

```bash
dotnet add package OnnxTextEmbeddings.NET.PgVector
dotnet add package OnnxTextEmbeddings.NET.SqliteVec
dotnet add package OnnxTextEmbeddings.NET.SqlServer
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

Search application records semantically without a vector database:

```csharp
var results = await semanticSearch.SearchAsync(
    "How do I restore PostgreSQL?",
    pages,
    page => page.Embeddings,
    new SemanticSearchRequest { Top = 10 });
```

## Lexical / BM25 search without a model

Core also includes deterministic in-memory `BM25-v1` lexical search with weighted fields:

```csharp
var results = await lexicalSearch.SearchAsync(
    "postgresql backup",
    pages,
    page =>
    [
        LexicalField.Create("title", page.Title, 8f),
        LexicalField.Create("description", page.Description, 3f),
        LexicalField.Create("body", page.Body, 1f)
    ]);
```

If an application only wants lexical search, it can register just:

```csharp
builder.Services.AddOnnxTextEmbeddingsLexicalSearch();
```

That path does **not** register or initialize Jasper/ONNX inference.

## Advanced semantic + lexical + hybrid queries

`SearchQuery` exposes the power-user layer while leaving the simple APIs intact:

```csharp
var query = SearchQuery.Create("postgresql backup")
    .Where(SearchFilter.And(
        SearchFilter.Equal("TenantId", tenantId),
        SearchFilter.NotEqual("Status", "Deleted")))
    .Add(SearchRetrievalStage.Semantic(
            "semantic-content",
            SearchFieldWeight.Create("content", 1f))
        .Candidates(250))
    .Add(SearchRetrievalStage.Lexical(
            "lexical-metadata",
            SearchFieldWeight.Create("title", 8f),
            SearchFieldWeight.Create("category", 5f),
            SearchFieldWeight.Create("description", 2f),
            SearchFieldWeight.Create("body", 1f))
        .Candidates(150))
    .UseReciprocalRankFusion()
    .Take(20);
```

Global filters apply to all retrieval stages. Individual stages can add their own filters, field weights, candidate budgets, and stage weights. `PostWhere(...)` exists for intentionally late filtering after fusion, though prefiltering is preferred whenever possible.

Hybrid search uses **Reciprocal Rank Fusion (RRF)** rather than blending raw cosine/BM25/database relevance scores. The raw score units from semantic search, SQLite FTS5, PostgreSQL full-text search, and SQL Server Full-Text Search are not directly comparable; their rank positions are.

See [Lexical, BM25, hybrid, and advanced search](docs/lexical-hybrid-search.md).

## Database-native search

For larger persisted working sets, PostgreSQL, SQLite, and SQL Server/Azure SQL can keep both broad semantic and lexical retrieval inside the database:

```text
portable/provider filters
          ↓
 ┌────────┴────────┐
 │                 │
vector search   lexical search
 │                 │
DefaultV1       native rank
 └────────┬────────┘
          ↓
          RRF
          ↓
      final results
```

| Package | Native semantic | Native lexical |
|---|---|---|
| `OnnxTextEmbeddings.NET.PgVector` | pgvector | `tsvector` + `ts_rank_cd` / `ts_rank` |
| `OnnxTextEmbeddings.NET.SqliteVec` | sqlite-vec `vec0` | FTS5 `bm25()` |
| `OnnxTextEmbeddings.NET.SqlServer` | SQL Server/Azure SQL vectors | Full-Text `FREETEXTTABLE` / `CONTAINSTABLE` |

The database integrations deliberately do **not** reimplement semantic scoring in SQL. They retrieve plausible direct chunks, then the same `DefaultV1` implementation used by in-memory search applies chunk-length confidence, bounded supporting evidence, and semantic-field weights.

Portable `SearchFilter` expressions are translated into parameterized native predicates so tenant/project/security/date/status filtering can happen before retrieval. Each provider also retains an `AdditionalWhereSql` + native command callback escape hatch when an application intentionally needs PostgreSQL JSONB, a SQL Server-specific predicate, a SQLite function, or another nonportable feature.

Plain SQLite remains valid generic persistence; **sqlite-vec is the official SQLite-native vector integration**, with FTS5 as its native lexical engine. See [Database-native search](docs/database-search.md).

## Reusable infrastructure underneath

OnnxTextEmbeddings.NET now consumes two reusable .NET 10 NuGet packages extracted from its original infrastructure:

```text
ModelArtifacts.NET
    acquisition / revisioning / downloads / staging / cache / verification / promotion

OnnxModelRuntime.NET
    InferenceSession hosting / bounded queue / concurrency / scheduling / recovery
```

They are transitive implementation dependencies of the core package; normal consumers still install/configure `OnnxTextEmbeddings.NET` exactly as before.

The ownership boundary is intentional:

- `ModelArtifacts.NET` owns generic files/artifact lifecycle but does not understand embeddings or ONNX execution.
- `OnnxModelRuntime.NET` owns generic ONNX session orchestration but does not understand tokenizers, tensor meaning, pooling, or vectors.
- `OnnxTextEmbeddings.NET` owns Jasper/custom embedding policy, tokenization/chunking, the embedding tensor contract, pooling/normalization, embedding-space identity, vector math, and search semantics.

A newly downloaded candidate is tokenizer + ONNX validated through a real inference call **before** it is promoted current. The previous known-good artifact/runtime remains active when an update candidate fails.

## Per-call token and vector overrides

The application-wide values are defaults, not permanent restrictions.

```csharp
var compactChunks = await embeddingService.EmbedDocumentAsync(
    text,
    new EmbeddingRequestOptions
    {
        MaxTokens = 512,
        VectorFormat = EmbeddingVectorFormat.Int8
    });
```

`MaxTokens` changes the actual document chunk/model-input ceiling for that call. It still cannot exceed the loaded model's hard maximum.

Queries remain exactly one vector and are never chunked. A caller may raise or lower that request's acceptance ceiling independently of the global `QueryMaxTokens`:

```csharp
var queryEmbedding = await embeddingService.EmbedQueryAsync(
    queryText,
    new QueryEmbeddingRequestOptions
    {
        MaxTokens = 2048,
        VectorFormat = EmbeddingVectorFormat.Float32
    });
```

The matching non-throwing validation API accepts the same request options:

```csharp
var count = await embeddingService.CountQueryTokensAsync(
    queryText,
    new QueryEmbeddingRequestOptions { MaxTokens = 2048 });

if (count.Fits)
    queryEmbedding = await embeddingService.EmbedQueryAsync(
        queryText,
        new QueryEmbeddingRequestOptions { MaxTokens = 2048 });
```

## CPU concurrency and multiple model instances

One ONNX `InferenceSession` can execute several inference calls concurrently, so normal concurrency does **not** require duplicate model weights in memory.

Default model topology:

```text
ModelInstanceCount = 1
ThreadsPerModel = 16
ConcurrentRequestsPerModel = Auto → 8
```

Automatic concurrency is `ThreadsPerModel / 2`, minimum one, capped at **8**. Explicit positive `ConcurrentRequestsPerModel` values always win.

The public options stay owned by OnnxTextEmbeddings.NET and map into `OnnxModelRuntime.NET` internally. Work is routed to the healthy instance with the fewest active requests, with rotating tie-breaking; failures are isolated/recovered per model instance.

**Multiple model instances usually do not increase aggregate throughput on a normal CPU host.** Once one model instance is busy enough, the bottleneck is commonly somewhere in the shared CPU/memory subsystem—often memory bandwidth/cache/memory-controller or board-level throughput—not a shortage of model copies. `ModelInstanceCount > 1` exists for experimentation, unusual hardware/topologies, and future tuning work rather than as a general performance recommendation.

See [Concurrency, load balancing, and recovery](docs/concurrency.md).

## Self-healing model instances

Each model copy has independent health and capacity tracking. A recoverable ONNX session/runtime failure removes that instance from routing, lets its active calls drain, disposes the old session, creates a fresh session generation, and returns the instance to service only after the replacement loads successfully.

If another instance is healthy, traffic continues there at reduced capacity. If every instance is unavailable, the bounded global queue waits for recovery instead of inventing capacity or permanently losing a request slot.

Memory-pressure failures rebuild the affected instance but are not immediately retried on another model copy, avoiding a cascading OOM. This recovery exists **inside a live process**; if the operating system or container runtime kills the entire .NET process for OOM, process/service-level restart is still required.

## Default model

The default is Jasper INT8:

- `magiccodingman/Jasper-Token-Compression-600M-ONNX-INT8`
- `magiccodingman/Jasper-Token-Compression-600M-ONNX-INT4`
- `magiccodingman/Jasper-Token-Compression-600M-ONNX-FP32`

Model precision and returned-vector precision are independent.

### Why Jasper?

Jasper Token Compression 600M was chosen because its dynamic token-compression architecture is an unusually good fit for CPU document embedding: measured latency stays remarkably flat as context grows, while the final Dynamic INT8 export remains small and very close to FP32 quality.

| Dynamic INT8 | 32 tokens | 512 tokens | 1024 tokens |
|---|---:|---:|---:|
| Latency | 44.4 ms | 63.2 ms | **84.6 ms** |
| Throughput | 721 tok/s | 8,104 tok/s | **12,101 tok/s** |
| Speed vs FP32 | 3.14× | 3.35× | **3.41×** |

The project intentionally defaults to **1024 tokens**: local quality testing found Jasper extremely strong through roughly 756 tokens and only slightly degraded around 1024, but with a much sharper long-tail falloff beyond 1024—consistent with the upstream model having been distilled only through 1024 tokens. See [Why Jasper is the default model](docs/jasper-model.md) for the complete CPU, memory, concurrency, fidelity, and length-quality results.

## Vector formats: FP32 by default, INT8 recommended for compact storage

Both document and query embeddings return **FP32 by default** for maximum compatibility.

| Format | Approx. payload per 2048-d vector | Notes |
|---|---:|---|
| INT4 | 1 KiB | Packed aggressive quantization |
| INT8 | 2 KiB | Recommended compact storage option |
| FP16 | 4 KiB | Half precision |
| FP32 | 8 KiB | Default; maximum interoperability |

Make INT8 the application-wide document default when compact storage matters:

```csharp
builder.Services.AddOnnxTextEmbeddings(options =>
{
    options.Vectors.DocumentFormat = EmbeddingVectorFormat.Int8;
});
```

Or choose dynamically:

```csharp
var tiny = await embeddingService.EmbedDocumentAsync(text, EmbeddingVectorFormat.Int4);
var compact = await embeddingService.EmbedDocumentAsync(text, EmbeddingVectorFormat.Int8);
var half = await embeddingService.EmbedDocumentAsync(text, EmbeddingVectorFormat.Float16);
var full = await embeddingService.EmbedDocumentAsync(text, EmbeddingVectorFormat.Float32);
```

### Convert vectors you already have

```csharp
EmbeddingVector fp32 = EmbeddingVector.FromFloat32(values);
EmbeddingVector fp16 = EmbeddingVector.FromFloat32(values, EmbeddingVectorFormat.Float16);
EmbeddingVector int8 = EmbeddingVector.FromFloat32(values, EmbeddingVectorFormat.Int8);
EmbeddingVector int4 = EmbeddingVector.FromFloat32(values, EmbeddingVectorFormat.Int4);

EmbeddingVector smaller = fp32.ConvertTo(EmbeddingVectorFormat.Int8);
```

Expanding a lower-precision vector back to FP32 changes its representation; it cannot restore fidelity already discarded by quantization.

## Combine many chunk embeddings into one

Keeping the original chunk array is the preferred representation for semantic retrieval. When a consumer explicitly requires exactly one vector, the returned `IReadOnlyList<TextEmbedding>` can be mathematically compressed:

```csharp
var chunks = await embeddingService.EmbedDocumentAsync(longText);
var single = chunks.CombineToSingle();
```

For multiple chunks, the default `SemanticCoverage-v1` profile performs full-dimensional FP32 semantic aggregation while giving highly repetitive semantic regions diminishing additional influence. It uses persisted source token ranges to avoid double-counting overlap and returns diagnostics such as `AggregationCoherence` and `MinimumSourceSimilarity`.

Aggregation, dimension reduction, and numeric conversion are separate stages. A caller can request all three in one operation:

```csharp
var single = chunks.CombineToSingle(new SingleEmbeddingOptions
{
    OutputDimensions = 512,
    OutputFormat = EmbeddingVectorFormat.Int8
});
```

That means:

```text
chunk vectors
   ↓
FP32 SemanticCoverage-v1 at full dimensions
   ↓
deterministic SRHT-v1 → 512
   ↓
normalize
   ↓
INT8
```

Dimension expansion is never allowed. A 2048-dimensional supplied representation cannot be turned into a meaningful 4096-dimensional embedding.

Reduced vectors occupy a new deterministic embedding space, so queries must receive the same transform:

```csharp
var queryEmbedding = await embeddingService.EmbedQueryAsync("database restore");
var reducedQuery = queryEmbedding.ReduceDimensions(512);

var cosine = EmbeddingVectorMath.CosineSimilarity(
    reducedQuery.Vector,
    single.Vector);
```

Direct chunk vectors can also be reduced without aggregation:

```csharp
var reducedChunk = chunks[0].ReduceDimensions(1024);
```

The reduced direct chunk preserves its source/chunk metadata and enters the same deterministic child space as a query reduced with the same profile.

See [Single-embedding aggregation](docs/single-embedding.md) and [Dimension reduction](docs/dimension-reduction.md).

## Native AOT and non-.NET interoperability

The core package declares Native AOT compatibility and is continuously AOT-published/tested on Linux, Windows, and macOS.

The repository also contains a separate `OnnxTextEmbeddings.Native` facade that publishes the canonical C# implementation as a Native AOT shared library with a versioned C ABI:

```text
OnnxTextEmbeddings.Native
        ↓
.so / .dll / .dylib
        ↓
stable C ABI
        ↓
Rust / C / C++ / Go / Zig / Python FFI / other bindings
```

This native facade is intentionally **not a NuGet package**. The project maintains the C ABI, public header, and cross-platform C interoperability tests; third-party language bindings are welcome without implying that every language wrapper becomes a first-party SDK.

See [Native AOT compatibility](docs/native-aot.md) and [Native interoperability](docs/native-interop.md).

## Configuration defaults

```text
Jasper model                  INT8
DocumentChunkMaxTokens        1024
QueryMaxTokens                1024
ModelInstanceCount            1
ThreadsPerModel               16
ConcurrentRequestsPerModel    Auto (8 at 16 threads/model)
QueueCapacity                 256
ChunkOverlapTokens            0
RepeatHeadingContext          true
Document vector format        FP32
Query vector format           FP32
Semantic scoring profile      DefaultV1
Lexical scoring profile       BM25-v1 (in memory)
Hybrid fusion                 RRF (when requested)
```

## Runtime diagnostics

`ITextEmbeddingService.ModelInfo` reports current model-instance health, active request counts, generation numbers, recovery counts, and the aggregate number of healthy/recovering instances. This is useful for health endpoints and production diagnostics without requiring a monitoring framework.

## Persistence

The core package owns no database. Store `TextEmbedding` or `SingleEmbedding` records wherever the application already stores data: memory, SQLite BLOBs, SQL Server `VARBINARY`, PostgreSQL `BYTEA`, JSON/files, or a native vector column through one of the optional database adapters.

Lexical indexes are likewise application schema: FTS5 tables, PostgreSQL `tsvector` columns/indexes, and SQL Server Full-Text indexes remain normal database objects that the provider query mappings target.

## Documentation

- [Getting started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Why Jasper is the default model](docs/jasper-model.md)
- [Concurrency, load balancing, and recovery](docs/concurrency.md)
- [Vector formats and conversion](docs/vector-formats.md)
- [Single-embedding aggregation](docs/single-embedding.md)
- [Dimension reduction](docs/dimension-reduction.md)
- [Model sources](docs/model-sources.md)
- [HTTP model manifest](docs/model-manifest.md)
- [Model artifacts, cache, and updates](docs/model-cache.md)
- [Chunking](docs/chunking.md)
- [Semantic search](docs/semantic-search.md)
- [DefaultV1 scoring](docs/semantic-scoring.md)
- [Lexical, BM25, hybrid, and advanced search](docs/lexical-hybrid-search.md)
- [Database-native semantic, lexical, and hybrid search](docs/database-search.md)
- [Persistence](docs/persistence.md)
- [SQLite persistence](docs/sqlite.md)
- [SQLite/sqlite-vec](docs/sqlite-vec.md)
- [PostgreSQL/pgvector](docs/postgres.md)
- [SQL Server 2025 / Azure SQL](docs/sql-server.md)
- [Native AOT compatibility](docs/native-aot.md)
- [Native interoperability / C ABI](docs/native-interop.md)
- [Performance](docs/performance.md)
- [Deployment](docs/deployment.md)
- [Troubleshooting](docs/troubleshooting.md)

## Scope

This library is intentionally not a RAG framework, vector database, ingestion platform, PDF parser, crawler, GPU framework, or distributed embedding service. It is a focused way to add high-quality local text embeddings plus useful lexical/hybrid search to an ordinary .NET application—or, through the Native AOT facade, to another runtime that wants to bind to the same engine.

## License

Apache-2.0.