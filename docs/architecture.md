# Architecture

The package has a deliberately short pipeline:

```text
Model source
   ↓
transactional cache
   ↓
Hugging Face tokenizer + token offsets
   ↓
structure-aware chunker
   ↓
bounded inference queue
   ↓
1..N ONNX Runtime sessions/model instances
   ├─ concurrent Run slot 1
   ├─ concurrent Run slot 2
   └─ ...
   ↓
versioned embedding records
   ↓
DefaultV1 semantic search / application persistence
```

## Model acquisition

Hugging Face repositories, local directories, and HTTP manifests converge on the same runtime snapshot contract. Large weights remain outside the NuGet package.

## Tokenization and chunking

The tokenizer is created once per active runtime. Source tokenization records offsets so chunk construction can map directly back to the original UTF-16 document. Markdown headings are recognized outside fenced code blocks and become structural boundaries/context.

## Inference concurrency

A model instance is one `InferenceSession`, not one request slot. ONNX Runtime sessions can service concurrent `Run()` calls, so each session is paired with multiple queue dispatchers.

Default runtime topology:

```text
ModelInstanceCount = 1
ThreadsPerModel = 16
ConcurrentRequestsPerModel = Auto → 8
```

This provides eight in-flight inference calls while keeping one model copy in memory. Additional model instances remain available when deliberately configured and multiply total concurrency as well as model/session memory.

## Token accounting API

The service exposes a simple source token count plus a query-aware count result. The latter distinguishes source tokens from final model-input tokens and can report an oversized query with `Fits = false` without throwing. The actual query embedding API still enforces the configured limit by exception.

## Stable records

The vector encoding and embedding record schema are explicitly versioned. Quantized vector bytes are self-describing. Document records preserve the model-space fingerprint and the historical token capacity used when the chunk was created.

## Semantic ranking

Search operates on stored vectors and does not require ONNX inference after a `QueryEmbedding` has been prepared. DefaultV1 scores each chunk, chooses strongest evidence, allows only two bounded support contributions, then applies the same evidence aggregation to weighted semantic fields.

## Storage boundary

Core has no persistence dependency. PostgreSQL helpers are isolated in `OnnxTextEmbeddings.NET.PgVector`; SQLite, SQL Server, files, and other stores can persist the portable core records directly.
