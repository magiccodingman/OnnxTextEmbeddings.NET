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
independent ONNX Runtime CPU workers
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

Inference uses a bounded channel. `WorkerCount = 1` is the default. Additional workers create independent ONNX sessions rather than attempting concurrent mutation of one session. Each worker receives its own intra-op thread budget.

## Stable records

The vector encoding and embedding record schema are explicitly versioned. Quantized vector bytes are self-describing. Document records preserve the model-space fingerprint and the historical token capacity used when the chunk was created.

## Semantic ranking

Search operates on stored vectors and does not require ONNX inference after a `QueryEmbedding` has been prepared. DefaultV1 scores each chunk, chooses strongest evidence, allows only two bounded support contributions, then applies the same evidence aggregation to weighted semantic fields.

## Storage boundary

Core has no persistence dependency. PostgreSQL helpers are isolated in `OnnxTextEmbeddings.NET.PgVector`; SQLite, SQL Server, files, and other stores can persist the portable core records directly.
