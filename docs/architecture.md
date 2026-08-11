# Architecture

The package pipeline is:

```text
Model source
   ↓
transactional cache
   ↓
Hugging Face tokenizer + token offsets
   ↓
structure-aware chunker / query validation
   ↓
bounded global inference queue
   ↓
least-loaded healthy-instance scheduler
   ↓
1..N managed ONNX model instances
   ├─ health + active slot accounting
   ├─ concurrent Run() calls
   └─ drain / rebuild / generation recovery
   ↓
versioned embedding records
   ↓
DefaultV1 semantic search / application persistence
```

## Model acquisition

Hugging Face repositories, local directories, and HTTP manifests converge on the same runtime snapshot contract. Large weights remain outside the NuGet package.

## Tokenization and request preparation

The tokenizer is created once per active model runtime. Source tokenization records offsets so chunks map directly back to original UTF-16 text.

Application-wide document/query token limits can be overridden per call. Document overrides alter the chunking ceiling; query overrides alter only validation because a semantic query is always one vector.

## Inference scheduling

A model instance is one ONNX `InferenceSession`, not one request slot. Each healthy instance can execute several concurrent `Run()` calls.

A single scheduler owns routing decisions. It tracks active request counts and chooses the least-loaded healthy instance with available capacity. Ties rotate between equal instances. This replaces nondeterministic competition between channel readers.

Default automatic concurrency profiles at 16 threads are 5 for built-in Jasper INT8 and 4 for other/custom models.

## Instance recovery

A recoverable session failure immediately removes that instance from future routing. Existing calls drain before the session is disposed. Recovery creates an entirely new `InferenceSession`, increments the instance generation, and only then restores the instance to the healthy pool.

Failed recovery attempts stay out of rotation and use bounded exponential backoff. With no healthy instance, the global bounded queue waits for recovery.

A normal recoverable runtime failure can retry its request once. Memory-pressure failures do not immediately retry cross-instance, reducing the chance of cascading memory exhaustion.

## Observability

`ModelRuntimeInfo` reports aggregate active/healthy/recovering counts and point-in-time `ModelInstanceRuntimeInfo` records. This lets an ASP.NET health endpoint expose reduced capacity or recovery without coupling the library to a monitoring stack.

## Stable records

The vector encoding and embedding record schema are explicitly versioned. Quantized vector bytes are self-describing. Document records preserve the model-space fingerprint and the historical token capacity actually used when the chunk was created—including per-call chunk overrides.

## Semantic ranking

Search operates on stored vectors and does not require ONNX inference after a `QueryEmbedding` has been prepared. DefaultV1 scores each chunk, chooses strongest evidence, allows only bounded supporting evidence, then applies the same principle to weighted semantic fields.

## Storage boundary

Core has no persistence dependency. PostgreSQL helpers are isolated in `OnnxTextEmbeddings.NET.PgVector`; SQLite, SQL Server, files, and other stores can persist portable core records directly.
