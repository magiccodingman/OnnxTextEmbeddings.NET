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
versioned TextEmbedding / QueryEmbedding records
   ↓
DefaultV1 semantic search / application persistence
```

An optional deterministic post-processing path exists when a consumer explicitly requires one vector:

```text
TextEmbedding[]
   ↓
validate common embedding space / dimensions
   ↓
decode + normalize FP32 working vectors
   ↓
SemanticCoverage-v1
   ↓
one native-dimensional semantic vector
   ↓
optional SRHT-v1 dimension reduction
   ↓
new reduced-space fingerprint when coordinates changed
   ↓
optional FP32 / FP16 / INT8 / INT4 conversion
   ↓
SingleEmbedding
```

Semantic aggregation, coordinate-space reduction, and numeric/storage conversion are intentionally independent stages because they lose different kinds of information.

## Model acquisition

Hugging Face repositories, local directories, and HTTP manifests converge on the same runtime snapshot contract. Large weights remain outside the NuGet package.

## Tokenization and request preparation

The tokenizer is created once per active model runtime. Source tokenization records offsets so chunks map directly back to original UTF-16 text.

Application-wide document/query token limits can be overridden per call. Document overrides alter the chunking ceiling; query overrides alter only validation because a semantic query is always one vector.

## Inference scheduling

A model instance is one ONNX `InferenceSession`, not one request slot. Each healthy instance can execute several concurrent `Run()` calls.

A single scheduler owns routing decisions. It tracks active request counts and chooses the least-loaded healthy instance with available capacity. Ties rotate between equal instances. Native inference is launched independently after reservation so the central scheduler remains free to route the rest of a burst.

Automatic request concurrency is `ThreadsPerModel / 2`, minimum one, capped at eight. With 16 threads/model, one model instance therefore exposes eight request slots by default.

Additional model copies are supported but deliberately not treated as a normal scaling primitive. CPU inference frequently runs into shared memory/cache/interconnect/platform throughput before it runs out of model instances, so extra sessions often add RAM without adding meaningful throughput. The multi-instance machinery is useful for experimentation, odd hardware topologies, HA recovery behavior, and future CPU/NUMA/affinity tuning.

## Instance recovery

A recoverable session failure immediately removes that instance from future routing. Existing calls drain before the session is disposed. Recovery creates an entirely new `InferenceSession`, increments the instance generation, and only then restores the instance to the healthy pool.

Failed recovery attempts stay out of rotation and use bounded exponential backoff. With no healthy instance, the global bounded queue waits for recovery.

A normal recoverable runtime failure can retry its request once. Memory-pressure failures do not immediately retry cross-instance, reducing the chance of cascading memory exhaustion.

## Observability

`ModelRuntimeInfo` reports aggregate active/healthy/recovering counts and point-in-time `ModelInstanceRuntimeInfo` records. This lets an ASP.NET health endpoint expose reduced capacity or recovery without coupling the library to a monitoring stack.

## Stable direct records

The vector encoding and embedding record schema are explicitly versioned. Quantized vector bytes are self-describing. Document records preserve the model-space fingerprint and the historical token capacity actually used when the chunk was created—including per-call chunk overrides.

## Single-embedding aggregation

`CombineToSingle` operates entirely after inference. It does not require source text or another model.

The combiner uses the persisted token ranges to apportion overlapping original source content symmetrically, computes pairwise semantic redundancy directly in the supplied full-dimensional space, and returns a `SingleEmbedding` rather than inventing direct-chunk metadata.

Aggregation itself does not create a new coordinate system: a native-dimensional aggregate remains in the source embedding space. `AggregationCoherence` exposes how strongly the contributing source directions agree.

## Dimension reduction

Dimension reduction occurs only after aggregation. `SRHT-v1` is deterministic and supports non-power-of-two source dimensions by internal zero-padding for the Hadamard transform.

Reduction *does* create a new coordinate space, so the embedding-space fingerprint is deterministically derived from the base fingerprint, reduction profile, source dimensions, and output dimensions. Query embeddings must apply the same transform before cosine comparison.

## Semantic ranking

Search operates on stored direct chunk vectors and does not require ONNX inference after a `QueryEmbedding` has been prepared. DefaultV1 scores each chunk, chooses strongest evidence, allows only bounded supporting evidence, then applies the same principle to weighted semantic fields.

`SingleEmbedding` is deliberately not silently treated as a `TextEmbedding`: aggregate source-token count describes all original content compressed into that vector and must not be interpreted as one direct chunk's length confidence.

## Storage boundary

Core has no persistence dependency. PostgreSQL helpers are isolated in `OnnxTextEmbeddings.NET.PgVector`; SQLite, SQL Server, files, and other stores can persist portable core records directly.
