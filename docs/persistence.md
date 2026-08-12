# Persistence

The core package is storage-neutral. Persistence belongs to the application, but **persist the embedding record, not just an anonymous float array**.

## Direct chunk records

For `TextEmbedding`, preserve at minimum:

- vector encoding version
- vector format and dimensions
- vector bytes
- quantization metadata for INT4/INT8
- model ID and source revision
- embedding-space fingerprint
- normalized flag
- document/source token counts
- historical token capacity
- UTF-16 and token ranges
- chunk index/count and boundary type
- heading path/context metadata
- dimension-reduction profile/source/output dimensions when the direct chunk is reduced

The fingerprint is critical. Search rejects stored vectors from a different embedding space.

A reduced direct chunk is still a `TextEmbedding`: its source/chunk metadata remains valid, while its `DimensionReduction` metadata and derived fingerprint identify the transformed coordinate space.

## SingleEmbedding records

When several chunks are intentionally compressed to one vector, persist the `SingleEmbedding` record rather than manufacturing a `TextEmbedding` row.

In addition to vector/identity information it preserves:

- direct vs aggregated representation kind
- source embedding count
- total original source-token count
- source dimensionality
- aggregation profile/version
- source-mass method
- neutral similarity baseline and profile constants
- aggregation coherence
- minimum source similarity
- medoid fallback usage
- dimension-reduction profile/source/output dimensions when reduced

Reduced-space fingerprints must be preserved exactly. A 512-dimensional `SRHT-v1` vector is not interchangeable with some other 512-dimensional projection.

## Binary vector payload

```csharp
byte[] payload = EmbeddingSerializer.SerializeVector(embedding.Vector);
EmbeddingVector vector = EmbeddingSerializer.DeserializeVector(payload);
```

This is a versioned portable payload suitable for SQLite BLOB, SQL Server VARBINARY, PostgreSQL BYTEA, or files.

## JSON records

Direct chunk:

```csharp
string json = EmbeddingSerializer.SerializeJson(embedding);
TextEmbedding restored = EmbeddingSerializer.DeserializeJson(json);
```

Whole direct-chunk collection:

```csharp
string json = EmbeddingSerializer.SerializeJson(chunks);
IReadOnlyList<TextEmbedding> restored =
    EmbeddingSerializer.DeserializeDocumentJson(json);
```

Aggregated/single vector:

```csharp
string json = EmbeddingSerializer.SerializeJson(single);
SingleEmbedding restored = EmbeddingSerializer.DeserializeSingleJson(json);
```

These JSON paths use source-generated `System.Text.Json` metadata and are compatible with Native AOT.

## Native database vector columns

The portable embedding record remains the source of metadata truth even when a database also stores a native vector column for candidate search.

A practical row often contains both:

```text
item/document key
field name + weight
embedding-space fingerprint
native database vector
complete TextEmbedding JSON (or equivalent structured metadata)
```

The native vector is optimized for database KNN/cosine retrieval. The complete direct record is what core DefaultV1 reranks after candidate selection.

Provider-specific dimensional/storage constraints may justify a reduced native candidate vector. In that case persist/search the corresponding reduced `TextEmbedding` record and compare it only with a query transformed into the same derived fingerprint.

## Query compatibility with reduced records

For a reduced direct chunk:

```csharp
var reducedQuery = query.ReduceDimensions(storedChunk.Vector.Dimensions);

if (reducedQuery.Identity.EmbeddingSpaceFingerprint !=
    storedChunk.Identity.EmbeddingSpaceFingerprint)
{
    throw new InvalidOperationException("Embedding spaces differ.");
}
```

The same rule applies to a reduced `SingleEmbedding`.

The fingerprint check is more important than the raw dimension count.

## Re-embedding after a model-space change

A model update can preserve the same embedding-space fingerprint (for example, packaging-only changes) or change it. When it changes, application data must be re-embedded before new query vectors are compared with old document vectors. The library intentionally does not rewrite application databases automatically.
