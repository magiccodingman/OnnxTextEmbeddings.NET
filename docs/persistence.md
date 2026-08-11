# Persistence

The core package is storage-neutral. Persistence belongs to the application, but **persist the embedding record, not just an anonymous float array**.

## Store these fields

At minimum preserve:

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

The fingerprint is critical. Search rejects stored vectors from a different embedding space.

## Binary vector payload

```csharp
byte[] payload = EmbeddingSerializer.SerializeVector(embedding.Vector);
EmbeddingVector vector = EmbeddingSerializer.DeserializeVector(payload);
```

This is a versioned portable payload suitable for SQLite BLOB, SQL Server VARBINARY, PostgreSQL BYTEA, or files.

## JSON record

```csharp
string json = EmbeddingSerializer.SerializeJson(embedding);
TextEmbedding restored = EmbeddingSerializer.DeserializeJson(json);
```

JSON is convenient when portability/debuggability matters more than raw row size.

## Re-embedding after a model-space change

A model update can preserve the same embedding-space fingerprint (for example, packaging-only changes) or change it. When it changes, application data must be re-embedded before new query vectors are compared with old document vectors. The library intentionally does not rewrite application databases automatically.
