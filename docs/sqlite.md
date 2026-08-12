# SQLite persistence

Plain SQLite remains a natural storage option for OnnxTextEmbeddings.NET because core records are storage-neutral.

A simple portable schema can store the versioned vector envelope and complete embedding metadata:

```sql
CREATE TABLE document_embeddings (
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    fingerprint TEXT NOT NULL,
    vector BLOB NOT NULL,
    record_json TEXT NOT NULL,
    PRIMARY KEY (document_id, chunk_index)
);
CREATE INDEX ix_embeddings_fingerprint ON document_embeddings(fingerprint);
```

`vector` can contain `EmbeddingSerializer.SerializeVector(...)`. `record_json` can contain `EmbeddingSerializer.SerializeJson(...)`.

For a small scoped working set, ordinary SQLite filters can select rows and core `ISemanticSearch` can rank the deserialized records in memory.

## Official SQLite vector search

When the goal is to keep cosine/KNN candidate work inside SQLite rather than materializing a large vector working set in .NET, use the official sqlite-vec adapter:

```bash
dotnet add package OnnxTextEmbeddings.NET.SqliteVec
```

See [SQLite/sqlite-vec](sqlite-vec.md).

The project therefore distinguishes:

```text
plain SQLite
  = generic portable persistence + optional in-memory search

SQLite + sqlite-vec
  = official SQLite-native semantic candidate search
```

There is no need for a separate plain-SQLite semantic-search abstraction whose only behavior would be selecting BLOBs and handing them to the already-existing in-memory scorer.
