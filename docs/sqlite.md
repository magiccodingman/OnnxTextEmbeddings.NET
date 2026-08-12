# SQLite

SQLite is a natural fit for the library's target scale: store compact vectors as BLOBs and rank a scoped candidate set in memory.

## Suggested schema

```sql
CREATE TABLE document_embeddings (
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    fingerprint TEXT NOT NULL,
    vector BLOB NOT NULL,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY (document_id, chunk_index)
);
CREATE INDEX ix_embeddings_fingerprint ON document_embeddings(fingerprint);
```

`vector` can contain `EmbeddingSerializer.SerializeVector(...)`. Keep the rest of `TextEmbedding` as structured columns or JSON depending on the application's query needs.

## Search flow

1. Apply ordinary SQL filters first (tenant, project, category, date, permissions).
2. Load the resulting candidate embeddings.
3. Create one `QueryEmbedding`.
4. Run `ISemanticSearch` in memory.

This avoids requiring a SQLite vector extension for the small-to-medium workloads this package targets.

## Compact storage

At 2048 dimensions, packed INT4 is roughly 1 KiB of vector bytes per chunk and INT8 roughly 2 KiB, before record metadata. Model precision does not dictate storage precision.
