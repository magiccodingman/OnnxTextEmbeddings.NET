# PostgreSQL and pgvector

Install the optional adapter only when PostgreSQL-native vector operations are useful:

```bash
dotnet add package OnnxTextEmbeddings.NET.PgVector
```

The core package stays free of PostgreSQL dependencies.

## Two storage modes

### Portable bytes

Store the core binary vector in `BYTEA` and rank scoped candidates in application memory. This preserves INT4/INT8 compactness exactly.

### Native pgvector

Convert an `EmbeddingVector` to pgvector `Vector`/`HalfVector` using the adapter when SQL-side cosine candidate selection is more important than compact INT4/INT8 storage.

Native pgvector storage is especially useful for broad candidate preselection:

```text
normal WHERE filters
       ↓
SQL cosine distance / top candidates
       ↓
load complete TextEmbedding metadata
       ↓
DefaultV1 final grouping + supporting evidence in application code
```

This hybrid design lets PostgreSQL do what it is good at—fast broad vector candidate search—while the library preserves its document/field evidence semantics.

## Fingerprint filtering

Always filter candidates to the query's embedding-space fingerprint before ranking. Equal dimensions do not imply compatible embedding spaces.

## Multi-field applications

A practical schema normally stores field name, chunk number, fingerprint, token metadata, and native vector beside the owning application row. Candidate queries can apply tenant/project/permission predicates before cosine ordering, then group returned rows into `SemanticField` values for final scoring.
