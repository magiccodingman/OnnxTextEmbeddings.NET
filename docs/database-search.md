# Database-native semantic search

OnnxTextEmbeddings.NET keeps one semantic ranking implementation while allowing databases with native vector support to perform broad candidate retrieval.

```text
QueryEmbedding
      ↓
relational filters + embedding-space fingerprint
      ↓
database-native cosine / KNN
      ↓
bounded direct-chunk candidates
      ↓
core DefaultV1 reranking
      ↓
SemanticSearchResult<TKey>
```

The database answers **which chunks plausibly matter**. Core answers **how chunk length confidence, strongest evidence, bounded supporting evidence, and semantic-field weights combine into the final item score**.

This prevents PostgreSQL, SQLite, and SQL Server from each acquiring their own subtly different implementation of `DefaultV1`.

## Official search integrations

| Backend | Package | Candidate search |
|---|---|---|
| In memory | `OnnxTextEmbeddings.NET` | Managed cosine + DefaultV1 |
| PostgreSQL | `OnnxTextEmbeddings.NET.PgVector` | pgvector cosine/KNN |
| SQLite | `OnnxTextEmbeddings.NET.SqliteVec` | sqlite-vec `vec0` KNN |
| SQL Server 2025 / Azure SQL | `OnnxTextEmbeddings.NET.SqlServer` | `VECTOR_DISTANCE` and optional approximate vector search |

Plain SQLite remains a perfectly valid persistence store. It simply is not a separate first-class vector-search provider: applications can deserialize scoped rows and use core in-memory search, while sqlite-vec is the official SQLite-native candidate path.

## Shared candidate protocol

Database providers normalize native rows into `SemanticCandidate<TKey>` records. A candidate carries:

- item key;
- field name and field weight;
- the complete direct `TextEmbedding` record;
- optional native similarity diagnostics.

`ISemanticCandidateReranker` then executes the same DefaultV1 scoring used by normal in-memory search.

`NativeSimilarity` is diagnostic/preselection information. It is not substituted for the canonical final score.

## Candidate over-fetching

A request for ten final items must not retrieve only ten chunks. Supporting chunks and multiple fields can affect final item ranking.

`DatabaseSemanticSearchOptions` therefore defaults candidate count to:

```text
max(100, Top × 10)
```

Callers can override `CandidateCount` explicitly when recall, database cost, or dataset shape warrants a different value.

```csharp
var options = new DatabaseSemanticSearchOptions
{
    Top = 10,
    CandidateCount = 250
};
```

Final reranking cannot recover a chunk that the candidate stage never returned. Approximate indexes therefore expose their approximate nature through retrieval diagnostics.

## Fingerprint safety

Every official provider filters by `EmbeddingSpaceFingerprint` during native candidate retrieval.

Equal dimensions do not make vectors compatible. A Jasper 512-dimensional `SRHT-v1` vector and some unrelated 512-dimensional vector are different spaces and must never be compared as if they were interchangeable.

## Relational filtering

Core deliberately does not invent a cross-database WHERE-clause DSL.

Each provider accepts its own static `AdditionalWhereSql` fragment and a parameter-configuration callback using that provider's native command type. The SQL fragment is treated as trusted application code; dynamic user input belongs in parameters.

This keeps tenant/project/security/date filtering in the database without turning OnnxTextEmbeddings.NET into an ORM.

## Direct chunk dimensional reduction

Database constraints sometimes require a smaller vector while chunk-level retrieval must remain intact.

Use `TextEmbedding.ReduceDimensions(...)` rather than `CombineToSingle()`:

```csharp
TextEmbedding reduced = chunk.ReduceDimensions(
    1024,
    EmbeddingVectorFormat.Float32);

QueryEmbedding reducedQuery = query.ReduceDimensions(1024);
```

The source range, chunk metadata, text, and context remain direct-chunk metadata. Only the vector coordinate space changes, and both document/query identities derive the same deterministic child fingerprint when the same SRHT profile and dimensions are used.

Aggregation is only for consumers that genuinely require one semantic vector for an entire multi-chunk document.
