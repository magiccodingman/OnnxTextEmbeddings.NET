# Database-native semantic, lexical, and hybrid search

OnnxTextEmbeddings.NET keeps search meaning in core while allowing PostgreSQL, SQLite, and SQL Server/Azure SQL to perform broad retrieval with their native vector and full-text engines.

```text
portable + provider-specific prefilters
                  ↓
        ┌─────────┴─────────┐
        │                   │
native semantic       native lexical
   retrieval             retrieval
        │                   │
DefaultV1 item rank     provider rank
        └─────────┬─────────┘
                  ↓
          reciprocal-rank fusion
                  ↓
          optional post-filter
                  ↓
               Top N
```

Semantic database retrieval still follows the original division of responsibility: the database answers **which chunks plausibly matter**, while core `DefaultV1` decides how chunk-length confidence, strongest evidence, bounded supporting evidence, and semantic-field weights combine into item scores.

Lexical ranking is deliberately provider-native. Only SQLite and the in-memory engine are described as BM25; PostgreSQL and SQL Server use their own native full-text relevance systems.

## Official integrations

| Backend | Package | Semantic | Lexical |
|---|---|---|---|
| In memory | `OnnxTextEmbeddings.NET` | managed cosine + DefaultV1 | `BM25-v1` |
| PostgreSQL | `OnnxTextEmbeddings.NET.PgVector` | pgvector | `tsvector` + `ts_rank_cd`/`ts_rank` |
| SQLite | `OnnxTextEmbeddings.NET.SqliteVec` | sqlite-vec `vec0` | FTS5 `bm25()` |
| SQL Server 2025 / Azure SQL | `OnnxTextEmbeddings.NET.SqlServer` | `VECTOR_DISTANCE` / optional vector search | `FREETEXTTABLE` / `CONTAINSTABLE` |

Plain SQLite remains valid generic persistence. sqlite-vec is the official SQLite-native vector integration; FTS5 provides its native lexical half.

## Semantic candidate protocol

Database semantic providers normalize rows into `SemanticCandidate<TKey>` records carrying item key, field name/weight, complete direct `TextEmbedding`, and optional native similarity diagnostics. `ISemanticCandidateReranker` then executes the same DefaultV1 scoring used by in-memory semantic search.

`NativeSimilarity` is diagnostic/preselection information; it is not substituted for canonical final semantic scoring.

## Candidate over-fetching

A request for ten final items must not retrieve only ten chunks because supporting chunks/multiple fields can affect final item ranking. `DatabaseSemanticSearchOptions` therefore defaults to:

```text
max(100, Top × 10)
```

Callers may set `CandidateCount` explicitly. Advanced SearchQuery stages also have their own candidate limits, allowing semantic and lexical retrieval to use different budgets.

## Embedding-space safety

Every official vector provider filters by `EmbeddingSpaceFingerprint` during native candidate retrieval. Equal dimensions are not sufficient compatibility: derived SRHT spaces and unrelated models must never be mixed.

## Portable relational filtering

Database query mappings accept a `SearchFilter` plus a logical-to-physical `FilterColumns` map:

```csharp
new PgVectorCandidateQuery
{
    // ... vector schema mapping ...
    Filter = SearchFilter.And(
        SearchFilter.Equal("TenantId", tenantId),
        SearchFilter.GreaterThanOrEqual("UpdatedAt", cutoff)),
    FilterColumns = new Dictionary<string, string>
    {
        ["TenantId"] = "tenant_id",
        ["UpdatedAt"] = "updated_at"
    }
};
```

The provider compiles this filter into parameterized SQL so ordinary tenant/security/status/date predicates stay inside the database before vector/lexical retrieval.

The existing `AdditionalWhereSql` + provider-native command callback remains the deliberate escape hatch for predicates outside the portable vocabulary. Raw SQL fragments are trusted application code; dynamic user values belong in parameters.

## Field restrictions and weights per query

Advanced semantic stages can restrict candidate retrieval to selected logical semantic fields and multiply persisted field weights by query-specific weights before `DefaultV1` reranking.

Lexical stages similarly select/weight logical text fields using the native provider mechanism:

- PostgreSQL maps logical fields to `tsvector` A/B/C/D labels;
- SQLite maps them to positional FTS5 `bm25()` weights and column filters;
- SQL Server searches selected full-text columns separately and rank-fuses them with the caller's field weights.

## Hybrid RRF

Database-native raw scores are intentionally not normalized into a fake common scale. `SearchRankFusion` combines stage ranking positions using Reciprocal Rank Fusion. This makes the same hybrid query meaningful whether one lexical provider returned an FTS5 BM25 value, a PostgreSQL rank, or a SQL Server `RANK`.

See [Lexical, BM25, hybrid, and advanced search](lexical-hybrid-search.md) for the SearchQuery model, filter vocabulary, post-filter semantics, and provider examples.

## Direct chunk dimensional reduction

Database constraints sometimes require a smaller vector while chunk-level retrieval must remain intact. Use `TextEmbedding.ReduceDimensions(...)`, not `CombineToSingle()`:

```csharp
var reducedChunk = chunk.ReduceDimensions(1024, EmbeddingVectorFormat.Float32);
var reducedQuery = query.ReduceDimensions(1024);
```

The direct source/chunk metadata is preserved while both records enter the same deterministic child embedding space.
