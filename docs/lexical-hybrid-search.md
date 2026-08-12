# Lexical, BM25, hybrid, and advanced search

OnnxTextEmbeddings.NET supports semantic retrieval, lexical retrieval, or arbitrary combinations of both. The simple APIs remain available, but advanced callers can build a `SearchQuery` with global filters, stage-specific filters, weighted fields, independent candidate budgets, and reciprocal-rank fusion.

The core rule is:

```text
filters define eligibility
retrieval stages find evidence
fusion combines rankings
optional post-filter runs last
Top is applied last
```

## Lexical-only search

In-memory lexical search uses the versioned `BM25-v1` profile. It does not require an embedding model.

```csharp
var lexical = new InMemoryLexicalSearch();
var results = await lexical.SearchAsync(
    "postgresql backup",
    pages,
    page =>
    [
        LexicalField.Create("title", page.Title, 8f),
        LexicalField.Create("description", page.Description, 3f),
        LexicalField.Create("body", page.Body, 1f)
    ],
    new LexicalSearchRequest { Top = 10 });
```

Applications that want only in-memory lexical search can register just:

```csharp
builder.Services.AddOnnxTextEmbeddingsLexicalSearch();
```

That registration does not register `ITextEmbeddingService`, does not download Jasper, and does not initialize ONNX Runtime.

`BM25-v1` currently uses `k1 = 1.2` and `b = 0.75` by default. Both values are request-configurable. Field weights are applied to term frequency before the BM25 saturation calculation so literal matches in titles/categories/tags can matter more than the same terms buried in a long body.

## Advanced SearchQuery

A `SearchQuery` is a retrieval plan rather than a mode enum:

```csharp
var query = SearchQuery.Create("postgresql backup")
    .Where(SearchFilter.And(
        SearchFilter.Equal("TenantId", tenantId),
        SearchFilter.NotEqual("Status", "Deleted"),
        SearchFilter.GreaterThanOrEqual("UpdatedAt", cutoff)))
    .Add(SearchRetrievalStage.Semantic(
            "semantic-content",
            SearchFieldWeight.Create("content", 1f))
        .Candidates(250))
    .Add(SearchRetrievalStage.Lexical(
            "lexical-metadata",
            SearchFieldWeight.Create("title", 8f),
            SearchFieldWeight.Create("category", 5f),
            SearchFieldWeight.Create("description", 2f),
            SearchFieldWeight.Create("body", 1f))
        .Candidates(150))
    .UseReciprocalRankFusion()
    .Take(20);
```

`SearchRetrievalStage` supports semantic and lexical stages today. The plan model deliberately allows more than one stage of either kind; a caller may run separate semantic fields or separate lexical strategies with different filters/weights and fuse all of them.

## Global, stage, and post filters

A global filter applies to every retrieval stage:

```csharp
query.Where(SearchFilter.Equal("TenantId", tenantId));
```

A stage filter applies only to one retriever:

```csharp
query.Add(
    SearchRetrievalStage.Semantic("published-semantic")
        .Where(SearchFilter.Equal("HasEmbedding", true)));
```

This is useful when, for example, lexical search can see a newly inserted row before its embedding has been generated.

`PostWhere(...)` is intentionally separate. It runs after retrieval/fusion, which means it can reduce the number of final results because excluded rows have already consumed candidate positions. Prefer global/stage prefilters whenever the condition can be expressed before retrieval.

Database advanced-search APIs require a `postFilterValues` callback when a `PostFilter` is present because only the application knows how to fetch/project arbitrary late-filter metadata for returned keys.

## Portable filter vocabulary

`SearchFilter` currently supports:

```text
Equal / NotEqual
GreaterThan / GreaterThanOrEqual
LessThan / LessThanOrEqual
In / NotIn
IsNull / IsNotNull
Contains / StartsWith / EndsWith
And / Or / Not
```

Core evaluates the same tree in memory. Database providers translate it to parameterized SQL through a logical-field-to-column mapping.

```csharp
FilterColumns = new Dictionary<string, string>
{
    ["TenantId"] = "tenant_id",
    ["UpdatedAt"] = "updated_at"
};
```

Filter values are emitted as command parameters, never string-concatenated into provider SQL.

The portable vocabulary is deliberately finite. PostgreSQL JSONB operators, SQL Server-specific predicates, SQLite custom functions, security predicates, and other backend-specific capabilities belong in the provider's `AdditionalWhereSql` + command-parameter callback escape hatch. Portable queries remain portable; intentional provider-native queries are allowed to be provider-native.

## Hybrid search uses reciprocal-rank fusion

Raw semantic and lexical scores are not comparable:

- cosine/DefaultV1 has its own meaning;
- in-memory BM25 has corpus-dependent magnitude;
- SQLite FTS5 has its own BM25 score convention;
- PostgreSQL `ts_rank`/`ts_rank_cd` are PostgreSQL relevance scores;
- SQL Server Full-Text `RANK` is SQL Server relevance.

For that reason hybrid search does **not** compute something like `0.5 * cosine + 0.5 * lexicalScore`.

Instead it uses Reciprocal Rank Fusion (RRF):

```text
contribution = stageWeight / (rankConstant + rank)
finalScore   = sum(contributions from every stage)
```

The default rank constant is 60. A result supported by both semantic and lexical evidence can therefore outrank a result that is first in only one list, without pretending their raw score units are interchangeable.

Each `SearchResult<T>` includes `SearchStageContribution` diagnostics containing the stage name, retrieval kind, source rank, source raw score, fusion contribution, and available semantic/lexical match details.

## In-memory advanced search

`IAdvancedSearch` accepts a `SearchDocument` projection:

```csharp
var results = await advanced.SearchAsync(
    query,
    pages,
    page => new SearchDocument()
        .Value("TenantId", page.TenantId)
        .Value("Status", page.Status)
        .Text("title", page.Title)
        .Text("description", page.Description)
        .Text("body", page.Body)
        .Semantic("content", page.Embeddings));
```

The query embedding is created lazily. A lexical-only `SearchQuery` never calls `EmbedQueryAsync`.

## Database provider behavior

The common query semantics are stable, but lexical ranking stays native to each database rather than pretending every engine implements BM25.

| Backend | Semantic retrieval | Lexical retrieval | Lexical weighting |
|---|---|---|---|
| In memory | managed cosine + DefaultV1 | `BM25-v1` | arbitrary per field |
| PostgreSQL | pgvector | `tsvector` + `ts_rank_cd`/`ts_rank` | mapped to A/B/C/D `tsvector` weights |
| SQLite | sqlite-vec `vec0` | FTS5 `bm25()` | native positional FTS5 BM25 weights |
| SQL Server / Azure SQL | `VECTOR_DISTANCE` / vector search | `FREETEXTTABLE` / `CONTAINSTABLE` | field searches fused by weighted RRF |

### PostgreSQL

`PgVectorSearchPlan` combines a `PgVectorCandidateQuery` and/or `PgVectorLexicalQuery`. Lexical mappings identify the prebuilt `tsvector` column and map logical fields to A/B/C/D labels. Query text can use web-search, plain, phrase, or native tsquery parsing.

### SQLite

`SqliteVecSearchPlan` combines a vec0 candidate mapping and/or `SqliteFts5Query`. FTS5 requires `ColumnOrder` because `bm25()` column weights are positional. `Plain` mode tokenizes/quotes ordinary text; `NativeSyntax` intentionally exposes FTS5 query operators to advanced callers.

### SQL Server / Azure SQL

`SqlServerSearchPlan` combines vector and Full-Text mappings. `FREETEXTTABLE` is the normal natural-language lexical mode; `CONTAINSTABLE` exposes SQL Server's advanced full-text query syntax. SQL Server does not expose an FTS5-style arbitrary BM25 weight vector for columns, so weighted logical fields are searched independently and fused by rank within the lexical stage before the outer SearchQuery fusion.

## Candidate counts

Each stage can explicitly set a candidate budget:

```csharp
SearchRetrievalStage.Semantic(...).Candidates(300)
SearchRetrievalStage.Lexical(...).Candidates(150)
```

When omitted, advanced search uses `max(100, Top * 10)` as the retrieval-stage target. Database semantic retrieval may over-fetch again internally because final `DefaultV1` item scoring can need multiple direct chunk candidates per item.

Candidate limits are recall/cost controls, not correctness guarantees: fusion and reranking cannot recover evidence a retrieval stage never returned.
