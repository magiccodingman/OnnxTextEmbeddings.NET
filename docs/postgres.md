# PostgreSQL and pgvector

Install the optional adapter when PostgreSQL-native semantic, lexical, or hybrid search is useful:

```bash
dotnet add package OnnxTextEmbeddings.NET.PgVector
```

The core package remains free of PostgreSQL dependencies.

## Native candidate search

`PgVectorSemanticSearch` performs PostgreSQL filtering and pgvector cosine candidate selection, then sends the bounded direct-chunk set through core DefaultV1 reranking.

```csharp
var search = serviceProvider.GetRequiredService<PgVectorSemanticSearch>();

var results = await search.SearchAsync<string>(
    connection,
    query,
    new PgVectorCandidateQuery
    {
        Table = "document_embeddings",
        ItemKeyColumn = "document_id",
        FieldNameColumn = "field_name",
        FingerprintColumn = "fingerprint",
        VectorColumn = "embedding",
        RecordJsonColumn = "record_json",
        FieldWeightColumn = "field_weight",
        SearchMode = PgVectorSearchMode.Exact
    },
    new DatabaseSemanticSearchOptions { Top = 10 });
```

Register the provider with:

```csharp
services.AddOnnxTextEmbeddings();
services.AddOnnxTextEmbeddingsPgVector();
```

## Exact and index-assisted modes

`PgVectorSearchMode.Exact` is the conservative mode. It evaluates the filtered rows' cosine distances inside a materialized CTE before applying Top-K ordering. That keeps normal relational filtering available while preventing an approximate pgvector KNN index from becoming the candidate source. It does not change PostgreSQL planner/session settings and is safe to use inside a caller-owned transaction.

`PgVectorSearchMode.Approximate` uses the direct pgvector distance ordering and leaves the planner free to use configured pgvector indexes such as HNSW/IVFFlat.

The returned `SemanticCandidateRetrievalInfo` records which mode produced the candidates.

## Full-text lexical search

`PgVectorLexicalSearch` uses PostgreSQL `tsvector`/`tsquery` retrieval and `ts_rank_cd` or `ts_rank` scoring. Application-owned schemas build and index the `tsvector`; the adapter maps logical search fields to PostgreSQL A/B/C/D text-search labels.

```csharp
var lexical = await lexicalSearch.SearchAsync<string>(
    connection,
    "postgresql backup",
    new PgVectorLexicalQuery
    {
        Table = "documents",
        ItemKeyColumn = "document_id",
        SearchVectorColumn = "search_vector",
        Fields =
        [
            new PgTextSearchField("title", PgTextSearchWeight.A),
            new PgTextSearchField("body", PgTextSearchWeight.D)
        ]
    },
    [SearchFieldWeight.Create("title", 8), SearchFieldWeight.Create("body", 1)],
    new DatabaseLexicalSearchOptions { Top = 10 });
```

Logical field weights are not restricted to PostgreSQL's `0..1` rank-weight range. The adapter normalizes the selected logical weights proportionally before passing them to PostgreSQL, so a caller can express `title = 8` and `body = 1` while PostgreSQL receives the same relative `1.0 : 0.125` preference.

Web-search, plain, phrase, and native tsquery parsing modes are available. Native mode intentionally exposes PostgreSQL's full text-search syntax.

## Hybrid and advanced search

`PgVectorAdvancedSearch` executes the common `SearchQuery` model against a `PgVectorSearchPlan` containing semantic and/or lexical mappings.

```csharp
var plan = new PgVectorSearchPlan
{
    Semantic = semanticMapping,
    Lexical = lexicalMapping
};

var results = await advanced.SearchAsync<string>(connection, searchQuery, plan);
```

Each stage can have independent fields, weights, filters, and candidate counts. Semantic and PostgreSQL lexical rankings are fused with reciprocal-rank fusion, so raw cosine/DefaultV1 and `ts_rank[_cd]` values are never treated as if they share a score scale.

See [Lexical, BM25, hybrid, and advanced search](lexical-hybrid-search.md) for the provider-neutral query model.

## Jasper and pgvector dimensions

Jasper returns 2048 dimensions.

A PostgreSQL `vector(2048)` column can store the native representation. Current pgvector vector-index dimensional limits are tighter than storage limits, so applications that require indexed Jasper retrieval have two practical choices:

- use `halfvec(2048)` through `PgVectorStorageKind.HalfVector`; or
- deterministically reduce direct chunks and queries to an indexable child space such as 1024 dimensions.

The adapter supports both pgvector `Vector` and `HalfVector` conversions.

## Portable bytes

Applications can also store `EmbeddingSerializer.SerializeVector(...)` in `BYTEA`. That preserves INT4/INT8/FP16/FP32 exactly, but native candidate search requires a pgvector-compatible vector column as well.

A schema may keep both when compact portable persistence and native PostgreSQL search are both valuable.

## Filtering

Semantic and lexical mappings can expose logical `FilterColumns`; the shared `SearchFilter` tree is translated to parameterized PostgreSQL predicates. The semantic provider always adds the query fingerprint predicate itself.

For PostgreSQL-specific predicates outside the portable filter vocabulary, use a trusted static `AdditionalWhereSql` fragment and the command-parameter callback. Do not interpolate user input into `AdditionalWhereSql`; bind it as normal Npgsql parameters.
