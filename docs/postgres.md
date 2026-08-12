# PostgreSQL and pgvector

Install the optional adapter when PostgreSQL-native vector candidate search is useful:

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

The provider always adds the query fingerprint predicate itself. Application filters can be added using a trusted static `AdditionalWhereSql` fragment and parameter callback.

Do not interpolate user input into `AdditionalWhereSql`; bind it as normal Npgsql parameters.
