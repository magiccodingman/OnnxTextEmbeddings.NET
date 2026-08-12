# SQLite and sqlite-vec

Install:

```bash
dotnet add package OnnxTextEmbeddings.NET.SqliteVec
```

The adapter uses sqlite-vec `vec0` tables to keep broad KNN/cosine candidate search inside SQLite, then sends only the bounded candidates through core DefaultV1 reranking. The same package also supports SQLite FTS5 lexical search and semantic + lexical hybrid plans.

## Load the extension

The pinned sqlite-vec NuGet dependency provides the native extension binaries. Load it before opening the connection:

```csharp
await using var connection = new SqliteConnection("Data Source=embeddings.db");
connection.LoadOnnxTextEmbeddingsSqliteVec();
await connection.OpenAsync();

var capabilities = await connection.GetSqliteVecCapabilitiesAsync();
Console.WriteLine(capabilities.Version);
```

The integration is continuously exercised on Linux, Windows, and macOS.

## FP32 table

Example:

```sql
CREATE VIRTUAL TABLE document_embeddings USING vec0(
    document_id text,
    field_name text,
    fingerprint text,
    tenant_id integer,
    embedding float[2048] distance_metric=cosine,
    +record_json text,
    +field_weight float
);
```

Then search through `SqliteVecSemanticSearch`:

```csharp
var result = await sqliteSearch.SearchAsync<string>(
    connection,
    query,
    new SqliteVecCandidateQuery
    {
        Table = "document_embeddings",
        ItemKeyColumn = "document_id",
        FieldNameColumn = "field_name",
        FingerprintColumn = "fingerprint",
        VectorColumn = "embedding",
        RecordJsonColumn = "record_json",
        FieldWeightColumn = "field_weight",
        StorageKind = SqliteVecStorageKind.Float32,
        FilterColumns = new Dictionary<string, string>
        {
            ["TenantId"] = "tenant_id"
        },
        Filter = SearchFilter.Equal("TenantId", tenantId)
    },
    new DatabaseSemanticSearchOptions { Top = 10 });
```

### Filterable vec0 columns

Portable semantic filters must map to normal sqlite-vec metadata/partition columns. A `+auxiliary` column is payload-only: it can be returned by `SELECT`, but sqlite-vec does not allow it as a KNN filter constraint.

Keep large payloads such as serialized records in auxiliary columns, and declare fields such as tenant, status, category, or other retrieval-time predicates as metadata/partition columns when they need to participate in semantic filtering.

vec0's KNN planner intentionally accepts a narrower metadata-filter grammar than ordinary SQLite SQL. `SqliteVecSemanticSearch` therefore uses the fast `MATCH`/KNN path when the filter can be represented faithfully as supported simple comparisons. Richer portable expressions—such as `OR`, `NOT`, null-sensitive complements, multiple constraints on the same metadata column, or provider-specific `AdditionalWhereSql`—automatically use an exact filtered scan with sqlite-vec's scalar `vec_distance_cosine()` function. This preserves the shared filter semantics at the cost of the expected scan-based performance tradeoff. `SemanticCandidateRetrievalInfo.Mode` reports `.../KNN` or `.../FilteredExactScan` so the choice is observable.

## Native INT8 search

sqlite-vec also supports native signed INT8 vectors and cosine search:

```sql
embedding int8[2048] distance_metric=cosine
```

Use:

```csharp
StorageKind = SqliteVecStorageKind.Int8
```

OnnxTextEmbeddings.NET's symmetric per-vector INT8 representation is a natural cosine fit because a positive per-vector scalar cancels during cosine normalization. The adapter passes the quantized coordinate bytes to sqlite-vec and core reranks the returned complete embedding records.

INT4 and FP16 do not map directly to sqlite-vec native storage in this adapter; convert to INT8 or FP32 for the native candidate column while retaining any preferred portable representation separately if needed.

## FTS5 lexical search

`SqliteFts5LexicalSearch` runs lexical retrieval through SQLite FTS5 and its native `bm25()` ranking. Logical fields map to FTS5 columns and per-query field weights map to the positional BM25 weight vector.

```csharp
var lexical = await fts.SearchAsync<string>(
    connection,
    "postgresql backup",
    lexicalMapping,
    [SearchFieldWeight.Create("title", 8), SearchFieldWeight.Create("body", 1)],
    new DatabaseLexicalSearchOptions { Top = 10 });
```

Plain query mode safely tokenizes/quotes ordinary user text. Native-syntax mode is an explicit escape hatch for callers that intentionally want FTS5 operators, column syntax, or `NEAR` expressions.

## Hybrid and advanced search

`SqliteVecAdvancedSearch` executes a shared `SearchQuery` against a `SqliteVecSearchPlan` containing semantic and/or lexical mappings. Each retrieval stage keeps its own filters, fields, weights, and candidate budget; the returned rankings are fused with reciprocal-rank fusion rather than mixing raw cosine and FTS5 BM25 scores.

```csharp
var plan = new SqliteVecSearchPlan
{
    Semantic = semanticMapping,
    Lexical = lexicalMapping
};

var results = await advanced.SearchAsync<string>(connection, searchQuery, plan);
```

See [Lexical, BM25, hybrid, and advanced search](lexical-hybrid-search.md) for the provider-neutral query model.

## Version policy

sqlite-vec is still pre-v1 upstream. For that reason its dependency and SQL behavior are isolated inside `OnnxTextEmbeddings.NET.SqliteVec` rather than exposed through core.

The project pins a tested sqlite-vec package version and runs real integration tests across supported desktop platforms. A future upstream breaking change can therefore be absorbed by the adapter instead of silently changing core contracts.
