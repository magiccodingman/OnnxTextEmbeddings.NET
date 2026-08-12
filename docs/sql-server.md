# SQL Server 2025 and Azure SQL vector and full-text search

Install:

```bash
dotnet add package OnnxTextEmbeddings.NET.SqlServer
```

The adapter targets SQL Server 2025 / Azure SQL native `VECTOR` operations while keeping final DefaultV1 ranking in core. It also supports SQL Server Full-Text Search and semantic + lexical hybrid plans.

## The 1998-dimension boundary

SQL Server's current `VECTOR` type supports at most **1998 dimensions**. Jasper produces 2048.

The adapter therefore provides direct document/query preparation helpers:

```csharp
TextEmbedding dbChunk = chunk.ToSqlServerVectorSpace();
QueryEmbedding dbQuery = query.ToSqlServerVectorSpace();
```

For Jasper, the default is:

```text
2048 native dimensions
        ↓
deterministic SRHT-v1
        ↓
1998 dimensions
        ↓
FP32 SQL Server VECTOR
```

Both sides derive the same reduced-space fingerprint. Chunk metadata remains direct chunk metadata; documents are **not** combined into one vector merely to satisfy SQL Server.

Callers can choose a smaller explicit profile such as 1536 or 1024 dimensions when that better fits their storage/index strategy.

## Exact search is the default

`SqlServerVectorSearchMode.Exact` uses native `VECTOR_DISTANCE('cosine', ...)` ordering.

```csharp
var result = await sqlServerSearch.SearchAsync<string>(
    connection,
    query,
    new SqlServerCandidateQuery
    {
        Table = "dbo.document_embeddings",
        ItemKeyColumn = "document_id",
        FieldNameColumn = "field_name",
        FingerprintColumn = "fingerprint",
        VectorColumn = "embedding",
        RecordJsonColumn = "record_json",
        VectorDimensions = 1998,
        SearchMode = SqlServerVectorSearchMode.Exact
    },
    new DatabaseSemanticSearchOptions { Top = 10 });
```

The query is automatically transformed to the configured database vector dimensions before native retrieval and final reranking.

## Approximate search

`SqlServerVectorSearchMode.Approximate` uses SQL Server/Azure SQL's native approximate vector-search surface and is explicit opt-in.

These capabilities are still preview-sensitive and differ by SQL Server/Azure SQL deployment. The adapter therefore does not automatically enable preview database features or create indexes.

Use:

```csharp
SqlServerVectorCapabilities capabilities =
    await SqlServerSemanticSearch.GetCapabilitiesAsync(connection);
```

to inspect native vector support, preview-feature state, and approximate-search availability before selecting that mode.

## Full-Text lexical search

`SqlServerFullTextSearch` uses `FREETEXTTABLE` for normal natural-language retrieval or `CONTAINSTABLE` when an application explicitly wants SQL Server full-text query syntax.

```csharp
var lexical = await fullText.SearchAsync<string>(
    connection,
    "postgresql backup",
    new SqlServerLexicalQuery
    {
        Table = "dbo.documents",
        ItemKeyColumn = "document_id",
        FullTextKeyColumn = "id",
        Fields =
        [
            new SqlServerFullTextField("title", "title"),
            new SqlServerFullTextField("body", "body")
        ]
    },
    [SearchFieldWeight.Create("title", 8), SearchFieldWeight.Create("body", 1)],
    new DatabaseLexicalSearchOptions { Top = 10 });
```

SQL Server does not expose an FTS5-style arbitrary BM25 column-weight vector. When several logical fields have different weights, the adapter performs the native full-text retrieval for those fields and combines their rank positions with the requested logical weights instead of pretending SQL Server's raw `RANK` values are portable.

Full-Text Search must be installed/enabled by the SQL Server deployment and full-text indexes must live in an application/user database; SQL Server does not allow full-text search in `master`, `tempdb`, or `model`.

## Hybrid and advanced search

`SqlServerAdvancedSearch` executes the common `SearchQuery` model against a `SqlServerSearchPlan` containing semantic and/or lexical mappings.

```csharp
var plan = new SqlServerSearchPlan
{
    Semantic = semanticMapping,
    Lexical = lexicalMapping
};

var results = await advanced.SearchAsync<string>(connection, searchQuery, plan);
```

Each stage can have independent fields, weights, filters, and candidate counts. The outer hybrid ranking uses reciprocal-rank fusion, so vector similarity/DefaultV1 and SQL Server Full-Text `RANK` are never treated as directly comparable numeric scores.

See [Lexical, BM25, hybrid, and advanced search](lexical-hybrid-search.md) for the provider-neutral query model.

## Schema ownership

The adapter does not create application tables, migrations, vector indexes, full-text catalogs/indexes, tenant filters, or permissions. It accepts schema/column mappings and performs retrieval against application-owned data.

Semantic and lexical mappings can expose logical `FilterColumns`; the shared `SearchFilter` tree is translated to parameterized SQL. Fingerprint filtering is always included by the semantic provider. SQL Server-specific predicates can still be supplied as trusted static SQL plus normal `SqlParameter` values.
