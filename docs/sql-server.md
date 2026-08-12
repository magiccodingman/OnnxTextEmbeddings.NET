# SQL Server 2025 and Azure SQL vector search

Install:

```bash
dotnet add package OnnxTextEmbeddings.NET.SqlServer
```

The adapter targets SQL Server 2025 / Azure SQL native `VECTOR` operations while keeping final DefaultV1 ranking in core.

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

## Schema ownership

The adapter does not create application tables, migrations, vector indexes, tenant filters, or permissions. It accepts schema/column mappings and performs candidate retrieval against application-owned data.

Fingerprint filtering is always included by the provider. Application predicates can be supplied as trusted static SQL plus normal `SqlParameter` values.
