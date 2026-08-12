# SQLite and sqlite-vec

Install:

```bash
dotnet add package OnnxTextEmbeddings.NET.SqliteVec
```

The adapter uses sqlite-vec `vec0` tables to keep broad KNN/cosine candidate search inside SQLite, then sends only the bounded candidates through core DefaultV1 reranking.

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
        StorageKind = SqliteVecStorageKind.Float32
    },
    new DatabaseSemanticSearchOptions { Top = 10 });
```

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

## Version policy

sqlite-vec is still pre-v1 upstream. For that reason its dependency and SQL behavior are isolated inside `OnnxTextEmbeddings.NET.SqliteVec` rather than exposed through core.

The project pins a tested sqlite-vec package version and runs real integration tests across supported desktop platforms. A future upstream breaking change can therefore be absorbed by the adapter instead of silently changing core contracts.
