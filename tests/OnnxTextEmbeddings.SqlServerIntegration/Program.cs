using Microsoft.Data;
using Microsoft.Data.SqlClient;
using Microsoft.Data.SqlTypes;
using Microsoft.Extensions.DependencyInjection;
using OnnxTextEmbeddings;
using OnnxTextEmbeddings.SqlServer;

var connectionString = Environment.GetEnvironmentVariable("SQLSERVER_CONNECTION_STRING")
    ?? "Server=localhost,1433;User ID=sa;Password=OnnxTextEmbeddings!2026;Encrypt=False;TrustServerCertificate=True;Initial Catalog=master";

await using var connection = new SqlConnection(connectionString);
for (var attempt = 1; ; attempt++)
{
    try
    {
        await connection.OpenAsync();
        break;
    }
    catch (SqlException) when (attempt < 30)
    {
        await Task.Delay(TimeSpan.FromSeconds(3));
    }
}

var capabilities = await SqlServerSemanticSearch.GetCapabilitiesAsync(connection);
if (!capabilities.SupportsVectorType || capabilities.MaximumDimensions != 1998)
    throw new InvalidOperationException("SQL Server 2025 VECTOR capability probe failed.");

const string table = "onnx_semantic_candidates";
await using (var drop = new SqlCommand($"IF OBJECT_ID('{table}', 'U') IS NOT NULL DROP TABLE {table}", connection))
    await drop.ExecuteNonQueryAsync();
await using (var create = new SqlCommand($"""
    CREATE TABLE {table} (
        item_id nvarchar(64) NOT NULL,
        field_name nvarchar(64) NOT NULL,
        fingerprint nvarchar(128) NOT NULL,
        embedding vector(4) NOT NULL,
        record_json nvarchar(max) NOT NULL,
        field_weight real NOT NULL
    )
    """, connection))
    await create.ExecuteNonQueryAsync();

const string fingerprint = "sqlserver-integration-space";
await InsertAsync("backup", TextEmbeddingFor([1f, 0f, 0f, 0f], fingerprint));
await InsertAsync("network", TextEmbeddingFor([0f, 1f, 0f, 0f], fingerprint));

var services = new ServiceCollection();
services.AddLogging();
services.AddOnnxTextEmbeddings(options => options.Initialization.WarmupOnStartup = false);
await using var provider = services.BuildServiceProvider();
var search = new SqlServerSemanticSearch(provider.GetRequiredService<ISemanticCandidateReranker>());
var query = new QueryEmbedding
{
    Vector = EmbeddingVector.FromFloat32([0.99f, 0.01f, 0f, 0f]),
    Identity = Identity(fingerprint),
    SourceTokenCount = 2,
    InputTokenCount = 2
};
var result = await search.SearchAsync<string>(
    connection,
    query,
    new SqlServerCandidateQuery
    {
        Table = table,
        ItemKeyColumn = "item_id",
        FieldNameColumn = "field_name",
        FingerprintColumn = "fingerprint",
        VectorColumn = "embedding",
        RecordJsonColumn = "record_json",
        FieldWeightColumn = "field_weight",
        VectorDimensions = 4,
        SearchMode = SqlServerVectorSearchMode.Exact
    },
    new DatabaseSemanticSearchOptions { Top = 1, CandidateCount = 10 });

if (result.Results.Count != 1 || result.Results[0].Item != "backup")
    throw new InvalidOperationException("SQL Server VECTOR_DISTANCE candidate retrieval + DefaultV1 reranking returned the wrong item.");
if (result.Retrieval.Provider != "SQL Server/Azure SQL" || result.Retrieval.Approximate)
    throw new InvalidOperationException("Unexpected SQL Server retrieval diagnostics.");

var sourceValues = Enumerable.Range(0, 2048).Select(i => (float)Math.Sin(i + 1)).ToArray();
EmbeddingVectorMath.NormalizeInPlace(sourceValues);
var sourceEmbedding = TextEmbeddingFor(sourceValues, "sqlserver-jsp-space");
var sourceQuery = new QueryEmbedding
{
    Vector = EmbeddingVector.FromFloat32(sourceValues),
    Identity = sourceEmbedding.Identity,
    SourceTokenCount = 10,
    InputTokenCount = 10
};
var databaseEmbedding = sourceEmbedding.ToSqlServerVectorSpace();
var databaseQuery = sourceQuery.ToSqlServerVectorSpace();
if (databaseEmbedding.Vector.Dimensions != 1998 || databaseQuery.Vector.Dimensions != 1998)
    throw new InvalidOperationException("Jasper-sized vectors should automatically fit SQL Server's 1998-dimension limit.");
if (databaseEmbedding.Identity.EmbeddingSpaceFingerprint != databaseQuery.Identity.EmbeddingSpaceFingerprint)
    throw new InvalidOperationException("SQL Server document/query SRHT preparation entered different embedding spaces.");

Console.WriteLine($"PASS SQL Server 2025 exact native vector search + 2048 -> 1998 SRHT preparation. Approximate available: {capabilities.SupportsApproximateSearch}.");

async Task InsertAsync(string itemId, TextEmbedding embedding)
{
    await using var insert = new SqlCommand($"""
        INSERT INTO {table}(item_id, field_name, fingerprint, embedding, record_json, field_weight)
        VALUES (@item, @field, @fingerprint, @embedding, @json, @weight)
        """, connection);
    insert.Parameters.AddWithValue("@item", itemId);
    insert.Parameters.AddWithValue("@field", "content");
    insert.Parameters.AddWithValue("@fingerprint", embedding.Identity.EmbeddingSpaceFingerprint);
    insert.Parameters.Add(new SqlParameter("@embedding", SqlDbTypeExtensions.Vector)
    {
        Value = new SqlVector<float>(embedding.Vector.ToFloat32())
    });
    insert.Parameters.AddWithValue("@json", EmbeddingSerializer.SerializeJson(embedding));
    insert.Parameters.AddWithValue("@weight", 1f);
    await insert.ExecuteNonQueryAsync();
}

static TextEmbedding TextEmbeddingFor(float[] values, string fingerprint) => new()
{
    Vector = EmbeddingVector.FromFloat32(values),
    Identity = Identity(fingerprint),
    Source = new EmbeddingSource
    {
        DocumentTokenCount = 10,
        CharacterRange = new Utf16TextRange(0, 10),
        TokenRange = new TokenRange(0, 10),
        TokenCount = 10,
        TokenCapacity = 10
    },
    Chunk = new EmbeddingChunkInfo
    {
        Index = 0,
        Count = 1,
        BoundaryKind = ChunkBoundaryKind.WholeDocument,
        InputTokenCount = 10
    }
};

static EmbeddingIdentity Identity(string fingerprint) => new()
{
    ModelId = "integration",
    SourceRevision = "1",
    EmbeddingSpaceFingerprint = fingerprint,
    IsNormalized = true
};
