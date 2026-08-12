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
    try { await connection.OpenAsync(); break; }
    catch (SqlException) when (attempt < 30) { await Task.Delay(TimeSpan.FromSeconds(3)); }
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
        tenant_id int NOT NULL,
        field_name nvarchar(64) NOT NULL,
        fingerprint nvarchar(128) NOT NULL,
        embedding vector(4) NOT NULL,
        record_json nvarchar(max) NOT NULL,
        field_weight real NOT NULL
    )
    """, connection))
    await create.ExecuteNonQueryAsync();

const string fingerprint = "sqlserver-integration-space";
await InsertAsync("backup", 1, TextEmbeddingFor([1f, 0f, 0f, 0f], fingerprint));
await InsertAsync("network", 1, TextEmbeddingFor([0f, 1f, 0f, 0f], fingerprint));
await InsertAsync("other-tenant", 2, TextEmbeddingFor([1f, 0f, 0f, 0f], fingerprint));

const string lexicalTable = "dbo.onnx_lexical_items";
await using (var drop = new SqlCommand("IF OBJECT_ID('dbo.onnx_lexical_items', 'U') IS NOT NULL DROP TABLE dbo.onnx_lexical_items", connection))
    await drop.ExecuteNonQueryAsync();
await using (var create = new SqlCommand("""
    CREATE TABLE dbo.onnx_lexical_items (
        id int NOT NULL CONSTRAINT PK_onnx_lexical_items PRIMARY KEY,
        item_id nvarchar(64) NOT NULL,
        tenant_id int NOT NULL,
        title nvarchar(4000) NOT NULL,
        body nvarchar(max) NOT NULL
    )
    """, connection))
    await create.ExecuteNonQueryAsync();
await InsertLexicalAsync(1, "backup", 1, "PostgreSQL Backup", "Restore and disaster recovery procedures.");
await InsertLexicalAsync(2, "network", 1, "Networking", "Firewall and routing documentation.");
await InsertLexicalAsync(3, "body-heavy", 1, "Database Operations", "PostgreSQL backup PostgreSQL backup PostgreSQL backup procedures.");
await InsertLexicalAsync(4, "other-tenant", 2, "PostgreSQL Backup", "Exact title in another tenant.");

var fullTextInstalled = Convert.ToInt32(await new SqlCommand("SELECT FULLTEXTSERVICEPROPERTY('IsFullTextInstalled')", connection).ExecuteScalarAsync());
if (fullTextInstalled != 1)
    throw new InvalidOperationException("SQL Server container does not have Full-Text Search installed.");
var catalogExists = Convert.ToInt32(await new SqlCommand("SELECT COUNT(*) FROM sys.fulltext_catalogs WHERE name = N'ote_integration_catalog'", connection).ExecuteScalarAsync());
if (catalogExists == 0)
    await new SqlCommand("CREATE FULLTEXT CATALOG ote_integration_catalog AS DEFAULT", connection).ExecuteNonQueryAsync();
await new SqlCommand("""
    CREATE FULLTEXT INDEX ON dbo.onnx_lexical_items(
        title LANGUAGE 1033,
        body LANGUAGE 1033
    ) KEY INDEX PK_onnx_lexical_items
      WITH CHANGE_TRACKING AUTO
    """, connection).ExecuteNonQueryAsync();
for (var attempt = 0; attempt < 60; attempt++)
{
    var status = Convert.ToInt32(await new SqlCommand("SELECT FULLTEXTCATALOGPROPERTY('ote_integration_catalog', 'PopulateStatus')", connection).ExecuteScalarAsync());
    if (status == 0) break;
    await Task.Delay(TimeSpan.FromSeconds(1));
}

var services = new ServiceCollection();
services.AddLogging();
services.AddOnnxTextEmbeddings(options => options.Initialization.WarmupOnStartup = false);
services.AddOnnxTextEmbeddingsSqlServer();
await using var provider = services.BuildServiceProvider();
var search = provider.GetRequiredService<SqlServerSemanticSearch>();
var lexicalSearch = provider.GetRequiredService<SqlServerFullTextSearch>();
var advancedSearch = provider.GetRequiredService<SqlServerAdvancedSearch>();
var query = new QueryEmbedding
{
    Vector = EmbeddingVector.FromFloat32([0.99f, 0.01f, 0f, 0f]),
    Identity = Identity(fingerprint),
    SourceTokenCount = 2,
    InputTokenCount = 2
};
var semanticMapping = new SqlServerCandidateQuery
{
    Table = table,
    ItemKeyColumn = "item_id",
    FieldNameColumn = "field_name",
    FingerprintColumn = "fingerprint",
    VectorColumn = "embedding",
    RecordJsonColumn = "record_json",
    FieldWeightColumn = "field_weight",
    VectorDimensions = 4,
    SearchMode = SqlServerVectorSearchMode.Exact,
    FilterColumns = new Dictionary<string, string> { ["TenantId"] = "tenant_id" },
    Filter = SearchFilter.Equal("TenantId", 1)
};
var result = await search.SearchAsync<string>(
    connection,
    query,
    semanticMapping,
    new DatabaseSemanticSearchOptions { Top = 1, CandidateCount = 10 });
if (result.Results.Count != 1 || result.Results[0].Item != "backup")
    throw new InvalidOperationException("SQL Server filtered VECTOR_DISTANCE retrieval returned the wrong item.");

var lexicalMapping = new SqlServerLexicalQuery
{
    Table = lexicalTable,
    ItemKeyColumn = "item_id",
    FullTextKeyColumn = "id",
    Fields = [new SqlServerFullTextField("title", "title"), new SqlServerFullTextField("body", "body")],
    FilterColumns = new Dictionary<string, string> { ["TenantId"] = "tenant_id" },
    Filter = SearchFilter.Equal("TenantId", 1)
};
var lexical = await lexicalSearch.SearchAsync<string>(
    connection,
    "postgresql backup",
    lexicalMapping,
    [SearchFieldWeight.Create("title", 8), SearchFieldWeight.Create("body", 1)],
    new DatabaseLexicalSearchOptions { Top = 2 });
if (lexical.Results.Count == 0 || lexical.Results[0].Item != "backup")
    throw new InvalidOperationException("SQL Server Full-Text weighted-field retrieval did not prefer the title match.");

var hybridQuery = SearchQuery.Create("postgresql backup")
    .Where(SearchFilter.Equal("TenantId", 1))
    .Add(SearchRetrievalStage.Semantic(SearchFieldWeight.Create("content", 1)).Candidates(10))
    .Add(SearchRetrievalStage.Lexical(SearchFieldWeight.Create("title", 8), SearchFieldWeight.Create("body", 1)).Candidates(10))
    .Take(2);
var hybrid = await advancedSearch.SearchAsync<string>(
    connection,
    hybridQuery,
    new SqlServerSearchPlan
    {
        Semantic = semanticMapping with { Filter = null },
        Lexical = lexicalMapping with { Filter = null }
    },
    semanticQuery: query);
if (hybrid.Count == 0 || hybrid[0].Item != "backup" || hybrid[0].Contributions.Count != 2)
    throw new InvalidOperationException("SQL Server semantic + lexical RRF did not return the jointly supported backup item.");

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
if (databaseEmbedding.Vector.Dimensions != 1998 || databaseQuery.Vector.Dimensions != 1998 ||
    databaseEmbedding.Identity.EmbeddingSpaceFingerprint != databaseQuery.Identity.EmbeddingSpaceFingerprint)
    throw new InvalidOperationException("SQL Server 2048 -> 1998 deterministic vector-space preparation failed.");

Console.WriteLine($"PASS SQL Server 2025 vector + Full-Text Search + portable filtering + hybrid RRF. Approximate available: {capabilities.SupportsApproximateSearch}.");

async Task InsertAsync(string itemId, int tenantId, TextEmbedding embedding)
{
    await using var insert = new SqlCommand($"""
        INSERT INTO {table}(item_id, tenant_id, field_name, fingerprint, embedding, record_json, field_weight)
        VALUES (@item, @tenant, @field, @fingerprint, @embedding, @json, @weight)
        """, connection);
    insert.Parameters.AddWithValue("@item", itemId);
    insert.Parameters.AddWithValue("@tenant", tenantId);
    insert.Parameters.AddWithValue("@field", "content");
    insert.Parameters.AddWithValue("@fingerprint", embedding.Identity.EmbeddingSpaceFingerprint);
    insert.Parameters.Add(new SqlParameter("@embedding", SqlDbTypeExtensions.Vector) { Value = new SqlVector<float>(embedding.Vector.ToFloat32()) });
    insert.Parameters.AddWithValue("@json", EmbeddingSerializer.SerializeJson(embedding));
    insert.Parameters.AddWithValue("@weight", 1f);
    await insert.ExecuteNonQueryAsync();
}

async Task InsertLexicalAsync(int id, string itemId, int tenantId, string title, string body)
{
    await using var insert = new SqlCommand("INSERT INTO dbo.onnx_lexical_items(id, item_id, tenant_id, title, body) VALUES (@id, @item, @tenant, @title, @body)", connection);
    insert.Parameters.AddWithValue("@id", id);
    insert.Parameters.AddWithValue("@item", itemId);
    insert.Parameters.AddWithValue("@tenant", tenantId);
    insert.Parameters.AddWithValue("@title", title);
    insert.Parameters.AddWithValue("@body", body);
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
