using Microsoft.Data.Sqlite;
using Microsoft.Extensions.DependencyInjection;
using OnnxTextEmbeddings;
using OnnxTextEmbeddings.SqliteVec;

await using var connection = new SqliteConnection("Data Source=:memory:");
connection.LoadOnnxTextEmbeddingsSqliteVec();
await connection.OpenAsync();
var capabilities = await connection.GetSqliteVecCapabilitiesAsync();
if (string.IsNullOrWhiteSpace(capabilities.Version))
    throw new InvalidOperationException("sqlite-vec did not report a version.");

var services = new ServiceCollection();
services.AddLogging();
services.AddOnnxTextEmbeddings(options => options.Initialization.WarmupOnStartup = false);
services.AddOnnxTextEmbeddingsSqliteVec();
await using var provider = services.BuildServiceProvider();
var search = provider.GetRequiredService<SqliteVecSemanticSearch>();
var lexicalSearch = provider.GetRequiredService<SqliteFts5LexicalSearch>();
var advancedSearch = provider.GetRequiredService<SqliteVecAdvancedSearch>();

const string lexicalTable = "onnx_lexical_items";
await using (var create = connection.CreateCommand())
{
    create.CommandText = $"CREATE VIRTUAL TABLE {lexicalTable} USING fts5(item_id UNINDEXED, tenant_id UNINDEXED, title, body)";
    await create.ExecuteNonQueryAsync();
}
await InsertLexicalAsync("backup", 1, "PostgreSQL Backup", "Restore and disaster recovery procedures.");
await InsertLexicalAsync("network", 1, "Networking", "Firewall and routing documentation.");
await InsertLexicalAsync("body-heavy", 1, "Database Operations", "PostgreSQL backup PostgreSQL backup PostgreSQL backup procedures.");
await InsertLexicalAsync("other-tenant", 2, "PostgreSQL Backup", "Exact title in another tenant.");

var lexicalMapping = new SqliteFts5Query
{
    Table = lexicalTable,
    ItemKeyColumn = "item_id",
    ColumnOrder = ["item_id", "tenant_id", "title", "body"],
    Fields = [new SqliteFts5Field("title", "title"), new SqliteFts5Field("body", "body")],
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
    throw new InvalidOperationException("SQLite FTS5 BM25 field weighting did not prefer the title match.");

await VerifyStorageAsync("float[4] distance_metric=cosine", SqliteVecStorageKind.Float32, "f32");
await VerifyStorageAsync("int8[4] distance_metric=cosine", SqliteVecStorageKind.Int8, "i8");
Console.WriteLine($"PASS SQLite/sqlite-vec {capabilities.Version} FP32 + INT8 + FTS5 BM25 + filtering + exact-filter fallback + hybrid RRF integration.");

async Task VerifyStorageAsync(string vectorDefinition, SqliteVecStorageKind storageKind, string suffix)
{
    var table = $"semantic_candidates_{suffix}";
    await using (var create = connection.CreateCommand())
    {
        create.CommandText = $"""
            CREATE VIRTUAL TABLE {table} USING vec0(
                item_id text,
                field_name text,
                fingerprint text,
                embedding {vectorDefinition},
                +record_json text,
                +field_weight float,
                tenant_id integer
            )
            """;
        await create.ExecuteNonQueryAsync();
    }

    const string fingerprint = "sqlite-vec-integration-space";
    await InsertAsync(table, "backup", 1, TextEmbeddingFor([1f, 0f, 0f, 0f], fingerprint, 0), storageKind);
    await InsertAsync(table, "network", 1, TextEmbeddingFor([0f, 1f, 0f, 0f], fingerprint, 0), storageKind);
    await InsertAsync(table, "other-tenant", 2, TextEmbeddingFor([1f, 0f, 0f, 0f], fingerprint, 0), storageKind);

    var query = new QueryEmbedding
    {
        Vector = EmbeddingVector.FromFloat32([0.99f, 0.01f, 0f, 0f]),
        Identity = Identity(fingerprint),
        SourceTokenCount = 2,
        InputTokenCount = 2
    };

    var semanticMapping = new SqliteVecCandidateQuery
    {
        Table = table,
        ItemKeyColumn = "item_id",
        FieldNameColumn = "field_name",
        FingerprintColumn = "fingerprint",
        VectorColumn = "embedding",
        RecordJsonColumn = "record_json",
        FieldWeightColumn = "field_weight",
        StorageKind = storageKind,
        FilterColumns = new Dictionary<string, string> { ["TenantId"] = "tenant_id" },
        Filter = SearchFilter.Equal("TenantId", 1)
    };
    var result = await search.SearchAsync<string>(
        connection,
        query,
        semanticMapping,
        new DatabaseSemanticSearchOptions { Top = 1, CandidateCount = 10 });
    if (result.Results.Count != 1 || result.Results[0].Item != "backup")
        throw new InvalidOperationException($"sqlite-vec {storageKind} filtered KNN candidate retrieval returned the wrong item.");

    if (storageKind == SqliteVecStorageKind.Float32)
    {
        var exactFallback = await search.SearchAsync<string>(
            connection,
            query,
            semanticMapping with
            {
                Filter = SearchFilter.Or(
                    SearchFilter.Equal("TenantId", 1),
                    SearchFilter.Equal("TenantId", 999))
            },
            new DatabaseSemanticSearchOptions { Top = 1, CandidateCount = 10 });
        if (exactFallback.Results.Count != 1 || exactFallback.Results[0].Item != "backup" ||
            !exactFallback.Retrieval.Mode.Contains("FilteredExactScan", StringComparison.Ordinal))
            throw new InvalidOperationException("SQLite rich-filter exact-scan fallback did not preserve semantic filtering.");

        var hybridQuery = SearchQuery.Create("postgresql backup")
            .Where(SearchFilter.Equal("TenantId", 1))
            .Add(SearchRetrievalStage.Semantic(SearchFieldWeight.Create("content", 1)).Candidates(10))
            .Add(SearchRetrievalStage.Lexical(SearchFieldWeight.Create("title", 8), SearchFieldWeight.Create("body", 1)).Candidates(10))
            .Take(2);
        var hybrid = await advancedSearch.SearchAsync<string>(
            connection,
            hybridQuery,
            new SqliteVecSearchPlan
            {
                Semantic = semanticMapping with { Filter = null },
                Lexical = lexicalMapping with { Filter = null }
            },
            semanticQuery: query);
        if (hybrid.Count == 0 || hybrid[0].Item != "backup" || hybrid[0].Contributions.Count != 2)
            throw new InvalidOperationException("SQLite semantic + lexical RRF did not return the jointly supported backup item.");
    }
}

async Task InsertAsync(string table, string itemId, int tenantId, TextEmbedding embedding, SqliteVecStorageKind storageKind)
{
    await using var insert = connection.CreateCommand();
    var constructor = storageKind == SqliteVecStorageKind.Int8 ? "vec_int8" : "vec_f32";
    insert.CommandText = $"""
        INSERT INTO {table}(item_id, field_name, fingerprint, embedding, record_json, field_weight, tenant_id)
        VALUES ($item, $field, $fingerprint, {constructor}($embedding), $json, $weight, $tenant)
        """;
    insert.Parameters.AddWithValue("$item", itemId);
    insert.Parameters.AddWithValue("$field", "content");
    insert.Parameters.AddWithValue("$fingerprint", embedding.Identity.EmbeddingSpaceFingerprint);
    var vector = storageKind == SqliteVecStorageKind.Int8
        ? embedding.Vector.ConvertTo(EmbeddingVectorFormat.Int8)
        : embedding.Vector.ConvertTo(EmbeddingVectorFormat.Float32);
    insert.Parameters.Add("$embedding", SqliteType.Blob).Value = vector.Data;
    insert.Parameters.AddWithValue("$json", EmbeddingSerializer.SerializeJson(embedding));
    insert.Parameters.AddWithValue("$weight", 1.0);
    insert.Parameters.AddWithValue("$tenant", tenantId);
    await insert.ExecuteNonQueryAsync();
}

async Task InsertLexicalAsync(string itemId, int tenantId, string title, string body)
{
    await using var insert = connection.CreateCommand();
    insert.CommandText = $"INSERT INTO {lexicalTable}(item_id, tenant_id, title, body) VALUES ($item, $tenant, $title, $body)";
    insert.Parameters.AddWithValue("$item", itemId);
    insert.Parameters.AddWithValue("$tenant", tenantId);
    insert.Parameters.AddWithValue("$title", title);
    insert.Parameters.AddWithValue("$body", body);
    await insert.ExecuteNonQueryAsync();
}

static TextEmbedding TextEmbeddingFor(float[] values, string fingerprint, int chunk) => new()
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
        Index = chunk,
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
