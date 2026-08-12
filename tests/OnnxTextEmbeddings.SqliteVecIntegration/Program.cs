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
await using var provider = services.BuildServiceProvider();
var reranker = provider.GetRequiredService<ISemanticCandidateReranker>();
var search = new SqliteVecSemanticSearch(reranker);

await VerifyStorageAsync("float[4] distance_metric=cosine", SqliteVecStorageKind.Float32, "f32");
await VerifyStorageAsync("int8[4] distance_metric=cosine", SqliteVecStorageKind.Int8, "i8");
Console.WriteLine($"PASS SQLite/sqlite-vec {capabilities.Version} FP32 + INT8 native candidate reranking integration.");

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
                +field_weight float
            )
            """;
        await create.ExecuteNonQueryAsync();
    }

    const string fingerprint = "sqlite-vec-integration-space";
    await InsertAsync(table, "backup", TextEmbeddingFor([1f, 0f, 0f, 0f], fingerprint, 0), storageKind);
    await InsertAsync(table, "network", TextEmbeddingFor([0f, 1f, 0f, 0f], fingerprint, 0), storageKind);

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
        new SqliteVecCandidateQuery
        {
            Table = table,
            ItemKeyColumn = "item_id",
            FieldNameColumn = "field_name",
            FingerprintColumn = "fingerprint",
            VectorColumn = "embedding",
            RecordJsonColumn = "record_json",
            FieldWeightColumn = "field_weight",
            StorageKind = storageKind
        },
        new DatabaseSemanticSearchOptions { Top = 1, CandidateCount = 10 });

    if (result.Results.Count != 1 || result.Results[0].Item != "backup")
        throw new InvalidOperationException($"sqlite-vec {storageKind} candidate retrieval returned the wrong item.");
    if (result.Retrieval.Provider != "SQLite/sqlite-vec" || result.Retrieval.Approximate)
        throw new InvalidOperationException("Unexpected sqlite-vec retrieval diagnostics.");
}

async Task InsertAsync(string table, string itemId, TextEmbedding embedding, SqliteVecStorageKind storageKind)
{
    await using var insert = connection.CreateCommand();
    var constructor = storageKind == SqliteVecStorageKind.Int8 ? "vec_int8" : "vec_f32";
    insert.CommandText = $"""
        INSERT INTO {table}(item_id, field_name, fingerprint, embedding, record_json, field_weight)
        VALUES ($item, $field, $fingerprint, {constructor}($embedding), $json, $weight)
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
