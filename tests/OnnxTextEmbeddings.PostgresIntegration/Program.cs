using Microsoft.Extensions.DependencyInjection;
using Npgsql;
using OnnxTextEmbeddings;
using OnnxTextEmbeddings.PgVector;
using Pgvector;

var connectionString = Environment.GetEnvironmentVariable("POSTGRES_CONNECTION_STRING")
    ?? "Host=localhost;Port=5432;Username=postgres;Password=postgres;Database=embeddings";

var dataSourceBuilder = new NpgsqlDataSourceBuilder(connectionString);
dataSourceBuilder.UseVector();
await using var dataSource = dataSourceBuilder.Build();
await using var connection = await dataSource.OpenConnectionAsync();

await using (var command = new NpgsqlCommand("CREATE EXTENSION IF NOT EXISTS vector", connection))
    await command.ExecuteNonQueryAsync();
connection.ReloadTypes();

await using (var command = new NpgsqlCommand("DROP TABLE IF EXISTS onnx_embedding_integration", connection))
    await command.ExecuteNonQueryAsync();
await using (var command = new NpgsqlCommand(
    "CREATE TABLE onnx_embedding_integration (id integer PRIMARY KEY, embedding vector(4) NOT NULL, portable bytea NOT NULL)", connection))
    await command.ExecuteNonQueryAsync();

var first = EmbeddingVector.FromFloat32(new[] { 1f, 0f, 0f, 0f }, EmbeddingVectorFormat.Int4);
var second = EmbeddingVector.FromFloat32(new[] { 0f, 1f, 0f, 0f }, EmbeddingVectorFormat.Int8);

await InsertPortableAsync(1, first);
await InsertPortableAsync(2, second);

var queryVector = EmbeddingVector.FromFloat32(new[] { 0.99f, 0.01f, 0f, 0f }, EmbeddingVectorFormat.Float32);
await using (var command = new NpgsqlCommand(
    "SELECT id, portable FROM onnx_embedding_integration ORDER BY embedding <=> $1 LIMIT 1", connection))
{
    command.Parameters.AddWithValue(queryVector.ToPgVector());
    await using var reader = await command.ExecuteReaderAsync();
    if (!await reader.ReadAsync())
        throw new InvalidOperationException("pgvector integration query returned no rows.");
    if (reader.GetInt32(0) != 1)
        throw new InvalidOperationException("pgvector cosine ordering did not return the expected nearest vector.");

    var restored = EmbeddingSerializer.DeserializeVector((byte[])reader[1]);
    if (restored.Format != EmbeddingVectorFormat.Int4 || restored.Dimensions != 4)
        throw new InvalidOperationException("Portable BYTEA vector did not round-trip with its encoding metadata.");
    if (EmbeddingVectorMath.CosineSimilarity(first, restored) < 0.999f)
        throw new InvalidOperationException("Portable BYTEA vector changed during PostgreSQL round-trip.");
}

const string candidateTable = "onnx_semantic_candidates";
await using (var command = new NpgsqlCommand($"DROP TABLE IF EXISTS {candidateTable}", connection))
    await command.ExecuteNonQueryAsync();
await using (var command = new NpgsqlCommand($"""
    CREATE TABLE {candidateTable} (
        item_id text NOT NULL,
        field_name text NOT NULL,
        fingerprint text NOT NULL,
        embedding vector(4) NOT NULL,
        record_json text NOT NULL,
        field_weight real NOT NULL
    )
    """, connection))
    await command.ExecuteNonQueryAsync();

const string fingerprint = "postgres-integration-space";
var backup = TextEmbeddingFor([1f, 0f, 0f, 0f], fingerprint, 0);
var backupSupport = TextEmbeddingFor([0.94f, 0.30f, 0f, 0f], fingerprint, 1);
var network = TextEmbeddingFor([0f, 1f, 0f, 0f], fingerprint, 0);
await InsertCandidateAsync("backup", "content", backup, 1f);
await InsertCandidateAsync("backup", "content", backupSupport, 1f);
await InsertCandidateAsync("network", "content", network, 1f);

var services = new ServiceCollection();
services.AddLogging();
services.AddOnnxTextEmbeddings(options => options.Initialization.WarmupOnStartup = false);
await using var provider = services.BuildServiceProvider();
var reranker = provider.GetRequiredService<ISemanticCandidateReranker>();
var databaseSearch = new PgVectorSemanticSearch(reranker);
var query = new QueryEmbedding
{
    Vector = queryVector,
    Identity = Identity(fingerprint),
    SourceTokenCount = 3,
    InputTokenCount = 3
};
var searchResult = await databaseSearch.SearchAsync<string>(
    connection,
    query,
    new PgVectorCandidateQuery
    {
        Table = candidateTable,
        ItemKeyColumn = "item_id",
        FieldNameColumn = "field_name",
        FingerprintColumn = "fingerprint",
        VectorColumn = "embedding",
        RecordJsonColumn = "record_json",
        FieldWeightColumn = "field_weight",
        SearchMode = PgVectorSearchMode.Exact
    },
    new DatabaseSemanticSearchOptions { Top = 1, CandidateCount = 10 });

if (searchResult.Results.Count != 1 || searchResult.Results[0].Item != "backup")
    throw new InvalidOperationException("pgvector candidate retrieval + DefaultV1 reranking did not return the expected item.");
if (searchResult.Retrieval.Provider != "PostgreSQL/pgvector" || searchResult.Retrieval.Approximate)
    throw new InvalidOperationException("Unexpected pgvector retrieval diagnostics.");

Console.WriteLine("PASS PostgreSQL pgvector + portable BYTEA + database-native candidate reranking integration.");

async Task InsertPortableAsync(int id, EmbeddingVector vector)
{
    await using var command = new NpgsqlCommand(
        "INSERT INTO onnx_embedding_integration (id, embedding, portable) VALUES ($1, $2, $3)", connection);
    command.Parameters.AddWithValue(id);
    command.Parameters.AddWithValue(vector.ToPgVector());
    command.Parameters.AddWithValue(EmbeddingSerializer.SerializeVector(vector));
    await command.ExecuteNonQueryAsync();
}

async Task InsertCandidateAsync(string itemId, string field, TextEmbedding embedding, float weight)
{
    await using var command = new NpgsqlCommand($"""
        INSERT INTO {candidateTable}(item_id, field_name, fingerprint, embedding, record_json, field_weight)
        VALUES ($1, $2, $3, $4, $5, $6)
        """, connection);
    command.Parameters.AddWithValue(itemId);
    command.Parameters.AddWithValue(field);
    command.Parameters.AddWithValue(embedding.Identity.EmbeddingSpaceFingerprint);
    command.Parameters.AddWithValue(embedding.Vector.ToPgVector());
    command.Parameters.AddWithValue(EmbeddingSerializer.SerializeJson(embedding));
    command.Parameters.AddWithValue(weight);
    await command.ExecuteNonQueryAsync();
}

static TextEmbedding TextEmbeddingFor(float[] values, string fingerprint, int chunk) => new()
{
    Vector = EmbeddingVector.FromFloat32(values),
    Identity = Identity(fingerprint),
    Source = new EmbeddingSource
    {
        DocumentTokenCount = 100,
        CharacterRange = new Utf16TextRange(chunk * 10, 10),
        TokenRange = new TokenRange(chunk * 10, 10),
        TokenCount = 10,
        TokenCapacity = 10
    },
    Chunk = new EmbeddingChunkInfo
    {
        Index = chunk,
        Count = 2,
        BoundaryKind = ChunkBoundaryKind.Paragraph,
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
