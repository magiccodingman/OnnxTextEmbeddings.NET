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

await InsertAsync(1, first);
await InsertAsync(2, second);

var query = EmbeddingVector.FromFloat32(new[] { 0.99f, 0.01f, 0f, 0f }, EmbeddingVectorFormat.Float32);
await using (var command = new NpgsqlCommand(
    "SELECT id, portable FROM onnx_embedding_integration ORDER BY embedding <=> $1 LIMIT 1", connection))
{
    command.Parameters.AddWithValue(query.ToPgVector());
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

Console.WriteLine("PostgreSQL pgvector + portable BYTEA integration passed.");

async Task InsertAsync(int id, EmbeddingVector vector)
{
    await using var command = new NpgsqlCommand(
        "INSERT INTO onnx_embedding_integration (id, embedding, portable) VALUES ($1, $2, $3)", connection);
    command.Parameters.AddWithValue(id);
    command.Parameters.AddWithValue(vector.ToPgVector());
    command.Parameters.AddWithValue(EmbeddingSerializer.SerializeVector(vector));
    await command.ExecuteNonQueryAsync();
}
