using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Npgsql;
using OnnxTextEmbeddings;
using OnnxTextEmbeddings.PgVector;
using Pgvector;

var connectionString = Environment.GetEnvironmentVariable("POSTGRES_CONNECTION_STRING")
    ?? "Host=localhost;Username=postgres;Password=postgres;Database=embeddings";

var hostBuilder = Host.CreateApplicationBuilder(args);
hostBuilder.Services.AddOnnxTextEmbeddings();
using var host = hostBuilder.Build();
await host.StartAsync();
var embeddingService = host.Services.GetRequiredService<ITextEmbeddingService>();

var dataSourceBuilder = new NpgsqlDataSourceBuilder(connectionString);
dataSourceBuilder.UseVector();
await using var dataSource = dataSourceBuilder.Build();
await using var connection = await dataSource.OpenConnectionAsync();
await using (var command = new NpgsqlCommand("CREATE EXTENSION IF NOT EXISTS vector", connection))
    await command.ExecuteNonQueryAsync();
connection.ReloadTypes();

await using (var command = new NpgsqlCommand("""
    CREATE TABLE IF NOT EXISTS sample_embeddings (
        document_id text NOT NULL,
        chunk_index integer NOT NULL,
        fingerprint text NOT NULL,
        embedding vector(2048) NOT NULL,
        record_json text NOT NULL,
        PRIMARY KEY(document_id, chunk_index)
    )
    """, connection))
    await command.ExecuteNonQueryAsync();

var chunks = await embeddingService.EmbedDocumentAsync("# Backups\n\nRestore PostgreSQL from a stored backup.");
foreach (var chunk in chunks)
{
    await using var command = new NpgsqlCommand("""
        INSERT INTO sample_embeddings(document_id, chunk_index, fingerprint, embedding, record_json)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (document_id, chunk_index) DO UPDATE SET
            fingerprint = EXCLUDED.fingerprint,
            embedding = EXCLUDED.embedding,
            record_json = EXCLUDED.record_json
        """, connection);
    command.Parameters.AddWithValue("backups");
    command.Parameters.AddWithValue(chunk.Chunk.Index);
    command.Parameters.AddWithValue(chunk.Identity.EmbeddingSpaceFingerprint);
    command.Parameters.AddWithValue(chunk.Vector.ToPgVector());
    command.Parameters.AddWithValue(EmbeddingSerializer.SerializeJson(chunk));
    await command.ExecuteNonQueryAsync();
}

var query = await embeddingService.EmbedQueryAsync("restore database backup");
await using (var command = new NpgsqlCommand("""
    SELECT document_id, chunk_index, record_json
    FROM sample_embeddings
    WHERE fingerprint = $1
    ORDER BY embedding <=> $2
    LIMIT 20
    """, connection))
{
    command.Parameters.AddWithValue(query.Identity.EmbeddingSpaceFingerprint);
    command.Parameters.AddWithValue(query.Vector.ToPgVector());
    await using var reader = await command.ExecuteReaderAsync();
    while (await reader.ReadAsync())
        Console.WriteLine($"{reader.GetString(0)} chunk {reader.GetInt32(1)}");
}

await host.StopAsync();
