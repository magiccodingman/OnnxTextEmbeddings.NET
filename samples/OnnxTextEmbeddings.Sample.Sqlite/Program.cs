using Microsoft.Data.Sqlite;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using OnnxTextEmbeddings;

var builder = Host.CreateApplicationBuilder(args);
builder.Services.AddOnnxTextEmbeddings(options =>
    options.Vectors.DocumentFormat = EmbeddingVectorFormat.Int4);
using var host = builder.Build();
await host.StartAsync();

var embeddingService = host.Services.GetRequiredService<ITextEmbeddingService>();
var semanticSearch = host.Services.GetRequiredService<ISemanticSearch>();

await using var connection = new SqliteConnection("Data Source=:memory:");
await connection.OpenAsync();
await using (var command = connection.CreateCommand())
{
    command.CommandText = """
        CREATE TABLE embeddings (
            document_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            record_json TEXT NOT NULL,
            PRIMARY KEY (document_id, chunk_index)
        );
        """;
    await command.ExecuteNonQueryAsync();
}

var documents = new Dictionary<string, string>
{
    ["backups"] = "# Backups\n\nRestore PostgreSQL from a stored backup.",
    ["networking"] = "# Networking\n\nWireGuard provides encrypted tunnels."
};

foreach (var (id, text) in documents)
{
    var chunks = await embeddingService.EmbedDocumentAsync(text);
    foreach (var chunk in chunks)
    {
        await using var insert = connection.CreateCommand();
        insert.CommandText = "INSERT INTO embeddings(document_id, chunk_index, record_json) VALUES ($id, $chunk, $json)";
        insert.Parameters.AddWithValue("$id", id);
        insert.Parameters.AddWithValue("$chunk", chunk.Chunk.Index);
        insert.Parameters.AddWithValue("$json", EmbeddingSerializer.SerializeJson(chunk));
        await insert.ExecuteNonQueryAsync();
    }
}

var indexed = new Dictionary<string, List<TextEmbedding>>(StringComparer.Ordinal);
await using (var command = connection.CreateCommand())
{
    command.CommandText = "SELECT document_id, record_json FROM embeddings ORDER BY document_id, chunk_index";
    await using var reader = await command.ExecuteReaderAsync();
    while (await reader.ReadAsync())
    {
        var id = reader.GetString(0);
        if (!indexed.TryGetValue(id, out var list))
            indexed[id] = list = [];
        list.Add(EmbeddingSerializer.DeserializeJson(reader.GetString(1)));
    }
}

var query = await embeddingService.EmbedQueryAsync("restore database backup");
var results = await semanticSearch.SearchAsync(query, indexed, pair => pair.Value);
foreach (var result in results)
    Console.WriteLine($"{result.Score:P1} {result.Item.Key}");

await host.StopAsync();
