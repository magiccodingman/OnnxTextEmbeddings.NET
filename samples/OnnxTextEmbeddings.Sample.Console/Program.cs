using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using OnnxTextEmbeddings;

var builder = Host.CreateApplicationBuilder(args);
builder.Services.AddOnnxTextEmbeddings();

using var host = builder.Build();
await host.StartAsync();

var embeddings = host.Services.GetRequiredService<ITextEmbeddingService>();
var search = host.Services.GetRequiredService<ISemanticSearch>();

var pages = new[]
{
    new Page("Backups", "# Backups\n\nRestore PostgreSQL from S3 by downloading the backup and applying it to the target database."),
    new Page("Networking", "# Networking\n\nWireGuard peers exchange encrypted traffic over UDP."),
    new Page("Cooking", "# Cooking\n\nRoast potatoes until crisp.")
};

var indexed = new List<IndexedPage>();
foreach (var page in pages)
    indexed.Add(new IndexedPage(page, await embeddings.EmbedAsync(page.Content)));

var results = await search.SearchAsync(
    "How do I restore my PostgreSQL backup?",
    indexed,
    x => x.Embeddings,
    new SemanticSearchRequest { Top = 3 });

foreach (var result in results)
    Console.WriteLine($"{result.Score:P1} - {result.Item.Page.Title}: {result.BestMatch.Embedding.Text}");

await host.StopAsync();

sealed record Page(string Title, string Content);
sealed record IndexedPage(Page Page, IReadOnlyList<TextEmbedding> Embeddings);
