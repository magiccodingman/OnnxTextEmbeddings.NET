using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using OnnxTextEmbeddings;

var precisionName = Environment.GetEnvironmentVariable("JASPER_PRECISION") ?? "Int8";
if (!Enum.TryParse<JasperModelPrecision>(precisionName, ignoreCase: true, out var precision))
    throw new ArgumentException($"Unknown JASPER_PRECISION '{precisionName}'. Use Int8, Int4, or Float32.");

var builder = Host.CreateApplicationBuilder(args);
builder.Services.AddOnnxTextEmbeddings(options => options.Model.UseJasper(precision));

using var host = builder.Build();
await host.StartAsync();

var embeddings = host.Services.GetRequiredService<ITextEmbeddingService>();
var search = host.Services.GetRequiredService<ISemanticSearch>();
await embeddings.WaitUntilReadyAsync();

Console.WriteLine($"Model: {embeddings.ModelInfo?.ModelId}");
Console.WriteLine($"Revision: {embeddings.ModelInfo?.SourceRevision}");
Console.WriteLine($"Dimensions: {embeddings.ModelInfo?.Dimensions}");

if (embeddings.ModelInfo?.Dimensions != 2048)
    throw new InvalidOperationException($"Expected Jasper to produce 2048 dimensions, got {embeddings.ModelInfo?.Dimensions}.");

var pages = new[]
{
    new Page("Backups", "# Backups\n\nRestore PostgreSQL from S3 by downloading the backup and applying it to the target database."),
    new Page("Networking", "# Networking\n\nWireGuard peers exchange encrypted traffic over UDP."),
    new Page("Cooking", "# Cooking\n\nRoast potatoes until crisp.")
};

var indexed = new List<IndexedPage>();
foreach (var page in pages)
{
    var vectors = await embeddings.EmbedDocumentAsync(page.Content);
    if (vectors.Count == 0)
        throw new InvalidOperationException($"No embedding was returned for {page.Title}.");
    indexed.Add(new IndexedPage(page, vectors));
}

var query = await embeddings.EmbedQueryAsync("How do I restore my PostgreSQL backup?");
var results = await search.SearchAsync(
    query,
    indexed,
    x => x.Embeddings,
    new SemanticSearchRequest { Top = 3 });

if (results.Count != 3 || results[0].Item.Page.Title != "Backups")
    throw new InvalidOperationException("Jasper semantic-search smoke test did not rank the backup page first.");

foreach (var result in results)
    Console.WriteLine($"{result.Score:P1} - {result.Item.Page.Title}: {result.BestMatch.Embedding.Text}");

await host.StopAsync();

sealed record Page(string Title, string Content);
sealed record IndexedPage(Page Page, IReadOnlyList<TextEmbedding> Embeddings);
