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
    new Page("Backups", "# Backups\n\nRestore PostgreSQL from S3 by downloading the database backup and applying it to the target database."),
    new Page("Networking", "# Networking\n\nWireGuard peers create encrypted network tunnels and exchange protected traffic over UDP."),
    new Page("Cooking", "# Cooking\n\nRoast potatoes in a hot oven until the outside is golden and crisp."),
    new Page("Certificates", "# TLS Certificates\n\nRenew the HTTPS certificate before expiration and reload the web server so TLS clients receive the new certificate."),
    new Page("Containers", "# Containers\n\nKubernetes schedules application containers into pods and keeps the requested replicas running across cluster nodes.")
};

var indexed = new List<IndexedPage>();
foreach (var page in pages)
{
    var vectors = await embeddings.EmbedDocumentAsync(page.Content);
    if (vectors.Count == 0)
        throw new InvalidOperationException($"No embedding was returned for {page.Title}.");
    indexed.Add(new IndexedPage(page, vectors));
}

var cases = new[]
{
    new SearchCase("How do I restore my PostgreSQL database backup?", "Backups"),
    new SearchCase("What creates an encrypted UDP network tunnel?", "Networking"),
    new SearchCase("How can I make roasted potatoes crispy?", "Cooking"),
    new SearchCase("What should I renew when my HTTPS TLS certificate is expiring?", "Certificates"),
    new SearchCase("What system schedules containers into pods across cluster nodes?", "Containers")
};

foreach (var testCase in cases)
{
    var query = await embeddings.EmbedQueryAsync(testCase.Query);
    var results = await search.SearchAsync(
        query,
        indexed,
        x => x.Embeddings,
        new SemanticSearchRequest { Top = 3 });

    if (results.Count == 0 || results[0].Item.Page.Title != testCase.ExpectedTitle)
    {
        var actual = results.Count == 0 ? "<no result>" : results[0].Item.Page.Title;
        throw new InvalidOperationException(
            $"Jasper {precision} semantic-search smoke test expected '{testCase.ExpectedTitle}' first for '{testCase.Query}', but got '{actual}'.");
    }

    Console.WriteLine($"PASS {testCase.ExpectedTitle}: {results[0].Score:P1} - {testCase.Query}");
}

await host.StopAsync();

sealed record Page(string Title, string Content);
sealed record IndexedPage(Page Page, IReadOnlyList<TextEmbedding> Embeddings);
sealed record SearchCase(string Query, string ExpectedTitle);
