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
Console.WriteLine($"Instances: {embeddings.ModelInfo?.ModelInstanceCount}; Threads/model: {embeddings.ModelInfo?.ThreadsPerModel}; Concurrent/model: {embeddings.ModelInfo?.ConcurrentRequestsPerModel}");

if (embeddings.ModelInfo?.Dimensions != 2048)
    throw new InvalidOperationException($"Expected Jasper to produce 2048 dimensions, got {embeddings.ModelInfo?.Dimensions}.");
if (embeddings.ModelInfo is not { ModelInstanceCount: 1, ThreadsPerModel: 16, ConcurrentRequestsPerModel: 8 })
    throw new InvalidOperationException("Default inference topology should be one model instance, 16 threads, and 8 concurrent requests per model.");

const string formatProbe = "A compact semantic embedding format probe.";
var defaultDocument = await embeddings.EmbedDocumentAsync(formatProbe);
if (defaultDocument.Count != 1 || defaultDocument[0].Vector.Format != EmbeddingVectorFormat.Float32)
    throw new InvalidOperationException("Default document embeddings should return Float32.");

foreach (var format in new[]
         {
             EmbeddingVectorFormat.Int4,
             EmbeddingVectorFormat.Int8,
             EmbeddingVectorFormat.Float16,
             EmbeddingVectorFormat.Float32
         })
{
    var document = await embeddings.EmbedDocumentAsync(formatProbe, format);
    if (document.Count != 1 || document[0].Vector.Format != format)
        throw new InvalidOperationException($"Per-call document format override failed for {format}.");

    var query = await embeddings.EmbedQueryAsync("semantic format probe", format);
    if (query.Vector.Format != format)
        throw new InvalidOperationException($"Per-call query format override failed for {format}.");
}
Console.WriteLine("PASS Float32 defaults and per-call INT4/INT8/FP16/FP32 return formats.");

const string normalQuery = "How do I restore my PostgreSQL database backup?";
var sourceTokenCount = await embeddings.CountTokensAsync(normalQuery);
var queryTokenCount = await embeddings.CountQueryTokensAsync(normalQuery);
if (sourceTokenCount <= 0 || queryTokenCount.SourceTokenCount != sourceTokenCount || !queryTokenCount.Fits)
    throw new InvalidOperationException("Token-count API returned unexpected values for a normal query.");

var oversizedQuery = string.Join(' ', Enumerable.Repeat("database backup restoration procedure", 400));
var oversizedCount = await embeddings.CountQueryTokensAsync(oversizedQuery);
if (oversizedCount.Fits || oversizedCount.InputTokenCount <= oversizedCount.QueryMaxTokens)
    throw new InvalidOperationException("Oversized query token counting should report Fits=false without throwing.");
try
{
    _ = await embeddings.EmbedQueryAsync(oversizedQuery);
    throw new InvalidOperationException("EmbedQueryAsync should reject an oversized query.");
}
catch (QueryTokenLimitExceededException)
{
    Console.WriteLine($"PASS oversized query count: {oversizedCount.InputTokenCount} > {oversizedCount.QueryMaxTokens}");
}

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
    new SearchCase(normalQuery, "Backups"),
    new SearchCase("What creates an encrypted UDP network tunnel?", "Networking"),
    new SearchCase("How can I make roasted potatoes crispy?", "Cooking"),
    new SearchCase("What should I renew when my HTTPS TLS certificate is expiring?", "Certificates"),
    new SearchCase("What system schedules containers into pods across cluster nodes?", "Containers")
};

foreach (var testCase in cases)
{
    var query = await embeddings.EmbedQueryAsync(testCase.Query);
    var results = await search.SearchAsync(query, indexed, x => x.Embeddings, new SemanticSearchRequest { Top = 3 });

    if (results.Count == 0 || results[0].Item.Page.Title != testCase.ExpectedTitle)
    {
        var actual = results.Count == 0 ? "<no result>" : results[0].Item.Page.Title;
        throw new InvalidOperationException(
            $"Jasper {precision} semantic-search smoke test expected '{testCase.ExpectedTitle}' first for '{testCase.Query}', but got '{actual}'.");
    }
    Console.WriteLine($"PASS {testCase.ExpectedTitle}: {results[0].Score:P1} - {testCase.Query}");
}

var concurrentTasks = Enumerable.Range(0, 8)
    .Select(i => embeddings.EmbedQueryAsync($"Concurrent embedding request {i}: PostgreSQL backup restore"))
    .ToArray();
var concurrentResults = await Task.WhenAll(concurrentTasks);
if (concurrentResults.Length != 8 || concurrentResults.Any(result => result.Vector.Dimensions != 2048))
    throw new InvalidOperationException("Shared-session concurrent inference smoke test failed.");
Console.WriteLine("PASS 8 concurrent query embeddings on one ONNX model instance.");

await host.StopAsync();

sealed record Page(string Title, string Content);
sealed record IndexedPage(Page Page, IReadOnlyList<TextEmbedding> Embeddings);
sealed record SearchCase(string Query, string ExpectedTitle);
