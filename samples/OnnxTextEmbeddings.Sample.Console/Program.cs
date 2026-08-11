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
var expectedConcurrency = precision == JasperModelPrecision.Int8 ? 5 : 4;
if (embeddings.ModelInfo is not { ModelInstanceCount: 1, ThreadsPerModel: 16 } runtime ||
    runtime.ConcurrentRequestsPerModel != expectedConcurrency || runtime.HealthyModelInstanceCount != 1)
    throw new InvalidOperationException($"Unexpected inference topology for Jasper {precision}; expected one healthy model, 16 threads, and {expectedConcurrency} concurrent requests/model.");

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

var longDocument = "# Backup Operations\n\n" + string.Join(' ', Enumerable.Repeat("database backup restoration procedure", 120));
var defaultChunks = await embeddings.EmbedDocumentAsync(longDocument);
var smallChunks = await embeddings.EmbedDocumentAsync(
    longDocument,
    new EmbeddingRequestOptions
    {
        MaxTokens = 64,
        VectorFormat = EmbeddingVectorFormat.Int8
    });
if (smallChunks.Count <= defaultChunks.Count || smallChunks.Any(chunk => chunk.Chunk.InputTokenCount > 64) ||
    smallChunks.Any(chunk => chunk.Vector.Format != EmbeddingVectorFormat.Int8))
    throw new InvalidOperationException("Per-call document token-limit override did not control chunking as expected.");
Console.WriteLine($"PASS per-call document chunk override: {defaultChunks.Count} default chunk(s) vs {smallChunks.Count} at 64 tokens.");

const string normalQuery = "How do I restore my PostgreSQL database backup?";
var sourceTokenCount = await embeddings.CountTokensAsync(normalQuery);
var queryTokenCount = await embeddings.CountQueryTokensAsync(normalQuery);
if (sourceTokenCount <= 0 || queryTokenCount.SourceTokenCount != sourceTokenCount || !queryTokenCount.Fits)
    throw new InvalidOperationException("Token-count API returned unexpected values for a normal query.");

var strictCount = await embeddings.CountQueryTokensAsync(
    normalQuery,
    new QueryEmbeddingRequestOptions { MaxTokens = 4 });
if (strictCount.Fits)
    throw new InvalidOperationException("A stricter per-call query limit should be reflected by CountQueryTokensAsync.");
try
{
    _ = await embeddings.EmbedQueryAsync(normalQuery, new QueryEmbeddingRequestOptions { MaxTokens = 4 });
    throw new InvalidOperationException("Per-call query limit should reject input above that request's ceiling.");
}
catch (QueryTokenLimitExceededException)
{
    Console.WriteLine("PASS stricter per-call query limit.");
}

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

var overrideWords = 1400;
var upwardOverrideVerified = false;
while (overrideWords >= 200)
{
    var candidate = string.Join(' ', Enumerable.Repeat("backup", overrideWords));
    var defaultCount = await embeddings.CountQueryTokensAsync(candidate);
    if (defaultCount.InputTokenCount > defaultCount.QueryMaxTokens &&
        (defaultCount.ModelMaxTokens is null || defaultCount.InputTokenCount <= defaultCount.ModelMaxTokens))
    {
        var request = new QueryEmbeddingRequestOptions
        {
            MaxTokens = defaultCount.InputTokenCount,
            VectorFormat = EmbeddingVectorFormat.Float16
        };
        var overrideCount = await embeddings.CountQueryTokensAsync(candidate, request);
        if (!overrideCount.Fits)
            throw new InvalidOperationException("Per-call query override should make a model-supported query fit.");
        var overrideEmbedding = await embeddings.EmbedQueryAsync(candidate, request);
        if (overrideEmbedding.Vector.Format != EmbeddingVectorFormat.Float16)
            throw new InvalidOperationException("Query request options should combine token-limit and vector-format overrides.");
        upwardOverrideVerified = true;
        Console.WriteLine($"PASS upward per-call query limit override: {defaultCount.QueryMaxTokens} -> {overrideCount.QueryMaxTokens} tokens.");
        break;
    }
    overrideWords -= 100;
}
if (!upwardOverrideVerified)
    Console.WriteLine("INFO loaded model does not expose headroom above the configured 1024-token query default; downward override behavior was still verified.");

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

var burstSize = expectedConcurrency + 3;
var concurrentTasks = Enumerable.Range(0, burstSize)
    .Select(i => embeddings.EmbedQueryAsync($"Concurrent embedding request {i}: PostgreSQL backup restore"))
    .ToArray();
var concurrentResults = await Task.WhenAll(concurrentTasks);
if (concurrentResults.Length != burstSize || concurrentResults.Any(result => result.Vector.Dimensions != 2048))
    throw new InvalidOperationException("Concurrent inference/queueing smoke test failed.");
Console.WriteLine($"PASS burst of {burstSize} requests through {expectedConcurrency} concurrent slot(s) on one ONNX model instance.");

await host.StopAsync();

sealed record Page(string Title, string Content);
sealed record IndexedPage(Page Page, IReadOnlyList<TextEmbedding> Embeddings);
sealed record SearchCase(string Query, string ExpectedTitle);
