using OnnxTextEmbeddings;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddOnnxTextEmbeddings();

var app = builder.Build();

app.MapGet("/embeddings/status", (ITextEmbeddingService service) => new
{
    service.Status.State,
    service.Status.Message,
    service.ModelInfo
});

app.MapPost("/embeddings", async (
    EmbedRequest request,
    ITextEmbeddingService service,
    CancellationToken cancellationToken) =>
{
    var result = await service.EmbedDocumentAsync(request.Text, cancellationToken);
    return Results.Ok(result);
});

app.MapPost("/query", async (
    QueryRequest request,
    ITextEmbeddingService service,
    CancellationToken cancellationToken) =>
{
    var result = await service.EmbedQueryAsync(request.Query, cancellationToken);
    return Results.Ok(result);
});

app.Run();

sealed record EmbedRequest(string Text);
sealed record QueryRequest(string Query);
