using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using OnnxTextEmbeddings;

var identity = new EmbeddingIdentity
{
    ModelId = "aot-smoke",
    SourceRevision = "1",
    EmbeddingSpaceFingerprint = "aot-space",
    IsNormalized = true
};
var embedding = new TextEmbedding
{
    Vector = EmbeddingVector.FromFloat32(new[] { 1f, 0f, 0f, 0f }),
    Identity = identity,
    Source = new EmbeddingSource
    {
        DocumentTokenCount = 4,
        CharacterRange = new Utf16TextRange(0, 4),
        TokenRange = new TokenRange(0, 4),
        TokenCount = 4,
        TokenCapacity = 4
    },
    Chunk = new EmbeddingChunkInfo
    {
        Index = 0,
        Count = 1,
        BoundaryKind = ChunkBoundaryKind.WholeDocument,
        InputTokenCount = 4
    },
    Text = "test"
};

var json = EmbeddingSerializer.SerializeJson(embedding);
var restored = EmbeddingSerializer.DeserializeJson(json);
var reduced = restored.ReduceDimensions(2);
if (reduced.Vector.Dimensions != 2 || reduced.DimensionReduction?.ProfileId != EmbeddingDimensionReductionProfiles.SrhtV1)
    throw new InvalidOperationException("AOT-safe serialization/dimension reduction smoke failed.");

var documentJson = EmbeddingSerializer.SerializeJson(new[] { embedding, embedding with { Chunk = embedding.Chunk with { Index = 1, Count = 2 } } });
if (EmbeddingSerializer.DeserializeDocumentJson(documentJson).Count != 2)
    throw new InvalidOperationException("AOT-safe document-array serialization smoke failed.");

Console.WriteLine("PASS Native AOT core serialization/vector smoke.");

if (Environment.GetEnvironmentVariable("AOT_JASPER_SMOKE") == "1")
{
    var services = new ServiceCollection();
    services.AddLogging(builder => builder.SetMinimumLevel(LogLevel.Warning));
    services.AddOnnxTextEmbeddings(options => options.Initialization.WarmupOnStartup = false);
    await using var provider = services.BuildServiceProvider();
    var service = provider.GetRequiredService<ITextEmbeddingService>();
    await service.WaitUntilReadyAsync();
    var query = await service.EmbedQueryAsync("restore a PostgreSQL database backup");
    if (query.Vector.Dimensions != 2048)
        throw new InvalidOperationException($"Expected Jasper 2048 dimensions, got {query.Vector.Dimensions}.");
    _ = EmbeddingSerializer.DeserializeQueryJson(EmbeddingSerializer.SerializeJson(query));
    Console.WriteLine("PASS Native AOT Jasper inference smoke.");
}
