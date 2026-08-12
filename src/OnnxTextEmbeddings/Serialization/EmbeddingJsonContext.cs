using System.Text.Json;
using System.Text.Json.Serialization;

namespace OnnxTextEmbeddings;

[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    GenerationMode = JsonSourceGenerationMode.Metadata)]
[JsonSerializable(typeof(TextEmbedding))]
[JsonSerializable(typeof(TextEmbedding[]))]
[JsonSerializable(typeof(QueryEmbedding))]
[JsonSerializable(typeof(SingleEmbedding))]
internal sealed partial class EmbeddingJsonContext : JsonSerializerContext
{
    internal static EmbeddingJsonContext Compact { get; } = new(new JsonSerializerOptions(JsonSerializerDefaults.Web));

    internal static EmbeddingJsonContext Indented { get; } = new(new JsonSerializerOptions(JsonSerializerDefaults.Web)
    {
        WriteIndented = true
    });
}
