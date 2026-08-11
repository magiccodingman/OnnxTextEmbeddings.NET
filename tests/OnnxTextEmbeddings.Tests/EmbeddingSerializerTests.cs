namespace OnnxTextEmbeddings.Tests;

public sealed class EmbeddingSerializerTests
{
    [Theory]
    [InlineData(EmbeddingVectorFormat.Int4)]
    [InlineData(EmbeddingVectorFormat.Int8)]
    [InlineData(EmbeddingVectorFormat.Float16)]
    [InlineData(EmbeddingVectorFormat.Float32)]
    public void VectorBinaryRoundTrips(EmbeddingVectorFormat format)
    {
        var values = new[] { 0.1f, -0.2f, 0.3f, 0.4f, -0.5f };
        EmbeddingVectorMath.NormalizeInPlace(values);
        var vector = EmbeddingVector.FromFloat32(values, format);
        var bytes = EmbeddingSerializer.SerializeVector(vector);
        var restored = EmbeddingSerializer.DeserializeVector(bytes);
        Assert.Equal(format, restored.Format);
        Assert.Equal(vector.Dimensions, restored.Dimensions);
        Assert.Equal(vector.Data, restored.Data);
        Assert.Equal(vector.Quantization, restored.Quantization);
    }

    [Fact]
    public void EmbeddingRecordJsonRoundTrips()
    {
        var embedding = TestEmbedding();
        var json = EmbeddingSerializer.SerializeJson(embedding);
        var restored = EmbeddingSerializer.DeserializeJson(json);

        Assert.Equal(EmbeddingProtocol.SchemaVersion, restored.SchemaVersion);
        Assert.Equal(embedding.Identity, restored.Identity);
        Assert.Equal(embedding.Source, restored.Source);
        Assert.Equal(embedding.Chunk.Index, restored.Chunk.Index);
        Assert.Equal(embedding.Chunk.Count, restored.Chunk.Count);
        Assert.Equal(embedding.Chunk.BoundaryKind, restored.Chunk.BoundaryKind);
        Assert.Equal(embedding.Chunk.HeadingPath, restored.Chunk.HeadingPath);
        Assert.Equal(embedding.Chunk.ContextTokenCount, restored.Chunk.ContextTokenCount);
        Assert.Equal(embedding.Chunk.ModelPrefixTokenCount, restored.Chunk.ModelPrefixTokenCount);
        Assert.Equal(embedding.Chunk.SpecialTokenCount, restored.Chunk.SpecialTokenCount);
        Assert.Equal(embedding.Chunk.InputTokenCount, restored.Chunk.InputTokenCount);
        Assert.Equal(embedding.Text, restored.Text);
        Assert.Equal(embedding.Context, restored.Context);
    }

    [Fact]
    public void JsonIgnoresUnknownFutureProperties()
    {
        var json = EmbeddingSerializer.SerializeJson(TestEmbedding());
        json = json.TrimEnd('}') + ",\"futureField\":123}";
        var restored = EmbeddingSerializer.DeserializeJson(json);
        Assert.Equal("hello", restored.Text);
    }

    private static TextEmbedding TestEmbedding() => new()
    {
        Vector = EmbeddingVector.FromFloat32(new[] { 1f, 0f }, EmbeddingVectorFormat.Int8),
        Identity = new EmbeddingIdentity
        {
            ModelId = "test",
            SourceRevision = "abc",
            EmbeddingSpaceFingerprint = "fingerprint",
            IsNormalized = true
        },
        Source = new EmbeddingSource
        {
            DocumentTokenCount = 2,
            CharacterRange = new Utf16TextRange(0, 5),
            TokenRange = new TokenRange(0, 1),
            TokenCount = 1,
            TokenCapacity = 10
        },
        Chunk = new EmbeddingChunkInfo
        {
            Index = 0,
            Count = 1,
            BoundaryKind = ChunkBoundaryKind.WholeDocument,
            InputTokenCount = 2
        },
        Text = "hello"
    };
}
