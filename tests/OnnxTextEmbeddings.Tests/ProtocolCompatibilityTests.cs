namespace OnnxTextEmbeddings.Tests;

public sealed class ProtocolCompatibilityTests
{
    [Fact]
    public void PublicProtocolNumericValuesAreStable()
    {
        Assert.Equal(1, EmbeddingProtocol.SchemaVersion);
        Assert.Equal(1, EmbeddingProtocol.VectorEncodingVersion);
        Assert.Equal(1, (int)EmbeddingVectorFormat.Int4);
        Assert.Equal(2, (int)EmbeddingVectorFormat.Int8);
        Assert.Equal(3, (int)EmbeddingVectorFormat.Float16);
        Assert.Equal(4, (int)EmbeddingVectorFormat.Float32);
        Assert.Equal(1, (int)EmbeddingQuantizationScheme.SymmetricPerVectorInt4V1);
        Assert.Equal(2, (int)EmbeddingQuantizationScheme.SymmetricPerVectorInt8V1);
    }

    [Fact]
    public void FutureJsonSchemaIsRejected()
    {
        var embedding = TestEmbedding();
        var json = EmbeddingSerializer.SerializeJson(embedding)
            .Replace("\"schemaVersion\":1", "\"schemaVersion\":99", StringComparison.Ordinal);

        Assert.Throws<EmbeddingSerializationException>(() => EmbeddingSerializer.DeserializeJson(json));
    }

    [Fact]
    public void FutureBinaryVectorEncodingIsRejected()
    {
        var vector = EmbeddingVector.FromFloat32(new[] { 0.6f, 0.8f }, EmbeddingVectorFormat.Int8);
        var bytes = EmbeddingSerializer.SerializeVector(vector);
        bytes[4] = 2;
        bytes[5] = 0;

        Assert.Throws<EmbeddingSerializationException>(() => EmbeddingSerializer.DeserializeVector(bytes));
    }

    [Fact]
    public void SameDimensionsDoNotOverrideEmbeddingSpaceIdentity()
    {
        var left = TestEmbedding();
        var right = left with { Identity = left.Identity with { EmbeddingSpaceFingerprint = "different" } };
        Assert.Equal(left.Vector.Dimensions, right.Vector.Dimensions);
        Assert.NotEqual(left.Identity.EmbeddingSpaceFingerprint, right.Identity.EmbeddingSpaceFingerprint);
    }

    private static TextEmbedding TestEmbedding() => new()
    {
        Vector = EmbeddingVector.FromFloat32(new[] { 0.6f, 0.8f }, EmbeddingVectorFormat.Int8),
        Identity = new EmbeddingIdentity
        {
            ModelId = "test",
            SourceRevision = "r1",
            EmbeddingSpaceFingerprint = "space-a",
            IsNormalized = true
        },
        Source = new EmbeddingSource
        {
            DocumentTokenCount = 2,
            CharacterRange = new Utf16TextRange(0, 4),
            TokenRange = new TokenRange(0, 2),
            TokenCount = 2,
            TokenCapacity = 10
        },
        Chunk = new EmbeddingChunkInfo
        {
            Index = 0,
            Count = 1,
            BoundaryKind = ChunkBoundaryKind.WholeDocument,
            InputTokenCount = 2
        },
        Text = "test"
    };
}
