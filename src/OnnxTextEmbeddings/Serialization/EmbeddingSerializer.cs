using System.Buffers.Binary;
using System.Text;
using System.Text.Json;

namespace OnnxTextEmbeddings;

public static class EmbeddingSerializer
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = false
    };

    public static string SerializeJson(TextEmbedding embedding, bool indented = false)
    {
        ArgumentNullException.ThrowIfNull(embedding);
        return JsonSerializer.Serialize(embedding, new JsonSerializerOptions(JsonOptions) { WriteIndented = indented });
    }

    public static TextEmbedding DeserializeJson(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        try
        {
            var value = JsonSerializer.Deserialize<TextEmbedding>(json, JsonOptions) ??
                throw new EmbeddingSerializationException("Embedding JSON contained no record.");
            EnsureSchema(value.SchemaVersion);
            return value;
        }
        catch (EmbeddingSerializationException)
        {
            throw;
        }
        catch (Exception ex) when (ex is JsonException or NotSupportedException)
        {
            throw new EmbeddingSerializationException("Unable to deserialize the embedding record.", ex);
        }
    }

    public static string SerializeJson(QueryEmbedding embedding, bool indented = false)
    {
        ArgumentNullException.ThrowIfNull(embedding);
        return JsonSerializer.Serialize(embedding, new JsonSerializerOptions(JsonOptions) { WriteIndented = indented });
    }

    public static QueryEmbedding DeserializeQueryJson(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        try
        {
            var value = JsonSerializer.Deserialize<QueryEmbedding>(json, JsonOptions) ??
                throw new EmbeddingSerializationException("Query embedding JSON contained no record.");
            EnsureSchema(value.SchemaVersion);
            return value;
        }
        catch (EmbeddingSerializationException)
        {
            throw;
        }
        catch (Exception ex) when (ex is JsonException or NotSupportedException)
        {
            throw new EmbeddingSerializationException("Unable to deserialize the query embedding record.", ex);
        }
    }

    public static string SerializeJson(SingleEmbedding embedding, bool indented = false)
    {
        ArgumentNullException.ThrowIfNull(embedding);
        return JsonSerializer.Serialize(embedding, new JsonSerializerOptions(JsonOptions) { WriteIndented = indented });
    }

    public static SingleEmbedding DeserializeSingleJson(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        try
        {
            var value = JsonSerializer.Deserialize<SingleEmbedding>(json, JsonOptions) ??
                throw new EmbeddingSerializationException("Single-embedding JSON contained no record.");
            EnsureSchema(value.SchemaVersion);
            _ = value.Vector.ToFloat32();
            return value;
        }
        catch (EmbeddingSerializationException)
        {
            throw;
        }
        catch (Exception ex) when (ex is JsonException or NotSupportedException)
        {
            throw new EmbeddingSerializationException("Unable to deserialize the single-embedding record.", ex);
        }
    }

    public static byte[] SerializeVector(EmbeddingVector vector)
    {
        ArgumentNullException.ThrowIfNull(vector);
        _ = vector.ToFloat32(); // validates the vector without changing it

        var metadataLength = vector.Quantization is null ? 0 : 8;
        const int headerLength = 20;
        var output = new byte[headerLength + metadataLength + vector.Data.Length];
        var span = output.AsSpan();
        Encoding.ASCII.GetBytes("OTEV", span[..4]);
        BinaryPrimitives.WriteUInt16LittleEndian(span[4..6], checked((ushort)vector.EncodingVersion));
        span[6] = (byte)vector.Format;
        span[7] = (byte)(vector.Quantization?.Scheme ?? EmbeddingQuantizationScheme.None);
        BinaryPrimitives.WriteInt32LittleEndian(span[8..12], vector.Dimensions);
        BinaryPrimitives.WriteInt32LittleEndian(span[12..16], metadataLength);
        BinaryPrimitives.WriteInt32LittleEndian(span[16..20], vector.Data.Length);

        var cursor = headerLength;
        if (vector.Quantization is { } q)
        {
            BinaryPrimitives.WriteSingleLittleEndian(span[cursor..(cursor + 4)], q.Scale);
            BinaryPrimitives.WriteSingleLittleEndian(span[(cursor + 4)..(cursor + 8)], q.InverseIntegerNorm);
            cursor += 8;
        }
        vector.Data.CopyTo(span[cursor..]);
        return output;
    }

    public static EmbeddingVector DeserializeVector(ReadOnlySpan<byte> bytes)
    {
        const int headerLength = 20;
        if (bytes.Length < headerLength || !bytes[..4].SequenceEqual("OTEV"u8))
            throw new EmbeddingSerializationException("Invalid embedding-vector binary header.");

        var version = BinaryPrimitives.ReadUInt16LittleEndian(bytes[4..6]);
        if (version != EmbeddingProtocol.VectorEncodingVersion)
            throw new EmbeddingSerializationException($"Unsupported embedding-vector encoding version {version}.");

        var format = (EmbeddingVectorFormat)bytes[6];
        var scheme = (EmbeddingQuantizationScheme)bytes[7];
        var dimensions = BinaryPrimitives.ReadInt32LittleEndian(bytes[8..12]);
        var metadataLength = BinaryPrimitives.ReadInt32LittleEndian(bytes[12..16]);
        var payloadLength = BinaryPrimitives.ReadInt32LittleEndian(bytes[16..20]);
        if (dimensions <= 0 || metadataLength < 0 || payloadLength < 0 ||
            bytes.Length != headerLength + metadataLength + payloadLength)
            throw new EmbeddingSerializationException("Embedding-vector binary lengths are invalid.");

        EmbeddingQuantizationInfo? quantization = null;
        var cursor = headerLength;
        if (scheme != EmbeddingQuantizationScheme.None)
        {
            if (metadataLength != 8)
                throw new EmbeddingSerializationException("Quantized vectors require an 8-byte v1 metadata section.");
            quantization = new EmbeddingQuantizationInfo
            {
                Scheme = scheme,
                Scale = BinaryPrimitives.ReadSingleLittleEndian(bytes[cursor..(cursor + 4)]),
                InverseIntegerNorm = BinaryPrimitives.ReadSingleLittleEndian(bytes[(cursor + 4)..(cursor + 8)])
            };
            cursor += metadataLength;
        }
        else if (metadataLength != 0)
        {
            throw new EmbeddingSerializationException("Unquantized vectors cannot contain v1 quantization metadata.");
        }

        var vector = new EmbeddingVector
        {
            EncodingVersion = version,
            Format = format,
            Dimensions = dimensions,
            Quantization = quantization,
            Data = bytes.Slice(cursor, payloadLength).ToArray()
        };
        _ = vector.ToFloat32(); // validate payload / format agreement
        return vector;
    }

    private static void EnsureSchema(int schemaVersion)
    {
        if (schemaVersion <= 0 || schemaVersion > EmbeddingProtocol.SchemaVersion)
            throw new EmbeddingSerializationException($"Unsupported embedding record schema version {schemaVersion}.");
    }
}
