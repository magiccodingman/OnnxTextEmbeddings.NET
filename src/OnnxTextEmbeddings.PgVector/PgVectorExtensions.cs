using Pgvector;

namespace OnnxTextEmbeddings.PgVector;

/// <summary>Conversions between OnnxTextEmbeddings vector records and pgvector's native CLR types.</summary>
public static class PgVectorExtensions
{
    public static Vector ToPgVector(this EmbeddingVector vector)
    {
        ArgumentNullException.ThrowIfNull(vector);
        return new Vector(vector.ToFloat32());
    }

    public static HalfVector ToPgHalfVector(this EmbeddingVector vector)
    {
        ArgumentNullException.ThrowIfNull(vector);
        return new HalfVector(vector.ToFloat32().Select(x => (Half)x).ToArray());
    }

    public static EmbeddingVector ToEmbeddingVector(
        this Vector vector,
        EmbeddingVectorFormat format = EmbeddingVectorFormat.Float32)
    {
        ArgumentNullException.ThrowIfNull(vector);
        return EmbeddingVector.FromFloat32(vector.ToArray(), format);
    }

    public static EmbeddingVector ToEmbeddingVector(
        this HalfVector vector,
        EmbeddingVectorFormat format = EmbeddingVectorFormat.Float16)
    {
        ArgumentNullException.ThrowIfNull(vector);
        var values = vector.ToArray().Select(x => (float)x).ToArray();
        return EmbeddingVector.FromFloat32(values, format);
    }
}
