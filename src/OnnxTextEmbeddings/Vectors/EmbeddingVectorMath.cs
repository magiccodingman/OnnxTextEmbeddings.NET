using System.Buffers.Binary;

namespace OnnxTextEmbeddings;

/// <summary>Conversion and cosine helpers for FP32, FP16, INT8, and packed INT4 vectors.</summary>
public static class EmbeddingVectorMath
{
    public static EmbeddingVector FromFloat32(
        ReadOnlySpan<float> values,
        EmbeddingVectorFormat format)
    {
        ValidateInput(values);

        return format switch
        {
            EmbeddingVectorFormat.Float32 => EncodeFloat32(values),
            EmbeddingVectorFormat.Float16 => EncodeFloat16(values),
            EmbeddingVectorFormat.Int8 => EncodeInt8(values),
            EmbeddingVectorFormat.Int4 => EncodeInt4(values),
            _ => throw new EmbeddingVectorFormatException($"Unsupported vector format: {format}.")
        };
    }

    public static EmbeddingVector Convert(
        EmbeddingVector vector,
        EmbeddingVectorFormat format)
    {
        ArgumentNullException.ThrowIfNull(vector);
        ValidateVector(vector);
        if (vector.Format == format)
            return vector;
        return FromFloat32(ToFloat32(vector), format);
    }

    public static float[] ToFloat32(EmbeddingVector vector)
    {
        ArgumentNullException.ThrowIfNull(vector);
        ValidateVector(vector);

        var result = new float[vector.Dimensions];
        switch (vector.Format)
        {
            case EmbeddingVectorFormat.Float32:
                for (var i = 0; i < result.Length; i++)
                    result[i] = BinaryPrimitives.ReadSingleLittleEndian(vector.Data.AsSpan(i * 4, 4));
                break;
            case EmbeddingVectorFormat.Float16:
                for (var i = 0; i < result.Length; i++)
                {
                    var bits = BinaryPrimitives.ReadUInt16LittleEndian(vector.Data.AsSpan(i * 2, 2));
                    result[i] = (float)BitConverter.UInt16BitsToHalf(bits);
                }
                break;
            case EmbeddingVectorFormat.Int8:
            {
                var q = RequireQuantization(vector, EmbeddingQuantizationScheme.SymmetricPerVectorInt8V1);
                for (var i = 0; i < result.Length; i++)
                    result[i] = unchecked((sbyte)vector.Data[i]) * q.Scale;
                break;
            }
            case EmbeddingVectorFormat.Int4:
            {
                var q = RequireQuantization(vector, EmbeddingQuantizationScheme.SymmetricPerVectorInt4V1);
                for (var i = 0; i < result.Length; i++)
                    result[i] = DecodeInt4(vector.Data, i) * q.Scale;
                break;
            }
            default:
                throw new EmbeddingVectorFormatException($"Unsupported vector format: {vector.Format}.");
        }

        return result;
    }

    public static float CosineSimilarity(EmbeddingVector left, EmbeddingVector right)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);
        ValidateVector(left);
        ValidateVector(right);

        if (left.Dimensions != right.Dimensions)
            throw new ArgumentException($"Vector dimensions differ: {left.Dimensions} vs {right.Dimensions}.");

        if (IsInteger(left.Format) && IsInteger(right.Format))
            return IntegerCosine(left, right);

        double dot = 0;
        double leftNorm = 0;
        double rightNorm = 0;
        for (var i = 0; i < left.Dimensions; i++)
        {
            var a = ReadComponent(left, i);
            var b = ReadComponent(right, i);
            dot += a * b;
            leftNorm += a * a;
            rightNorm += b * b;
        }

        if (leftNorm <= 0 || rightNorm <= 0)
            return 0;

        return (float)(dot / Math.Sqrt(leftNorm * rightNorm));
    }

    public static void NormalizeInPlace(Span<float> values)
    {
        ValidateInput(values);
        double sum = 0;
        foreach (var value in values)
            sum += value * value;
        if (sum <= 0)
            throw new EmbeddingVectorFormatException("Cannot normalize a zero vector.");
        var inv = (float)(1.0 / Math.Sqrt(sum));
        for (var i = 0; i < values.Length; i++)
            values[i] *= inv;
    }

    private static EmbeddingVector EncodeFloat32(ReadOnlySpan<float> values)
    {
        var data = new byte[values.Length * 4];
        for (var i = 0; i < values.Length; i++)
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(i * 4, 4), values[i]);
        return new EmbeddingVector
        {
            Format = EmbeddingVectorFormat.Float32,
            Dimensions = values.Length,
            Data = data
        };
    }

    private static EmbeddingVector EncodeFloat16(ReadOnlySpan<float> values)
    {
        var data = new byte[values.Length * 2];
        for (var i = 0; i < values.Length; i++)
        {
            var half = (Half)values[i];
            BinaryPrimitives.WriteUInt16LittleEndian(
                data.AsSpan(i * 2, 2),
                BitConverter.HalfToUInt16Bits(half));
        }
        return new EmbeddingVector
        {
            Format = EmbeddingVectorFormat.Float16,
            Dimensions = values.Length,
            Data = data
        };
    }

    private static EmbeddingVector EncodeInt8(ReadOnlySpan<float> values)
    {
        var maxAbs = MaxAbs(values);
        if (maxAbs <= 0)
            throw new EmbeddingVectorFormatException("Cannot quantize a zero vector.");

        var scale = maxAbs / 127f;
        var data = new byte[values.Length];
        double integerNormSquared = 0;
        for (var i = 0; i < values.Length; i++)
        {
            var q = Math.Clamp((int)MathF.Round(values[i] / scale), -127, 127);
            data[i] = unchecked((byte)(sbyte)q);
            integerNormSquared += q * q;
        }

        return new EmbeddingVector
        {
            Format = EmbeddingVectorFormat.Int8,
            Dimensions = values.Length,
            Data = data,
            Quantization = new EmbeddingQuantizationInfo
            {
                Scheme = EmbeddingQuantizationScheme.SymmetricPerVectorInt8V1,
                Scale = scale,
                InverseIntegerNorm = (float)(1.0 / Math.Sqrt(integerNormSquared))
            }
        };
    }

    private static EmbeddingVector EncodeInt4(ReadOnlySpan<float> values)
    {
        var maxAbs = MaxAbs(values);
        if (maxAbs <= 0)
            throw new EmbeddingVectorFormatException("Cannot quantize a zero vector.");

        var scale = maxAbs / 7f;
        var data = new byte[(values.Length + 1) / 2];
        double integerNormSquared = 0;
        for (var i = 0; i < values.Length; i++)
        {
            var q = Math.Clamp((int)MathF.Round(values[i] / scale), -7, 7);
            var nibble = (byte)(q & 0x0F);
            var byteIndex = i >> 1;
            if ((i & 1) == 0)
                data[byteIndex] = nibble;
            else
                data[byteIndex] |= (byte)(nibble << 4);
            integerNormSquared += q * q;
        }

        return new EmbeddingVector
        {
            Format = EmbeddingVectorFormat.Int4,
            Dimensions = values.Length,
            Data = data,
            Quantization = new EmbeddingQuantizationInfo
            {
                Scheme = EmbeddingQuantizationScheme.SymmetricPerVectorInt4V1,
                Scale = scale,
                InverseIntegerNorm = (float)(1.0 / Math.Sqrt(integerNormSquared))
            }
        };
    }

    private static float IntegerCosine(EmbeddingVector left, EmbeddingVector right)
    {
        var leftQ = left.Quantization ?? throw new EmbeddingVectorFormatException("Missing quantization metadata.");
        var rightQ = right.Quantization ?? throw new EmbeddingVectorFormatException("Missing quantization metadata.");
        long dot = 0;
        for (var i = 0; i < left.Dimensions; i++)
            dot += ReadIntegerComponent(left, i) * ReadIntegerComponent(right, i);
        return dot * leftQ.InverseIntegerNorm * rightQ.InverseIntegerNorm;
    }

    private static float ReadComponent(EmbeddingVector vector, int index) => vector.Format switch
    {
        EmbeddingVectorFormat.Float32 => BinaryPrimitives.ReadSingleLittleEndian(vector.Data.AsSpan(index * 4, 4)),
        EmbeddingVectorFormat.Float16 => (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(vector.Data.AsSpan(index * 2, 2))),
        EmbeddingVectorFormat.Int8 => ReadIntegerComponent(vector, index) * RequireQuantization(vector, EmbeddingQuantizationScheme.SymmetricPerVectorInt8V1).Scale,
        EmbeddingVectorFormat.Int4 => ReadIntegerComponent(vector, index) * RequireQuantization(vector, EmbeddingQuantizationScheme.SymmetricPerVectorInt4V1).Scale,
        _ => throw new EmbeddingVectorFormatException($"Unsupported vector format: {vector.Format}.")
    };

    private static int ReadIntegerComponent(EmbeddingVector vector, int index) => vector.Format switch
    {
        EmbeddingVectorFormat.Int8 => unchecked((sbyte)vector.Data[index]),
        EmbeddingVectorFormat.Int4 => DecodeInt4(vector.Data, index),
        _ => throw new EmbeddingVectorFormatException($"{vector.Format} is not an integer vector format.")
    };

    private static int DecodeInt4(ReadOnlySpan<byte> data, int index)
    {
        var packed = data[index >> 1];
        var nibble = (index & 1) == 0 ? packed & 0x0F : (packed >> 4) & 0x0F;
        return nibble >= 8 ? nibble - 16 : nibble;
    }

    private static bool IsInteger(EmbeddingVectorFormat format) =>
        format is EmbeddingVectorFormat.Int4 or EmbeddingVectorFormat.Int8;

    private static EmbeddingQuantizationInfo RequireQuantization(
        EmbeddingVector vector,
        EmbeddingQuantizationScheme expected)
    {
        var info = vector.Quantization ??
            throw new EmbeddingVectorFormatException("Quantized vector is missing quantization metadata.");
        if (info.Scheme != expected)
            throw new EmbeddingVectorFormatException($"Expected quantization scheme {expected}, found {info.Scheme}.");
        return info;
    }

    private static float MaxAbs(ReadOnlySpan<float> values)
    {
        var max = 0f;
        foreach (var value in values)
            max = Math.Max(max, Math.Abs(value));
        return max;
    }

    private static void ValidateInput(ReadOnlySpan<float> values)
    {
        if (values.IsEmpty)
            throw new EmbeddingVectorFormatException("Embedding vectors cannot be empty.");
        foreach (var value in values)
        {
            if (!float.IsFinite(value))
                throw new EmbeddingVectorFormatException("Embedding vectors cannot contain NaN or infinity.");
        }
    }

    private static void ValidateVector(EmbeddingVector vector)
    {
        if (vector.EncodingVersion != EmbeddingProtocol.VectorEncodingVersion)
            throw new EmbeddingVectorFormatException($"Unsupported vector encoding version {vector.EncodingVersion}.");
        if (vector.Dimensions <= 0)
            throw new EmbeddingVectorFormatException("Vector dimensions must be positive.");

        var expectedLength = vector.Format switch
        {
            EmbeddingVectorFormat.Float32 => checked(vector.Dimensions * 4),
            EmbeddingVectorFormat.Float16 => checked(vector.Dimensions * 2),
            EmbeddingVectorFormat.Int8 => vector.Dimensions,
            EmbeddingVectorFormat.Int4 => (vector.Dimensions + 1) / 2,
            _ => throw new EmbeddingVectorFormatException($"Unsupported vector format: {vector.Format}.")
        };

        if (vector.Data.Length != expectedLength)
            throw new EmbeddingVectorFormatException($"Vector payload length is {vector.Data.Length}; expected {expectedLength}.");
    }
}
