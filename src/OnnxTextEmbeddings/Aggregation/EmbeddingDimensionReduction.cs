using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;

namespace OnnxTextEmbeddings;

/// <summary>Deterministic dimension-reduction helpers for embedding-space-compatible query/document transforms.</summary>
public static class EmbeddingDimensionReduction
{
    public static QueryEmbedding ReduceDimensions(
        this QueryEmbedding query,
        int outputDimensions,
        EmbeddingVectorFormat? outputFormat = null,
        EmbeddingDimensionReductionStrategy strategy = EmbeddingDimensionReductionStrategy.Auto)
    {
        ArgumentNullException.ThrowIfNull(query);
        if (outputDimensions == query.Vector.Dimensions && outputFormat is null)
            return query;

        var sourceDimensions = query.Vector.Dimensions;
        ValidateDimensions(sourceDimensions, outputDimensions);
        var values = query.Vector.ToFloat32();
        EmbeddingVectorMath.NormalizeInPlace(values);

        var reduced = outputDimensions == sourceDimensions
            ? values
            : Reduce(values, outputDimensions, strategy);
        var format = outputFormat ?? query.Vector.Format;
        var profile = outputDimensions == sourceDimensions ? null : ResolveProfile(strategy);
        var identity = profile is null
            ? query.Identity with { IsNormalized = true }
            : CreateReducedIdentity(query.Identity, sourceDimensions, outputDimensions, profile);
        var reduction = profile is null
            ? query.DimensionReduction
            : CreateReductionInfo(profile, sourceDimensions, outputDimensions);

        return query with
        {
            Vector = EmbeddingVector.FromFloat32(reduced, format),
            Identity = identity,
            DimensionReduction = reduction
        };
    }

    /// <summary>
    /// Applies the same deterministic coordinate transform used for reduced queries to one direct document chunk.
    /// Chunk/source metadata is preserved; this is dimensional reduction, not aggregation.
    /// </summary>
    public static TextEmbedding ReduceDimensions(
        this TextEmbedding embedding,
        int outputDimensions,
        EmbeddingVectorFormat? outputFormat = null,
        EmbeddingDimensionReductionStrategy strategy = EmbeddingDimensionReductionStrategy.Auto)
    {
        ArgumentNullException.ThrowIfNull(embedding);
        if (outputDimensions == embedding.Vector.Dimensions && outputFormat is null)
            return embedding;

        var sourceDimensions = embedding.Vector.Dimensions;
        ValidateDimensions(sourceDimensions, outputDimensions);
        var values = embedding.Vector.ToFloat32();
        EmbeddingVectorMath.NormalizeInPlace(values);

        var reduced = outputDimensions == sourceDimensions
            ? values
            : Reduce(values, outputDimensions, strategy);
        var format = outputFormat ?? embedding.Vector.Format;
        var profile = outputDimensions == sourceDimensions ? null : ResolveProfile(strategy);
        var identity = profile is null
            ? embedding.Identity with { IsNormalized = true }
            : CreateReducedIdentity(embedding.Identity, sourceDimensions, outputDimensions, profile);
        var reduction = profile is null
            ? embedding.DimensionReduction
            : CreateReductionInfo(profile, sourceDimensions, outputDimensions);

        return embedding with
        {
            Vector = EmbeddingVector.FromFloat32(reduced, format),
            Identity = identity,
            DimensionReduction = reduction
        };
    }

    internal static float[] Reduce(
        ReadOnlySpan<float> source,
        int outputDimensions,
        EmbeddingDimensionReductionStrategy strategy)
    {
        ValidateDimensions(source.Length, outputDimensions);
        if (outputDimensions == source.Length)
            return source.ToArray();

        return ResolveProfile(strategy) switch
        {
            EmbeddingDimensionReductionProfiles.SrhtV1 => ReduceSrhtV1(source, outputDimensions),
            var profile => throw new ArgumentOutOfRangeException(nameof(strategy), $"Unsupported dimension-reduction profile '{profile}'.")
        };
    }

    internal static EmbeddingIdentity CreateReducedIdentity(
        EmbeddingIdentity source,
        int sourceDimensions,
        int outputDimensions,
        string profile)
    {
        var fingerprint = DeriveFingerprint(
            source.EmbeddingSpaceFingerprint,
            profile,
            sourceDimensions,
            outputDimensions);
        return source with
        {
            EmbeddingSpaceFingerprint = fingerprint,
            IsNormalized = true
        };
    }

    internal static EmbeddingDimensionReductionInfo CreateReductionInfo(
        string profile,
        int sourceDimensions,
        int outputDimensions) => new()
    {
        ProfileId = profile,
        ProfileVersion = 1,
        SourceDimensions = sourceDimensions,
        OutputDimensions = outputDimensions
    };

    internal static string ResolveProfile(EmbeddingDimensionReductionStrategy strategy) => strategy switch
    {
        EmbeddingDimensionReductionStrategy.Auto => EmbeddingDimensionReductionProfiles.SrhtV1,
        EmbeddingDimensionReductionStrategy.SrhtV1 => EmbeddingDimensionReductionProfiles.SrhtV1,
        _ => throw new ArgumentOutOfRangeException(nameof(strategy), strategy, "Unsupported dimension-reduction strategy.")
    };

    internal static string DeriveFingerprint(
        string baseFingerprint,
        string profile,
        int sourceDimensions,
        int outputDimensions)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(baseFingerprint);
        var payload = $"OnnxTextEmbeddings|EmbeddingSpaceTransform|{baseFingerprint}|{profile}|{sourceDimensions}|{outputDimensions}";
        return Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(payload)));
    }

    private static float[] ReduceSrhtV1(ReadOnlySpan<float> source, int outputDimensions)
    {
        var sourceDimensions = source.Length;
        var paddedDimensions = NextPowerOfTwo(sourceDimensions);
        var work = new double[paddedDimensions];
        for (var i = 0; i < sourceDimensions; i++)
            work[i] = source[i] * DeterministicSign(sourceDimensions, i);

        FastWalshHadamard(work);
        var normalization = 1.0 / Math.Sqrt(paddedDimensions);
        for (var i = 0; i < work.Length; i++)
            work[i] *= normalization;

        var coordinates = Enumerable.Range(0, paddedDimensions)
            .Select(index => new Coordinate(index, DeterministicCoordinateKey(sourceDimensions, index)))
            .OrderBy(x => x.Key)
            .ThenBy(x => x.Index)
            .Take(outputDimensions)
            .ToArray();

        var result = new float[outputDimensions];
        for (var i = 0; i < result.Length; i++)
            result[i] = (float)work[coordinates[i].Index];
        EmbeddingVectorMath.NormalizeInPlace(result);
        return result;
    }

    private static void FastWalshHadamard(Span<double> values)
    {
        for (var width = 1; width < values.Length; width <<= 1)
        {
            var block = width << 1;
            for (var start = 0; start < values.Length; start += block)
            {
                for (var offset = 0; offset < width; offset++)
                {
                    var left = values[start + offset];
                    var right = values[start + offset + width];
                    values[start + offset] = left + right;
                    values[start + offset + width] = left - right;
                }
            }
        }
    }

    private static int DeterministicSign(int sourceDimensions, int index)
    {
        var hash = StableHash("sign", sourceDimensions, index);
        return (hash & 1UL) == 0 ? 1 : -1;
    }

    private static ulong DeterministicCoordinateKey(int sourceDimensions, int index) =>
        StableHash("coordinate", sourceDimensions, index);

    private static ulong StableHash(string domain, int sourceDimensions, int index)
    {
        var payload = Encoding.UTF8.GetBytes($"OnnxTextEmbeddings|SRHT-v1|{domain}|{sourceDimensions}|{index}");
        var hash = SHA256.HashData(payload);
        return BinaryPrimitives.ReadUInt64LittleEndian(hash.AsSpan(0, sizeof(ulong)));
    }

    private static int NextPowerOfTwo(int value)
    {
        if (value <= 0)
            throw new ArgumentOutOfRangeException(nameof(value));
        var result = 1;
        while (result < value)
            result = checked(result << 1);
        return result;
    }

    private static void ValidateDimensions(int sourceDimensions, int outputDimensions)
    {
        if (sourceDimensions <= 0)
            throw new ArgumentOutOfRangeException(nameof(sourceDimensions));
        if (outputDimensions <= 0)
            throw new ArgumentOutOfRangeException(nameof(outputDimensions), "Output dimensions must be greater than zero.");
        if (outputDimensions > sourceDimensions)
            throw new ArgumentOutOfRangeException(
                nameof(outputDimensions),
                $"Requested embedding dimension {outputDimensions} exceeds the maximum available dimension of {sourceDimensions}.");
    }

    private readonly record struct Coordinate(int Index, ulong Key);
}
