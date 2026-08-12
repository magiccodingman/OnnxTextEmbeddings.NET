using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using OnnxTextEmbeddings;

BenchmarkRunner.Run<VectorBenchmarks>();

[MemoryDiagnoser]
public class VectorBenchmarks
{
    private float[] _values = null!;
    private EmbeddingVector _query = null!;
    private EmbeddingVector _int4 = null!;
    private EmbeddingVector _int8 = null!;
    private EmbeddingVector _float16 = null!;
    private EmbeddingVector _float32 = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);
        _values = Enumerable.Range(0, 2048).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        EmbeddingVectorMath.NormalizeInPlace(_values);
        _query = EmbeddingVector.FromFloat32(_values, EmbeddingVectorFormat.Float32);
        _int4 = EmbeddingVector.FromFloat32(_values, EmbeddingVectorFormat.Int4);
        _int8 = EmbeddingVector.FromFloat32(_values, EmbeddingVectorFormat.Int8);
        _float16 = EmbeddingVector.FromFloat32(_values, EmbeddingVectorFormat.Float16);
        _float32 = EmbeddingVector.FromFloat32(_values, EmbeddingVectorFormat.Float32);
    }

    [Benchmark]
    public EmbeddingVector EncodeInt4() => EmbeddingVector.FromFloat32(_values, EmbeddingVectorFormat.Int4);

    [Benchmark]
    public EmbeddingVector EncodeInt8() => EmbeddingVector.FromFloat32(_values, EmbeddingVectorFormat.Int8);

    [Benchmark]
    public float CosineInt4() => EmbeddingVectorMath.CosineSimilarity(_query, _int4);

    [Benchmark]
    public float CosineInt8() => EmbeddingVectorMath.CosineSimilarity(_query, _int8);

    [Benchmark]
    public float CosineFloat16() => EmbeddingVectorMath.CosineSimilarity(_query, _float16);

    [Benchmark]
    public float CosineFloat32() => EmbeddingVectorMath.CosineSimilarity(_query, _float32);
}
