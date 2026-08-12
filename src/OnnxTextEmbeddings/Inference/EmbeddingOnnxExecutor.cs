using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxModelRuntime;

namespace OnnxTextEmbeddings;

/// <summary>
/// The embedding-specific boundary intentionally left in this package. OnnxModelRuntime.NET owns session hosting,
/// queueing, scheduling and recovery; this adapter owns the sentence-embedding tensor contract and output meaning.
/// </summary>
internal sealed class EmbeddingOnnxExecutor : IOnnxModelExecutor<TokenizedModelInput, float[]>
{
    private int _embeddingDimensions;

    public int? EmbeddingDimensions
    {
        get
        {
            var value = Volatile.Read(ref _embeddingDimensions);
            return value == 0 ? null : value;
        }
    }

    public float[] Execute(
        InferenceSession session,
        TokenizedModelInput input,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(input);
        cancellationToken.ThrowIfCancellationRequested();

        var sequenceLength = input.InputIds.Length;
        var inputs = new List<NamedOnnxValue>(3);
        foreach (var name in session.InputMetadata.Keys)
        {
            if (name.Equals("input_ids", StringComparison.OrdinalIgnoreCase))
                inputs.Add(NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(input.InputIds, new[] { 1, sequenceLength })));
            else if (name.Equals("attention_mask", StringComparison.OrdinalIgnoreCase))
                inputs.Add(NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(input.AttentionMask, new[] { 1, sequenceLength })));
            else if (name.Equals("token_type_ids", StringComparison.OrdinalIgnoreCase))
            {
                var typeIds = input.TokenTypeIds ?? new long[sequenceLength];
                inputs.Add(NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(typeIds, new[] { 1, sequenceLength })));
            }
            else
                throw new ModelValidationException($"Unsupported required ONNX model input '{name}'. Configure a compatible sentence-embedding model.");
        }

        if (!session.InputMetadata.Keys.Any(name => name.Equals("input_ids", StringComparison.OrdinalIgnoreCase)))
            throw new ModelValidationException("The ONNX model does not expose an input_ids input.");

        cancellationToken.ThrowIfCancellationRequested();
        using var results = session.Run(inputs);
        var first = results.FirstOrDefault() ?? throw new ModelValidationException("The ONNX model produced no outputs.");
        var tensor = first.AsTensor<float>();
        var dimensions = tensor.Dimensions.ToArray();
        var raw = first.AsEnumerable<float>().ToArray();

        float[] vector;
        if (dimensions.Length is 1 or 2)
            vector = raw;
        else if (dimensions.Length == 3 && dimensions[0] == 1 && dimensions[1] > 0 && dimensions[2] > 0)
        {
            var tokens = Math.Min(dimensions[1], sequenceLength);
            var width = dimensions[2];
            vector = new float[width];
            var included = 0;
            for (var token = 0; token < tokens; token++)
            {
                if (input.AttentionMask[token] == 0)
                    continue;
                var offset = token * width;
                for (var d = 0; d < width; d++)
                    vector[d] += raw[offset + d];
                included++;
            }

            if (included == 0)
                throw new ModelValidationException("The ONNX output cannot be pooled because the attention mask contains no active tokens.");
            for (var d = 0; d < vector.Length; d++)
                vector[d] /= included;
        }
        else
            throw new ModelValidationException($"Unsupported ONNX output rank/shape: [{string.Join(',', dimensions)}].");

        var previous = Interlocked.CompareExchange(ref _embeddingDimensions, vector.Length, 0);
        if (previous != 0 && previous != vector.Length)
            throw new ModelValidationException($"ONNX model instances disagree on embedding dimensions ({previous} vs {vector.Length}).");

        EmbeddingVectorMath.NormalizeInPlace(vector);
        return vector;
    }
}
