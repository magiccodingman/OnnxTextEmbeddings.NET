using System.Threading.Channels;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxTextEmbeddings;

internal sealed class InferenceWorkerPool : IAsyncDisposable
{
    private sealed record WorkItem(TokenizedModelInput Input, TaskCompletionSource<float[]> Completion, CancellationToken CancellationToken);

    private readonly Channel<WorkItem> _channel;
    private readonly InferenceSession[] _sessions;
    private readonly Task[] _workers;
    private readonly CancellationTokenSource _shutdown = new();

    public InferenceWorkerPool(string modelPath, InferenceOptions options)
    {
        var workerCount = options.WorkerCount;
        var threads = options.ThreadsPerWorker > 0
            ? options.ThreadsPerWorker
            : Math.Max(1, Math.Min(options.MaximumAutoThreadsPerWorker, Environment.ProcessorCount / workerCount));

        _channel = Channel.CreateBounded<WorkItem>(new BoundedChannelOptions(options.QueueCapacity)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = false,
            SingleWriter = false,
            AllowSynchronousContinuations = false
        });

        _sessions = new InferenceSession[workerCount];
        for (var i = 0; i < workerCount; i++)
        {
            using var sessionOptions = new SessionOptions
            {
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                IntraOpNumThreads = threads
            };
            _sessions[i] = new InferenceSession(modelPath, sessionOptions);
        }

        EmbeddingDimensions = ResolveEmbeddingDimensions(_sessions[0]);
        _workers = _sessions.Select(session => Task.Run(() => WorkerLoopAsync(session, _shutdown.Token))).ToArray();
    }

    public int? EmbeddingDimensions { get; }

    public async Task<float[]> RunAsync(TokenizedModelInput input, CancellationToken cancellationToken)
    {
        var completion = new TaskCompletionSource<float[]>(TaskCreationOptions.RunContinuationsAsynchronously);
        var item = new WorkItem(input, completion, cancellationToken);
        await _channel.Writer.WriteAsync(item, cancellationToken).ConfigureAwait(false);
        return await completion.Task.WaitAsync(cancellationToken).ConfigureAwait(false);
    }

    private async Task WorkerLoopAsync(InferenceSession session, CancellationToken cancellationToken)
    {
        try
        {
            await foreach (var work in _channel.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
            {
                if (work.CancellationToken.IsCancellationRequested)
                {
                    work.Completion.TrySetCanceled(work.CancellationToken);
                    continue;
                }

                try
                {
                    work.Completion.TrySetResult(Run(session, work.Input));
                }
                catch (Exception ex)
                {
                    work.Completion.TrySetException(new InferenceException("ONNX embedding inference failed.", ex));
                }
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
        }
    }

    private static float[] Run(InferenceSession session, TokenizedModelInput input)
    {
        var sequenceLength = input.InputIds.Length;
        var inputs = new List<NamedOnnxValue>(3);
        foreach (var name in session.InputMetadata.Keys)
        {
            if (name.Equals("input_ids", StringComparison.OrdinalIgnoreCase))
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(input.InputIds, new[] { 1, sequenceLength })));
            }
            else if (name.Equals("attention_mask", StringComparison.OrdinalIgnoreCase))
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(input.AttentionMask, new[] { 1, sequenceLength })));
            }
            else if (name.Equals("token_type_ids", StringComparison.OrdinalIgnoreCase))
            {
                var typeIds = input.TokenTypeIds ?? new long[sequenceLength];
                inputs.Add(NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(typeIds, new[] { 1, sequenceLength })));
            }
            else
            {
                throw new ModelValidationException($"Unsupported required ONNX model input '{name}'. Configure a compatible sentence-embedding model.");
            }
        }

        using var results = session.Run(inputs);
        var first = results.FirstOrDefault() ?? throw new ModelValidationException("The ONNX model produced no outputs.");
        var tensor = first.AsTensor<float>();
        var dimensions = tensor.Dimensions.ToArray();
        var raw = first.AsEnumerable<float>().ToArray();

        float[] vector;
        if (dimensions.Length is 1 or 2)
        {
            vector = raw;
        }
        else if (dimensions.Length == 3 && dimensions[0] == 1 && dimensions[1] > 0 && dimensions[2] > 0)
        {
            var tokens = Math.Min(dimensions[1], sequenceLength);
            var width = dimensions[2];
            vector = new float[width];
            var included = 0;
            for (var token = 0; token < tokens; token++)
            {
                if (input.AttentionMask[token] == 0) continue;
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
        {
            throw new ModelValidationException($"Unsupported ONNX output rank/shape: [{string.Join(',', dimensions)}].");
        }

        EmbeddingVectorMath.NormalizeInPlace(vector);
        return vector;
    }

    private static int? ResolveEmbeddingDimensions(InferenceSession session)
    {
        var output = session.OutputMetadata.Values.FirstOrDefault();
        if (output is null) return null;
        var dimensions = output.Dimensions;
        if (dimensions.Length >= 2 && dimensions[^1] > 0)
            return dimensions[^1];
        if (dimensions.Length == 1 && dimensions[0] > 0)
            return dimensions[0];
        return null;
    }

    public async ValueTask DisposeAsync()
    {
        _channel.Writer.TryComplete();
        _shutdown.Cancel();
        try { await Task.WhenAll(_workers).ConfigureAwait(false); }
        catch (OperationCanceledException) { }
        foreach (var session in _sessions)
            session.Dispose();
        _shutdown.Dispose();
    }
}
