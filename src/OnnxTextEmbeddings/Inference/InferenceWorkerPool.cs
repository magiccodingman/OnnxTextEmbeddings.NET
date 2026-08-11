using System.Collections.Concurrent;
using System.Threading.Channels;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxTextEmbeddings;

public enum ModelInstanceHealth
{
    Starting = 0,
    Healthy = 1,
    Draining = 2,
    Recovering = 3,
    Faulted = 4,
    Disposed = 5
}

public sealed record ModelInstanceRuntimeInfo(
    int Index,
    ModelInstanceHealth Health,
    int ActiveRequests,
    int MaxConcurrentRequests,
    int Generation,
    int TotalRecoveries,
    int RecoveryAttempts,
    string? LastFailure);

internal interface IInferenceSessionHandle : IDisposable
{
    int? EmbeddingDimensions { get; }
    float[] Run(TokenizedModelInput input);
}

internal interface IInferenceSessionFactory
{
    IInferenceSessionHandle Create(string modelPath, int threadsPerModel);
}

internal sealed class OnnxInferenceSessionFactory : IInferenceSessionFactory
{
    public IInferenceSessionHandle Create(string modelPath, int threadsPerModel) =>
        new OnnxInferenceSessionHandle(modelPath, threadsPerModel);
}

internal sealed class RecoverableSessionException(string message, Exception innerException) : Exception(message, innerException);
internal sealed class SessionMemoryPressureException(string message, Exception innerException) : Exception(message, innerException);

internal sealed class OnnxInferenceSessionHandle : IInferenceSessionHandle
{
    private readonly InferenceSession _session;

    public OnnxInferenceSessionHandle(string modelPath, int threadsPerModel)
    {
        using var sessionOptions = new SessionOptions
        {
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            IntraOpNumThreads = threadsPerModel
        };
        _session = new InferenceSession(modelPath, sessionOptions);
        ValidateContract(_session);
        EmbeddingDimensions = ResolveEmbeddingDimensions(_session);
    }

    public int? EmbeddingDimensions { get; }

    public float[] Run(TokenizedModelInput input)
    {
        try
        {
            return RunCore(_session, input);
        }
        catch (OnnxRuntimeException ex) when (LooksLikeMemoryPressure(ex))
        {
            throw new SessionMemoryPressureException("ONNX Runtime reported a memory-allocation failure.", ex);
        }
        catch (OnnxRuntimeException ex)
        {
            throw new RecoverableSessionException("ONNX Runtime reported a session/runtime failure.", ex);
        }
    }

    private static float[] RunCore(InferenceSession session, TokenizedModelInput input)
    {
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
            throw new ModelValidationException($"Unsupported ONNX output rank/shape: [{string.Join(',', dimensions)}].");

        EmbeddingVectorMath.NormalizeInPlace(vector);
        return vector;
    }

    private static void ValidateContract(InferenceSession session)
    {
        var names = session.InputMetadata.Keys.ToArray();
        if (!names.Any(name => name.Equals("input_ids", StringComparison.OrdinalIgnoreCase)))
            throw new ModelValidationException("The ONNX model does not expose an input_ids input.");
        if (!names.Any(name => name.Equals("attention_mask", StringComparison.OrdinalIgnoreCase)))
            throw new ModelValidationException("The ONNX model does not expose an attention_mask input.");
        foreach (var name in names)
        {
            if (!name.Equals("input_ids", StringComparison.OrdinalIgnoreCase) &&
                !name.Equals("attention_mask", StringComparison.OrdinalIgnoreCase) &&
                !name.Equals("token_type_ids", StringComparison.OrdinalIgnoreCase))
                throw new ModelValidationException($"Unsupported required ONNX model input '{name}'. Configure a compatible sentence-embedding model.");
        }
    }

    private static bool LooksLikeMemoryPressure(Exception exception)
    {
        var message = exception.Message;
        return message.Contains("out of memory", StringComparison.OrdinalIgnoreCase) ||
               message.Contains("failed to allocate", StringComparison.OrdinalIgnoreCase) ||
               message.Contains("memory allocation", StringComparison.OrdinalIgnoreCase) ||
               message.Contains("not enough memory", StringComparison.OrdinalIgnoreCase);
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

    public void Dispose() => _session.Dispose();
}

internal sealed class InferenceWorkerPool : IAsyncDisposable
{
    private sealed record WorkItem(
        TokenizedModelInput Input,
        TaskCompletionSource<float[]> Completion,
        CancellationToken CancellationToken,
        int InfrastructureRetries = 0);

    private sealed class ModelInstance(
        int index,
        IInferenceSessionHandle session,
        int maxConcurrentRequests)
    {
        public int Index { get; } = index;
        public IInferenceSessionHandle? Session { get; set; } = session;
        public int MaxConcurrentRequests { get; } = maxConcurrentRequests;
        public int ActiveRequests { get; set; }
        public ModelInstanceHealth Health { get; set; } = ModelInstanceHealth.Healthy;
        public int Generation { get; set; } = 1;
        public int TotalRecoveries { get; set; }
        public int RecoveryAttempts { get; set; }
        public string? LastFailure { get; set; }
        public TaskCompletionSource<bool>? Drained { get; set; }
        public Task? RecoveryTask { get; set; }
    }

    private readonly Channel<WorkItem> _channel;
    private readonly ModelInstance[] _instances;
    private readonly IInferenceSessionFactory _sessionFactory;
    private readonly string _modelPath;
    private readonly ILogger? _logger;
    private readonly object _gate = new();
    private readonly SemaphoreSlim _capacitySignal = new(0);
    private readonly ConcurrentDictionary<long, Task> _inflight = new();
    private readonly CancellationTokenSource _shutdown = new();
    private readonly Task _scheduler;
    private long _nextWorkId;
    private int _tieBreakerCursor;
    private bool _disposed;

    public InferenceWorkerPool(
        string modelPath,
        InferenceOptions options,
        JasperModelPrecision? jasperPrecision = null,
        ILogger? logger = null)
        : this(modelPath, options.Resolve(jasperPrecision), new OnnxInferenceSessionFactory(), logger)
    {
    }

    internal InferenceWorkerPool(
        string modelPath,
        ResolvedInferenceOptions resolved,
        IInferenceSessionFactory sessionFactory,
        ILogger? logger = null)
    {
        _modelPath = modelPath;
        _sessionFactory = sessionFactory;
        _logger = logger;
        ModelInstanceCount = resolved.ModelInstanceCount;
        ThreadsPerModel = resolved.ThreadsPerModel;
        ConcurrentRequestsPerModel = resolved.ConcurrentRequestsPerModel;

        _channel = Channel.CreateBounded<WorkItem>(new BoundedChannelOptions(resolved.QueueCapacity)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = true,
            SingleWriter = false,
            AllowSynchronousContinuations = false
        });

        _instances = new ModelInstance[resolved.ModelInstanceCount];
        try
        {
            for (var i = 0; i < _instances.Length; i++)
            {
                var session = _sessionFactory.Create(modelPath, resolved.ThreadsPerModel);
                _instances[i] = new ModelInstance(i, session, resolved.ConcurrentRequestsPerModel);
            }
        }
        catch
        {
            foreach (var instance in _instances)
                instance?.Session?.Dispose();
            throw;
        }

        EmbeddingDimensions = _instances[0].Session?.EmbeddingDimensions;
        _scheduler = Task.Run(() => SchedulerLoopAsync(_shutdown.Token));
    }

    public int? EmbeddingDimensions { get; }
    public int ModelInstanceCount { get; }
    public int ThreadsPerModel { get; }
    public int ConcurrentRequestsPerModel { get; }
    public int TotalConcurrentRequests => ModelInstanceCount * ConcurrentRequestsPerModel;

    public IReadOnlyList<ModelInstanceRuntimeInfo> GetRuntimeInfo()
    {
        lock (_gate)
        {
            return _instances.Select(instance => new ModelInstanceRuntimeInfo(
                instance.Index,
                instance.Health,
                instance.ActiveRequests,
                instance.MaxConcurrentRequests,
                instance.Generation,
                instance.TotalRecoveries,
                instance.RecoveryAttempts,
                instance.LastFailure)).ToArray();
        }
    }

    public async Task<float[]> RunAsync(TokenizedModelInput input, CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var completion = new TaskCompletionSource<float[]>(TaskCreationOptions.RunContinuationsAsynchronously);
        var item = new WorkItem(input, completion, cancellationToken);
        await _channel.Writer.WriteAsync(item, cancellationToken).ConfigureAwait(false);
        return await completion.Task.WaitAsync(cancellationToken).ConfigureAwait(false);
    }

    private async Task SchedulerLoopAsync(CancellationToken cancellationToken)
    {
        WorkItem? pending = null;
        try
        {
            await foreach (var work in _channel.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
            {
                pending = work;
                if (work.CancellationToken.IsCancellationRequested)
                {
                    work.Completion.TrySetCanceled(work.CancellationToken);
                    pending = null;
                    continue;
                }

                ModelInstance? instance;
                while ((instance = TryReserveLeastLoaded()) is null)
                {
                    await _capacitySignal.WaitAsync(cancellationToken).ConfigureAwait(false);
                    if (work.CancellationToken.IsCancellationRequested)
                    {
                        work.Completion.TrySetCanceled(work.CancellationToken);
                        pending = null;
                        break;
                    }
                }

                if (pending is null || instance is null)
                    continue;

                StartExecution(instance, work);
                pending = null;
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            pending?.Completion.TrySetCanceled(cancellationToken);
        }
        finally
        {
            while (_channel.Reader.TryRead(out var queued))
                queued.Completion.TrySetCanceled(cancellationToken);
        }
    }

    private ModelInstance? TryReserveLeastLoaded()
    {
        lock (_gate)
        {
            var minimum = int.MaxValue;
            foreach (var instance in _instances)
            {
                if (instance.Health != ModelInstanceHealth.Healthy ||
                    instance.Session is null ||
                    instance.ActiveRequests >= instance.MaxConcurrentRequests)
                    continue;
                minimum = Math.Min(minimum, instance.ActiveRequests);
            }

            if (minimum == int.MaxValue)
                return null;

            for (var offset = 0; offset < _instances.Length; offset++)
            {
                var index = (_tieBreakerCursor + offset) % _instances.Length;
                var instance = _instances[index];
                if (instance.Health != ModelInstanceHealth.Healthy ||
                    instance.Session is null ||
                    instance.ActiveRequests >= instance.MaxConcurrentRequests ||
                    instance.ActiveRequests != minimum)
                    continue;

                instance.ActiveRequests++;
                _tieBreakerCursor = (index + 1) % _instances.Length;
                return instance;
            }

            return null;
        }
    }

    private void StartExecution(ModelInstance instance, WorkItem work)
    {
        var id = Interlocked.Increment(ref _nextWorkId);
        var task = ExecuteWorkAsync(instance, work);
        _inflight[id] = task;
        _ = ObserveCompletionAsync(id, task);
    }

    private async Task ObserveCompletionAsync(long id, Task task)
    {
        try { await task.ConfigureAwait(false); }
        finally { _inflight.TryRemove(id, out _); }
    }

    private async Task ExecuteWorkAsync(ModelInstance instance, WorkItem work)
    {
        var retry = false;
        try
        {
            if (work.CancellationToken.IsCancellationRequested)
            {
                work.Completion.TrySetCanceled(work.CancellationToken);
                return;
            }

            IInferenceSessionHandle session;
            lock (_gate)
                session = instance.Session ?? throw new InferenceException("The selected model instance no longer has an active session.");

            work.Completion.TrySetResult(session.Run(work.Input));
        }
        catch (Exception ex) when (ex is RecoverableSessionException or SessionMemoryPressureException or OutOfMemoryException)
        {
            var memoryPressure = ex is SessionMemoryPressureException or OutOfMemoryException;
            BeginRecovery(instance, ex, memoryPressure);

            if (!memoryPressure && work.InfrastructureRetries < 1 && !work.CancellationToken.IsCancellationRequested)
                retry = true;
            else
                work.Completion.TrySetException(new InferenceException(
                    memoryPressure
                        ? "ONNX embedding inference failed because the model instance encountered memory pressure and is being rebuilt."
                        : "ONNX embedding inference failed after a recoverable model-instance failure.",
                    ex));
        }
        catch (Exception ex)
        {
            work.Completion.TrySetException(ex is InferenceException
                ? ex
                : new InferenceException("ONNX embedding inference failed.", ex));
        }
        finally
        {
            ReleaseReservation(instance);
        }

        if (retry)
        {
            var retryWork = work with { InfrastructureRetries = work.InfrastructureRetries + 1 };
            try
            {
                await _channel.Writer.WriteAsync(retryWork, work.CancellationToken).ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is ChannelClosedException or OperationCanceledException)
            {
                if (work.CancellationToken.IsCancellationRequested)
                    work.Completion.TrySetCanceled(work.CancellationToken);
                else
                    work.Completion.TrySetException(new InferenceException("The inference retry could not be queued.", ex));
            }
        }
    }

    private void ReleaseReservation(ModelInstance instance)
    {
        var signalCapacity = false;
        TaskCompletionSource<bool>? drained = null;
        lock (_gate)
        {
            instance.ActiveRequests = Math.Max(0, instance.ActiveRequests - 1);
            if (instance.Health == ModelInstanceHealth.Healthy)
                signalCapacity = true;
            else if (instance.ActiveRequests == 0)
                drained = instance.Drained;
        }

        drained?.TrySetResult(true);
        if (signalCapacity)
            _capacitySignal.Release();
    }

    private void BeginRecovery(ModelInstance instance, Exception exception, bool memoryPressure)
    {
        var start = false;
        lock (_gate)
        {
            instance.LastFailure = exception.GetBaseException().Message;
            if (instance.Health == ModelInstanceHealth.Healthy)
            {
                instance.Health = ModelInstanceHealth.Draining;
                instance.Drained = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
                if (instance.ActiveRequests == 0)
                    instance.Drained.TrySetResult(true);
                start = true;
            }
        }

        if (!start)
            return;

        _logger?.LogWarning(exception,
            "Model instance {ModelInstance} marked unhealthy. It will drain active work and be rebuilt.",
            instance.Index);

        var task = RecoverInstanceAsync(instance, memoryPressure, _shutdown.Token);
        lock (_gate)
            instance.RecoveryTask = task;
        _ = ObserveRecoveryAsync(instance.Index, task);
    }

    private async Task ObserveRecoveryAsync(int instanceIndex, Task task)
    {
        try { await task.ConfigureAwait(false); }
        catch (OperationCanceledException) when (_shutdown.IsCancellationRequested) { }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Unexpected recovery-loop failure for model instance {ModelInstance}.", instanceIndex);
        }
    }

    private async Task RecoverInstanceAsync(ModelInstance instance, bool memoryPressure, CancellationToken cancellationToken)
    {
        Task drainedTask;
        lock (_gate)
            drainedTask = instance.Drained?.Task ?? Task.CompletedTask;
        await drainedTask.WaitAsync(cancellationToken).ConfigureAwait(false);

        IInferenceSessionHandle? oldSession;
        lock (_gate)
        {
            instance.Health = ModelInstanceHealth.Recovering;
            oldSession = instance.Session;
            instance.Session = null;
        }
        oldSession?.Dispose();

        var attempt = 0;
        while (!cancellationToken.IsCancellationRequested)
        {
            attempt++;
            lock (_gate)
            {
                instance.Health = ModelInstanceHealth.Recovering;
                instance.RecoveryAttempts = attempt;
            }

            var delay = GetRecoveryDelay(attempt, memoryPressure);
            if (delay > TimeSpan.Zero)
                await Task.Delay(delay, cancellationToken).ConfigureAwait(false);

            try
            {
                var fresh = await Task.Run(
                    () => _sessionFactory.Create(_modelPath, ThreadsPerModel),
                    cancellationToken).ConfigureAwait(false);

                lock (_gate)
                {
                    if (_disposed)
                    {
                        fresh.Dispose();
                        return;
                    }
                    instance.Session = fresh;
                    instance.Health = ModelInstanceHealth.Healthy;
                    instance.Generation++;
                    instance.TotalRecoveries++;
                    instance.RecoveryAttempts = 0;
                    instance.LastFailure = null;
                    instance.Drained = null;
                }

                _logger?.LogInformation(
                    "Model instance {ModelInstance} recovered successfully as generation {Generation}.",
                    instance.Index,
                    instance.Generation);
                _capacitySignal.Release();
                return;
            }
            catch (Exception ex) when (!cancellationToken.IsCancellationRequested)
            {
                lock (_gate)
                {
                    instance.Health = ModelInstanceHealth.Faulted;
                    instance.LastFailure = ex.GetBaseException().Message;
                }
                _logger?.LogWarning(ex,
                    "Model instance {ModelInstance} recovery attempt {Attempt} failed. The instance remains out of rotation.",
                    instance.Index,
                    attempt);
            }
        }
    }

    private static TimeSpan GetRecoveryDelay(int attempt, bool memoryPressure)
    {
        if (attempt <= 1)
            return memoryPressure ? TimeSpan.FromSeconds(1) : TimeSpan.Zero;
        var milliseconds = Math.Min(10_000, 250 * Math.Pow(2, Math.Min(attempt - 2, 6)));
        return TimeSpan.FromMilliseconds(milliseconds);
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;
        _channel.Writer.TryComplete();
        _shutdown.Cancel();
        try { await _scheduler.ConfigureAwait(false); }
        catch (OperationCanceledException) { }

        var inflight = _inflight.Values.ToArray();
        if (inflight.Length > 0)
        {
            try { await Task.WhenAll(inflight).ConfigureAwait(false); }
            catch { }
        }

        Task[] recoveryTasks;
        lock (_gate)
            recoveryTasks = _instances.Select(instance => instance.RecoveryTask).OfType<Task>().ToArray();
        if (recoveryTasks.Length > 0)
        {
            try { await Task.WhenAll(recoveryTasks).ConfigureAwait(false); }
            catch (OperationCanceledException) { }
        }

        lock (_gate)
        {
            foreach (var instance in _instances)
            {
                instance.Session?.Dispose();
                instance.Session = null;
                instance.Health = ModelInstanceHealth.Disposed;
            }
        }

        _capacitySignal.Dispose();
        _shutdown.Dispose();
    }
}
