using System.Collections.Concurrent;

namespace OnnxTextEmbeddings.Tests;

public sealed class InferenceWorkerPoolTests
{
    private static readonly TokenizedModelInput Input = new([1, 2], [1, 1], null);

    [Fact]
    public async Task TwoIdleInstances_ReceiveOneRequestEachBeforeEitherGetsASecond()
    {
        using var gate = new ManualResetEventSlim(false);
        var factory = new FakeSessionFactory(_ => new FakeSession(_ =>
        {
            gate.Wait(TestContext.Current.CancellationToken);
            return [1f, 0f];
        }));
        await using var pool = new InferenceWorkerPool(
            "fake.onnx",
            new ResolvedInferenceOptions(2, 1, 2, 16),
            factory);

        var first = pool.RunAsync(Input, TestContext.Current.CancellationToken);
        var second = pool.RunAsync(Input, TestContext.Current.CancellationToken);

        try
        {
            await WaitUntilAsync(
                () => factory.Sessions.Count >= 2 && factory.Sessions.Take(2).All(session => session.RunCount >= 1),
                TestContext.Current.CancellationToken);

            Assert.Equal(1, factory.Sessions[0].RunCount);
            Assert.Equal(1, factory.Sessions[1].RunCount);
        }
        finally
        {
            gate.Set();
        }

        await Task.WhenAll(first, second);
    }

    [Fact]
    public async Task RecoverableFailure_QuarantinesInstanceRetriesElsewhereAndRestoresFreshGeneration()
    {
        using var recoveryGate = new ManualResetEventSlim(false);
        var factory = new FakeSessionFactory(createIndex => createIndex switch
        {
            0 => new FakeSession(_ => throw new RecoverableSessionException("session failed", new InvalidOperationException("native failure"))),
            1 => new FakeSession(_ => [0f, 1f]),
            _ => CreateBlockedRecoverySession(recoveryGate)
        });
        await using var pool = new InferenceWorkerPool(
            "fake.onnx",
            new ResolvedInferenceOptions(2, 1, 1, 16),
            factory);

        var result = await pool.RunAsync(Input, TestContext.Current.CancellationToken);
        Assert.Equal(new[] { 0f, 1f }, result);

        await WaitUntilAsync(
            () => factory.CreateCount >= 3 && pool.GetRuntimeInfo()[0].Health != ModelInstanceHealth.Healthy,
            TestContext.Current.CancellationToken);

        var duringRecovery = pool.GetRuntimeInfo()[0];
        Assert.Equal(0, duringRecovery.ActiveRequests);
        Assert.Contains(duringRecovery.Health, new[] { ModelInstanceHealth.Recovering, ModelInstanceHealth.Faulted });

        recoveryGate.Set();
        await WaitUntilAsync(
            () => pool.GetRuntimeInfo()[0] is { Health: ModelInstanceHealth.Healthy, Generation: 2 },
            TestContext.Current.CancellationToken);

        var recovered = pool.GetRuntimeInfo()[0];
        Assert.Equal(0, recovered.ActiveRequests);
        Assert.Equal(1, recovered.TotalRecoveries);
        Assert.Equal(0, recovered.RecoveryAttempts);
    }

    [Fact]
    public async Task SingleInstanceFailure_PausesQueuedWorkUntilFreshSessionIsHealthy()
    {
        using var recoveryGate = new ManualResetEventSlim(false);
        var factory = new FakeSessionFactory(createIndex => createIndex == 0
            ? new FakeSession(_ => throw new RecoverableSessionException("session failed", new InvalidOperationException("native failure")))
            : CreateBlockedRecoverySession(recoveryGate));
        await using var pool = new InferenceWorkerPool(
            "fake.onnx",
            new ResolvedInferenceOptions(1, 1, 1, 16),
            factory);

        var first = pool.RunAsync(Input, TestContext.Current.CancellationToken);
        var second = pool.RunAsync(Input, TestContext.Current.CancellationToken);

        await WaitUntilAsync(
            () => factory.CreateCount >= 2 && pool.GetRuntimeInfo()[0].Health != ModelInstanceHealth.Healthy,
            TestContext.Current.CancellationToken);

        Assert.False(first.IsCompleted);
        Assert.False(second.IsCompleted);
        Assert.Equal(0, pool.GetRuntimeInfo()[0].ActiveRequests);

        recoveryGate.Set();
        var results = await Task.WhenAll(first, second);
        Assert.All(results, result => Assert.Equal(new[] { 1f, 0f }, result));

        await WaitUntilAsync(
            () => pool.GetRuntimeInfo()[0] is { Health: ModelInstanceHealth.Healthy, Generation: 2, ActiveRequests: 0 },
            TestContext.Current.CancellationToken);
    }

    [Fact]
    public async Task MemoryPressureFailure_DoesNotImmediatelyRetryOnAnotherInstance()
    {
        using var recoveryGate = new ManualResetEventSlim(false);
        var factory = new FakeSessionFactory(createIndex => createIndex switch
        {
            0 => new FakeSession(_ => throw new OutOfMemoryException("simulated model allocation failure")),
            1 => new FakeSession(_ => [0f, 1f]),
            _ => CreateBlockedRecoverySession(recoveryGate)
        });
        await using var pool = new InferenceWorkerPool(
            "fake.onnx",
            new ResolvedInferenceOptions(2, 1, 1, 16),
            factory);

        try
        {
            await Assert.ThrowsAsync<InferenceException>(async () =>
                await pool.RunAsync(Input, TestContext.Current.CancellationToken));
            Assert.Equal(0, factory.Sessions[1].RunCount);
        }
        finally
        {
            recoveryGate.Set();
        }
    }

    private static FakeSession CreateBlockedRecoverySession(ManualResetEventSlim recoveryGate)
    {
        recoveryGate.Wait(TestContext.Current.CancellationToken);
        return new FakeSession(_ => [1f, 0f]);
    }

    private static async Task WaitUntilAsync(Func<bool> predicate, CancellationToken cancellationToken)
    {
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(TimeSpan.FromSeconds(5));
        while (!predicate())
            await Task.Delay(10, timeout.Token);
    }

    private sealed class FakeSessionFactory(Func<int, FakeSession> create) : IInferenceSessionFactory
    {
        private readonly ConcurrentQueue<FakeSession> _sessions = new();
        private int _createCount;

        public int CreateCount => Volatile.Read(ref _createCount);
        public IReadOnlyList<FakeSession> Sessions => _sessions.ToArray();

        public IInferenceSessionHandle Create(string modelPath, int threadsPerModel)
        {
            var index = Interlocked.Increment(ref _createCount) - 1;
            var session = create(index);
            _sessions.Enqueue(session);
            return session;
        }
    }

    private sealed class FakeSession(Func<TokenizedModelInput, float[]> run) : IInferenceSessionHandle
    {
        private int _runCount;
        public int? EmbeddingDimensions => 2;
        public int RunCount => Volatile.Read(ref _runCount);

        public float[] Run(TokenizedModelInput input)
        {
            Interlocked.Increment(ref _runCount);
            return run(input);
        }

        public void Dispose() { }
    }
}
