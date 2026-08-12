namespace OnnxTextEmbeddings;

/// <summary>Embedding-facing projection of the generic OnnxModelRuntime.NET instance health model.</summary>
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
