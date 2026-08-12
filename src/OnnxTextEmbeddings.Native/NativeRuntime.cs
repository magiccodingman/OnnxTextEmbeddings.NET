using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace OnnxTextEmbeddings.Native;

internal sealed class NativeEmbeddingService : IDisposable
{
    private readonly ServiceProvider provider;

    public NativeEmbeddingService(OteOptions options)
    {
        var services = new ServiceCollection();
        services.AddLogging();
        services.AddOnnxTextEmbeddings(configuration => ApplyOptions(configuration, options));
        provider = services.BuildServiceProvider();
        Service = provider.GetRequiredService<ITextEmbeddingService>();
    }

    public ITextEmbeddingService Service { get; }

    public void Dispose() => provider.DisposeAsync().AsTask().GetAwaiter().GetResult();

    private static void ApplyOptions(OnnxTextEmbeddingsOptions target, OteOptions source)
    {
        target.Initialization.WarmupOnStartup = false;
        if (source.ModelPrecision != 0)
        {
            if (!Enum.IsDefined(typeof(JasperModelPrecision), source.ModelPrecision))
                throw new ArgumentOutOfRangeException(nameof(source.ModelPrecision));
            target.Model.UseJasper((JasperModelPrecision)source.ModelPrecision);
        }
        if (source.DocumentMaxTokens > 0) target.DocumentChunkMaxTokens = source.DocumentMaxTokens;
        if (source.QueryMaxTokens > 0) target.QueryMaxTokens = source.QueryMaxTokens;
        if (source.ModelInstanceCount > 0) target.Inference.ModelInstanceCount = source.ModelInstanceCount;
        if (source.ThreadsPerModel > 0) target.Inference.ThreadsPerModel = source.ThreadsPerModel;
        if (source.ConcurrentRequestsPerModel > 0) target.Inference.ConcurrentRequestsPerModel = source.ConcurrentRequestsPerModel;
        if (source.QueueCapacity > 0) target.Inference.QueueCapacity = source.QueueCapacity;
        if (source.DocumentVectorFormat != 0) target.Vectors.DocumentFormat = ParseFormat(source.DocumentVectorFormat);
        if (source.QueryVectorFormat != 0) target.Vectors.QueryFormat = ParseFormat(source.QueryVectorFormat);
    }

    private static EmbeddingVectorFormat ParseFormat(int value)
    {
        if (!Enum.IsDefined(typeof(EmbeddingVectorFormat), value) || value == 0)
            throw new ArgumentOutOfRangeException(nameof(value));
        return (EmbeddingVectorFormat)value;
    }
}

internal static unsafe class NativeRuntime
{
    internal const uint AbiVersion = 1;

    [ThreadStatic]
    private static string? lastError;

    internal static void ClearError() => lastError = null;
    internal static void SetError(Exception exception) => lastError = exception.GetBaseException().Message;
    internal static void SetError(string message) => lastError = message;
    internal static string LastError => lastError ?? string.Empty;

    internal static OteStatus MapException(Exception exception) => exception switch
    {
        ArgumentException => OteStatus.InvalidArgument,
        QueryTokenLimitExceededException => OteStatus.QueryTooLong,
        EmbeddingSpaceMismatchException => OteStatus.EmbeddingSpaceMismatch,
        EmbeddingSerializationException => OteStatus.SerializationError,
        OutOfMemoryException => OteStatus.OutOfMemory,
        ModelSourceException or ModelDownloadException or ModelValidationException or InferenceException => OteStatus.ModelError,
        _ => OteStatus.InternalError
    };

    internal static NativeEmbeddingService GetService(nint handle)
    {
        if (handle == 0)
            throw new ArgumentException("The service handle is null.", nameof(handle));
        try
        {
            var gcHandle = GCHandle.FromIntPtr(handle);
            return gcHandle.Target as NativeEmbeddingService ?? throw new ArgumentException("The service handle is invalid.", nameof(handle));
        }
        catch (Exception ex) when (ex is not ArgumentException)
        {
            throw new ArgumentException("The service handle is invalid.", nameof(handle), ex);
        }
    }

    internal static string ReadUtf8(byte* data, nuint length)
    {
        if (length > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(length));
        if (data is null && length != 0)
            throw new ArgumentNullException(nameof(data));
        return length == 0 ? string.Empty : Encoding.UTF8.GetString(new ReadOnlySpan<byte>(data, checked((int)length)));
    }

    internal static ReadOnlySpan<byte> ReadBytes(byte* data, nuint length)
    {
        if (length > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(length));
        if (data is null && length != 0)
            throw new ArgumentNullException(nameof(data));
        return length == 0 ? ReadOnlySpan<byte>.Empty : new ReadOnlySpan<byte>(data, checked((int)length));
    }

    internal static void WriteBuffer(ReadOnlySpan<byte> bytes, OteBuffer* output)
    {
        if (output is null)
            throw new ArgumentNullException(nameof(output));
        output->Data = null;
        output->Length = 0;
        if (bytes.IsEmpty)
            return;
        var memory = (byte*)NativeMemory.Alloc((nuint)bytes.Length);
        if (memory is null)
            throw new OutOfMemoryException();
        bytes.CopyTo(new Span<byte>(memory, bytes.Length));
        output->Data = memory;
        output->Length = (nuint)bytes.Length;
    }

    internal static EmbeddingVectorFormat? ParseOptionalFormat(int value)
    {
        if (value == 0)
            return null;
        if (!Enum.IsDefined(typeof(EmbeddingVectorFormat), value))
            throw new ArgumentOutOfRangeException(nameof(value));
        return (EmbeddingVectorFormat)value;
    }
}
