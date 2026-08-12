using System.Runtime.InteropServices;

namespace OnnxTextEmbeddings.Native;

internal enum OteStatus : int
{
    Ok = 0,
    InvalidArgument = 1,
    BufferTooSmall = 2,
    InvalidHandle = 3,
    ModelError = 4,
    QueryTooLong = 5,
    EmbeddingSpaceMismatch = 6,
    SerializationError = 7,
    OutOfMemory = 8,
    InternalError = 255
}

[StructLayout(LayoutKind.Sequential)]
internal struct OteOptions
{
    public uint StructSize;
    public uint AbiVersion;
    public int ModelPrecision;
    public int DocumentMaxTokens;
    public int QueryMaxTokens;
    public int ModelInstanceCount;
    public int ThreadsPerModel;
    public int ConcurrentRequestsPerModel;
    public int QueueCapacity;
    public int DocumentVectorFormat;
    public int QueryVectorFormat;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct OteBuffer
{
    public byte* Data;
    public nuint Length;
}

[StructLayout(LayoutKind.Sequential)]
internal struct OteQueryTokenCount
{
    public int SourceTokenCount;
    public int InputTokenCount;
    public int QueryMaxTokens;
    public int ModelMaxTokens;
    public int HasModelMaxTokens;
    public int Fits;
}
