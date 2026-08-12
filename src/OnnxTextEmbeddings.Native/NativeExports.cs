using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace OnnxTextEmbeddings.Native;

internal static unsafe class NativeExports
{
    [UnmanagedCallersOnly(EntryPoint = "ote_abi_version", CallConvs = [typeof(CallConvCdecl)])]
    public static uint AbiVersion() => NativeRuntime.AbiVersion;

    [UnmanagedCallersOnly(EntryPoint = "ote_get_last_error", CallConvs = [typeof(CallConvCdecl)])]
    public static int GetLastError(byte* buffer, nuint bufferLength, nuint* requiredLength)
    {
        try
        {
            var bytes = Encoding.UTF8.GetBytes(NativeRuntime.LastError);
            if (requiredLength is not null)
                *requiredLength = (nuint)bytes.Length;
            if (buffer is null || bufferLength < (nuint)bytes.Length)
                return bytes.Length == 0 ? (int)OteStatus.Ok : (int)OteStatus.BufferTooSmall;
            bytes.CopyTo(new Span<byte>(buffer, bytes.Length));
            return (int)OteStatus.Ok;
        }
        catch
        {
            return (int)OteStatus.InternalError;
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "ote_service_create", CallConvs = [typeof(CallConvCdecl)])]
    public static int ServiceCreate(OteOptions* options, nint* outputHandle)
    {
        NativeRuntime.ClearError();
        NativeEmbeddingService? service = null;
        try
        {
            if (outputHandle is null)
                throw new ArgumentNullException(nameof(outputHandle));
            *outputHandle = 0;
            var resolved = options is null ? default : *options;
            if (options is not null)
            {
                if (resolved.AbiVersion != 0 && resolved.AbiVersion != NativeRuntime.AbiVersion)
                    throw new ArgumentException($"Unsupported ABI version {resolved.AbiVersion}.");
                if (resolved.StructSize != 0 && resolved.StructSize < (uint)sizeof(OteOptions))
                    throw new ArgumentException("ote_options.struct_size is smaller than the v1 structure.");
            }
            service = new NativeEmbeddingService(resolved);
            *outputHandle = NativeRuntime.AddService(service);
            service = null;
            return (int)OteStatus.Ok;
        }
        catch (Exception ex)
        {
            service?.Dispose();
            NativeRuntime.SetError(ex);
            return (int)NativeRuntime.MapException(ex);
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "ote_service_destroy", CallConvs = [typeof(CallConvCdecl)])]
    public static int ServiceDestroy(nint handle)
    {
        NativeRuntime.ClearError();
        try
        {
            if (handle == 0)
                return (int)OteStatus.Ok;
            NativeRuntime.RemoveService(handle).Dispose();
            return (int)OteStatus.Ok;
        }
        catch (Exception ex)
        {
            NativeRuntime.SetError(ex);
            return (int)NativeRuntime.MapException(ex);
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "ote_service_wait_ready", CallConvs = [typeof(CallConvCdecl)])]
    public static int ServiceWaitReady(nint handle)
    {
        NativeRuntime.ClearError();
        try
        {
            NativeRuntime.GetService(handle).Service.WaitUntilReadyAsync().GetAwaiter().GetResult();
            return (int)OteStatus.Ok;
        }
        catch (Exception ex)
        {
            NativeRuntime.SetError(ex);
            return (int)NativeRuntime.MapException(ex);
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "ote_service_model_dimensions", CallConvs = [typeof(CallConvCdecl)])]
    public static int ServiceModelDimensions(nint handle, int* dimensions)
    {
        NativeRuntime.ClearError();
        try
        {
            if (dimensions is null)
                throw new ArgumentNullException(nameof(dimensions));
            var service = NativeRuntime.GetService(handle).Service;
            service.WaitUntilReadyAsync().GetAwaiter().GetResult();
            *dimensions = service.ModelInfo?.Dimensions ?? 0;
            return (int)OteStatus.Ok;
        }
        catch (Exception ex)
        {
            NativeRuntime.SetError(ex);
            return (int)NativeRuntime.MapException(ex);
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "ote_count_tokens", CallConvs = [typeof(CallConvCdecl)])]
    public static int CountTokens(nint handle, byte* text, nuint textLength, int* tokenCount)
    {
        NativeRuntime.ClearError();
        try
        {
            if (tokenCount is null)
                throw new ArgumentNullException(nameof(tokenCount));
            var value = NativeRuntime.ReadUtf8(text, textLength);
            *tokenCount = NativeRuntime.GetService(handle).Service.CountTokensAsync(value).GetAwaiter().GetResult();
            return (int)OteStatus.Ok;
        }
        catch (Exception ex)
        {
            NativeRuntime.SetError(ex);
            return (int)NativeRuntime.MapException(ex);
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "ote_count_query_tokens", CallConvs = [typeof(CallConvCdecl)])]
    public static int CountQueryTokens(nint handle, byte* text, nuint textLength, OteQueryTokenCount* output)
    {
        NativeRuntime.ClearError();
        try
        {
            if (output is null)
                throw new ArgumentNullException(nameof(output));
            var value = NativeRuntime.ReadUtf8(text, textLength);
            var count = NativeRuntime.GetService(handle).Service.CountQueryTokensAsync(value).GetAwaiter().GetResult();
            *output = new OteQueryTokenCount
            {
                SourceTokenCount = count.SourceTokenCount,
                InputTokenCount = count.InputTokenCount,
                QueryMaxTokens = count.QueryMaxTokens,
                ModelMaxTokens = count.ModelMaxTokens ?? 0,
                HasModelMaxTokens = count.ModelMaxTokens.HasValue ? 1 : 0,
                Fits = count.Fits ? 1 : 0
            };
            return (int)OteStatus.Ok;
        }
        catch (Exception ex)
        {
            NativeRuntime.SetError(ex);
            return (int)NativeRuntime.MapException(ex);
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "ote_embed_query_json", CallConvs = [typeof(CallConvCdecl)])]
    public static int EmbedQueryJson(nint handle, byte* text, nuint textLength, int vectorFormat, OteBuffer* output)
    {
        NativeRuntime.ClearError();
        try
        {
            var value = NativeRuntime.ReadUtf8(text, textLength);
            var format = NativeRuntime.ParseOptionalFormat(vectorFormat);
            var service = NativeRuntime.GetService(handle).Service;
            var embedding = format is null
                ? service.EmbedQueryAsync(value).GetAwaiter().GetResult()
                : service.EmbedQueryAsync(value, format.Value).GetAwaiter().GetResult();
            NativeRuntime.WriteBuffer(Encoding.UTF8.GetBytes(EmbeddingSerializer.SerializeJson(embedding)), output);
            return (int)OteStatus.Ok;
        }
        catch (Exception ex)
        {
            NativeRuntime.SetError(ex);
            return (int)NativeRuntime.MapException(ex);
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "ote_embed_document_json", CallConvs = [typeof(CallConvCdecl)])]
    public static int EmbedDocumentJson(nint handle, byte* text, nuint textLength, int vectorFormat, OteBuffer* output)
    {
        NativeRuntime.ClearError();
        try
        {
            var value = NativeRuntime.ReadUtf8(text, textLength);
            var format = NativeRuntime.ParseOptionalFormat(vectorFormat);
            var service = NativeRuntime.GetService(handle).Service;
            var embeddings = format is null
                ? service.EmbedDocumentAsync(value).GetAwaiter().GetResult()
                : service.EmbedDocumentAsync(value, format.Value).GetAwaiter().GetResult();
            NativeRuntime.WriteBuffer(Encoding.UTF8.GetBytes(EmbeddingSerializer.SerializeJson(embeddings)), output);
            return (int)OteStatus.Ok;
        }
        catch (Exception ex)
        {
            NativeRuntime.SetError(ex);
            return (int)NativeRuntime.MapException(ex);
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "ote_vector_convert", CallConvs = [typeof(CallConvCdecl)])]
    public static int VectorConvert(byte* vectorBytes, nuint vectorLength, int targetFormat, OteBuffer* output)
    {
        NativeRuntime.ClearError();
        try
        {
            var format = NativeRuntime.ParseOptionalFormat(targetFormat) ?? throw new ArgumentOutOfRangeException(nameof(targetFormat));
            var vector = EmbeddingSerializer.DeserializeVector(NativeRuntime.ReadBytes(vectorBytes, vectorLength));
            NativeRuntime.WriteBuffer(EmbeddingSerializer.SerializeVector(vector.ConvertTo(format)), output);
            return (int)OteStatus.Ok;
        }
        catch (Exception ex)
        {
            NativeRuntime.SetError(ex);
            return (int)NativeRuntime.MapException(ex);
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "ote_vector_cosine", CallConvs = [typeof(CallConvCdecl)])]
    public static int VectorCosine(byte* left, nuint leftLength, byte* right, nuint rightLength, float* similarity)
    {
        NativeRuntime.ClearError();
        try
        {
            if (similarity is null)
                throw new ArgumentNullException(nameof(similarity));
            var leftVector = EmbeddingSerializer.DeserializeVector(NativeRuntime.ReadBytes(left, leftLength));
            var rightVector = EmbeddingSerializer.DeserializeVector(NativeRuntime.ReadBytes(right, rightLength));
            *similarity = EmbeddingVectorMath.CosineSimilarity(leftVector, rightVector);
            return (int)OteStatus.Ok;
        }
        catch (Exception ex)
        {
            NativeRuntime.SetError(ex);
            return (int)NativeRuntime.MapException(ex);
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "ote_vector_cosine_f32", CallConvs = [typeof(CallConvCdecl)])]
    public static int VectorCosineFloat32(float* left, int leftDimensions, float* right, int rightDimensions, float* similarity)
    {
        NativeRuntime.ClearError();
        try
        {
            if (left is null || right is null || similarity is null)
                throw new ArgumentNullException("Vector pointers and output similarity are required.");
            if (leftDimensions <= 0 || leftDimensions != rightDimensions)
                throw new ArgumentException("Float32 vector dimensions must be positive and equal.");
            var leftVector = EmbeddingVector.FromFloat32(new ReadOnlySpan<float>(left, leftDimensions));
            var rightVector = EmbeddingVector.FromFloat32(new ReadOnlySpan<float>(right, rightDimensions));
            *similarity = EmbeddingVectorMath.CosineSimilarity(leftVector, rightVector);
            return (int)OteStatus.Ok;
        }
        catch (Exception ex)
        {
            NativeRuntime.SetError(ex);
            return (int)NativeRuntime.MapException(ex);
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "ote_reduce_query_json", CallConvs = [typeof(CallConvCdecl)])]
    public static int ReduceQueryJson(byte* json, nuint jsonLength, int outputDimensions, int outputFormat, OteBuffer* output)
    {
        NativeRuntime.ClearError();
        try
        {
            var embedding = EmbeddingSerializer.DeserializeQueryJson(NativeRuntime.ReadUtf8(json, jsonLength));
            var reduced = embedding.ReduceDimensions(outputDimensions, NativeRuntime.ParseOptionalFormat(outputFormat));
            NativeRuntime.WriteBuffer(Encoding.UTF8.GetBytes(EmbeddingSerializer.SerializeJson(reduced)), output);
            return (int)OteStatus.Ok;
        }
        catch (Exception ex)
        {
            NativeRuntime.SetError(ex);
            return (int)NativeRuntime.MapException(ex);
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "ote_reduce_text_embedding_json", CallConvs = [typeof(CallConvCdecl)])]
    public static int ReduceTextEmbeddingJson(byte* json, nuint jsonLength, int outputDimensions, int outputFormat, OteBuffer* output)
    {
        NativeRuntime.ClearError();
        try
        {
            var embedding = EmbeddingSerializer.DeserializeJson(NativeRuntime.ReadUtf8(json, jsonLength));
            var reduced = embedding.ReduceDimensions(outputDimensions, NativeRuntime.ParseOptionalFormat(outputFormat));
            NativeRuntime.WriteBuffer(Encoding.UTF8.GetBytes(EmbeddingSerializer.SerializeJson(reduced)), output);
            return (int)OteStatus.Ok;
        }
        catch (Exception ex)
        {
            NativeRuntime.SetError(ex);
            return (int)NativeRuntime.MapException(ex);
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "ote_combine_to_single_json", CallConvs = [typeof(CallConvCdecl)])]
    public static int CombineToSingleJson(byte* json, nuint jsonLength, int outputDimensions, int outputFormat, OteBuffer* output)
    {
        NativeRuntime.ClearError();
        try
        {
            var embeddings = EmbeddingSerializer.DeserializeDocumentJson(NativeRuntime.ReadUtf8(json, jsonLength));
            var combined = embeddings.CombineToSingle(new SingleEmbeddingOptions
            {
                OutputDimensions = outputDimensions > 0 ? outputDimensions : null,
                OutputFormat = NativeRuntime.ParseOptionalFormat(outputFormat)
            });
            NativeRuntime.WriteBuffer(Encoding.UTF8.GetBytes(EmbeddingSerializer.SerializeJson(combined)), output);
            return (int)OteStatus.Ok;
        }
        catch (Exception ex)
        {
            NativeRuntime.SetError(ex);
            return (int)NativeRuntime.MapException(ex);
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "ote_buffer_free", CallConvs = [typeof(CallConvCdecl)])]
    public static void BufferFree(OteBuffer* buffer)
    {
        if (buffer is null)
            return;
        if (buffer->Data is not null)
            NativeMemory.Free(buffer->Data);
        buffer->Data = null;
        buffer->Length = 0;
    }
}
