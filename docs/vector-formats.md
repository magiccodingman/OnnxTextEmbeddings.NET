# Vector formats and conversion

Model precision and returned/stored vector precision are separate choices.

The default Jasper model is INT8 because that is an inference-model choice. The public embedding API now defaults returned **document and query vectors to FP32** because FP32 is the most interoperable representation with databases, vector libraries, and external APIs.

## Supported formats

| Format | Approx. payload for 2048 dimensions | Characteristics |
|---|---:|---|
| INT4 | 1 KiB | Packed, aggressive lossy quantization |
| INT8 | 2 KiB | Compact lossy quantization; recommended storage option |
| FP16 | 4 KiB | Standard half precision |
| FP32 | 8 KiB | Default; maximum compatibility/fidelity |

Payload figures exclude protocol/database metadata.

## Global defaults

```csharp
builder.Services.AddOnnxTextEmbeddings(options =>
{
    options.Vectors.DocumentFormat = EmbeddingVectorFormat.Int8;
    options.Vectors.QueryFormat = EmbeddingVectorFormat.Float32;
});
```

The package itself defaults both values to `Float32`. For applications storing many vectors, `Int8` is the recommended document-storage setting because it reduces the raw vector payload by roughly 4x relative to FP32 while retaining strong retrieval quality in the package's Jasper tests.

## Per-call formats

Configured formats are defaults, not restrictions:

```csharp
var normal = await service.EmbedDocumentAsync(text); // configured default
var int4 = await service.EmbedDocumentAsync(text, EmbeddingVectorFormat.Int4);
var int8 = await service.EmbedDocumentAsync(text, EmbeddingVectorFormat.Int8);
var fp16 = await service.EmbedDocumentAsync(text, EmbeddingVectorFormat.Float16);
var fp32 = await service.EmbedDocumentAsync(text, EmbeddingVectorFormat.Float32);

var queryFp32 = await service.EmbedQueryAsync(query);
var queryInt8 = await service.EmbedQueryAsync(query, EmbeddingVectorFormat.Int8);
```

The built-in ONNX service applies the requested representation directly to the original FP32 inference output. A per-call FP32 request therefore never travels through INT8/INT4 first.

## Convert existing float32 values

```csharp
EmbeddingVector fp32 = EmbeddingVector.FromFloat32(values);
EmbeddingVector fp16 = EmbeddingVector.FromFloat32(values, EmbeddingVectorFormat.Float16);
EmbeddingVector int8 = EmbeddingVector.FromFloat32(values, EmbeddingVectorFormat.Int8);
EmbeddingVector int4 = EmbeddingVector.FromFloat32(values, EmbeddingVectorFormat.Int4);
```

`FromFloat32(values)` intentionally preserves FP32 when the format argument is omitted.

Existing `EmbeddingVector` values can also be converted:

```csharp
EmbeddingVector compact = fp32.ConvertTo(EmbeddingVectorFormat.Int8);
float[] reconstructed = compact.ToFloat32();
```

### Expansion is not fidelity recovery

The library can represent an INT4 or INT8 vector again as FP32 because some downstream APIs require float arrays. That operation is **dequantization**, not restoration.

```text
original FP32 → INT8 → reconstructed FP32
```

The last value still contains only the information retained by INT8. Precision discarded during FP32 → FP16/INT8/INT4 cannot be recovered later.

For that reason, when high fidelity may be needed later, keep the original FP32 vector or request FP32 directly from the embedding call.
