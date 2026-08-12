# Native interoperability and C ABI

`OnnxTextEmbeddings.Native` is the project's official low-level interoperability boundary for callers outside .NET.

The project maintains the C ABI and interoperability tests. Third-party language bindings are welcome, but are not automatically first-party supported SDKs.

Header:

```text
native/include/onnx_text_embeddings.h
```

## ABI v1 principles

- C ABI, not CLR object graphs.
- Explicit `OTE_ABI_VERSION`.
- `struct_size` and `abi_version` on options from v1.
- Opaque service handles.
- UTF-8 text passed as pointer + explicit length.
- Blocking calls in v1; foreign runtimes can schedule them on their own worker/task systems.
- Managed exceptions never cross the native boundary.
- Stable numeric status codes and a thread-local `ote_get_last_error` message.
- Memory allocated by the library is released with `ote_buffer_free`.
- No caller is required to pair a foreign allocator with .NET/native-AOT allocations.

## Exported functionality

ABI v1 exposes:

- ABI/version discovery;
- service create/destroy/readiness;
- model dimensions;
- source/query token counting;
- document and query embedding as versioned JSON records;
- OTEV vector conversion;
- cosine over OTEV vectors;
- direct float32 cosine;
- query/direct-chunk dimensional reduction;
- `CombineToSingle` over document-embedding JSON;
- explicit buffer release.

Generic `.NET` composition concepts such as `ISemanticSearch<T>` and database command/provider APIs are intentionally not mirrored into C v1.

## Native bundle and sidecars

The Native AOT facade is a self-contained .NET runtime artifact, but it still uses native libraries supplied by model/runtime dependencies such as the Hugging Face tokenizer binding and ONNX Runtime.

Treat the published directory as the native deployment bundle: keep the generated `OnnxTextEmbeddings.Native` shared library and its RID-specific native sidecars together on the host loader path. This is the same practical layout used by a Native-AOT executable that consumes those dependencies directly.

The C integration tests deliberately compile the host executable beside that published bundle so the test exercises the distribution layout a native consumer should preserve.

## Example C lifecycle

```c
ote_options options = {0};
options.struct_size = sizeof(options);
options.abi_version = OTE_ABI_VERSION;
options.model_precision = OTE_JASPER_INT8;

intptr_t service = 0;
if (ote_service_create(&options, &service) != OTE_OK) {
    /* read ote_get_last_error */
}

ote_service_wait_ready(service);

const char *query = "restore PostgreSQL backup";
ote_buffer result = {0};
ote_embed_query_json(
    service,
    (const uint8_t*)query,
    strlen(query),
    OTE_VECTOR_FLOAT32,
    &result);

/* consume result.data/result.length */
ote_buffer_free(&result);
ote_service_destroy(service);
```

## Continuous interop testing

The Native AOT workflow:

1. publishes and executes the managed core as a Native AOT application on Linux, Windows, and macOS;
2. publishes the Native AOT shared library;
3. compiles a standalone C program against the public header;
4. dynamically loads the generated `.so`, `.dll`, or `.dylib` and validates ABI v1 on Linux, Windows, and macOS;
5. on Linux, loads Jasper INT8 from a managed Native AOT executable;
6. in the same gate, runs from C all the way through the Native AOT shared library, model loading, Jasper query inference, returned JSON, buffer release, and service destruction.

This is the practical compatibility promise offered to Rust, C++, Go, Zig, Python FFI, Node native wrappers, and other ecosystems: they receive a stable C boundary to bind against without needing to reproduce the C# implementation.
