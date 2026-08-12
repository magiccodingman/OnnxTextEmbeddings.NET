# Native AOT compatibility

`OnnxTextEmbeddings.NET` declares `IsAotCompatible=true` and treats Native AOT support as a continuously tested compatibility promise.

This does **not** mean the normal NuGet library is precompiled native code. A .NET application can consume the same package normally, while an application choosing Native AOT can publish the complete application ahead of time.

## AOT-safe core

Reflection-backed `System.Text.Json` metadata has been replaced by source-generated serialization metadata for the persisted embedding protocol and internal cache records.

CI publishes and executes a Native AOT smoke application on:

- Linux;
- Windows;
- macOS.

The smoke verifies serialization and deterministic dimension reduction without requiring the model. Native dependency behavior is additionally exercised through the native interoperability build and real Jasper smoke.

## Native shared-library facade

The repository also contains:

```text
src/OnnxTextEmbeddings.Native
```

This project references the canonical managed core and publishes it as a Native AOT shared library. It is intentionally **not a NuGet package**: its consumers are C-compatible native callers rather than managed package consumers.

See [Native interoperability](native-interop.md).
