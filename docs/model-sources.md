# Model sources

## Jasper presets

```csharp
options.Model.UseJasper(JasperModelPrecision.Int8);    // default
options.Model.UseJasper(JasperModelPrecision.Int4);
options.Model.UseJasper(JasperModelPrecision.Float32);
```

These map to the three `magiccodingman/Jasper-Token-Compression-600M-ONNX-*` repositories.

## Custom Hugging Face repository

```csharp
options.Model.UseHuggingFace("owner/repository", revision: "main");
```

A repository must expose at least one `.onnx` file and `tokenizer.json`. Runtime-associated `.json`, `.txt`, `.model`, `.data`, and `.onnx_data` files are downloaded as well. README and `.gitattributes` are intentionally excluded.

Pin a commit/tag/branch by setting `revision`. For a private repository set `options.Model.AccessToken`; the token is sent as a bearer token to Hugging Face.

When multiple ONNX files exist, set `options.Model.ModelFile` explicitly so model selection is deterministic.

## Local directory / offline deployment

```csharp
options.Model.UseLocalDirectory("/opt/my-embedding-model");
```

Use this for fully offline images or deployments where model assets are provisioned separately. The directory must contain the tokenizer and the selected ONNX runtime assets.

## HTTP manifest

```csharp
options.Model.UseHttpManifest(new Uri("https://example.com/models/embed/manifest.json"));
```

The manifest lists model assets and can provide SHA-256 hashes and expected sizes. Relative URLs resolve against the manifest URL. See [model-manifest.md](model-manifest.md).

## Update policies

- `OnStartup` checks the configured remote source during initialization and can activate a newer revision.
- `Manual` uses a valid cache until `UpdateModelAsync` is called.
- `Never` is intended for fixed/offline deployments.

A remote update failure never replaces a working runtime.
