# HTTP model manifest

The HTTP source accepts a compact JSON manifest. The currently implemented v1 shape is intentionally simple:

```json
{
  "modelId": "acme/my-embedder",
  "revision": "2026-08-11",
  "assets": [
    {
      "path": "model.onnx",
      "url": "model.onnx",
      "size": 123456789,
      "sha256": "0123456789abcdef..."
    },
    "tokenizer.json",
    "config.json"
  ]
}
```

`modelId` and `revision` are optional metadata. `assets` is required.

Each asset may be either a string path or an object:

- `path` — required destination path inside the snapshot.
- `url` — optional download URL; defaults to `path`.
- `size` — optional expected byte length.
- `sha256` — optional expected SHA-256 hex digest.

Relative URLs resolve against the manifest URI. The cache validates safe relative asset paths before writing them so a remote manifest cannot escape the snapshot directory.

The runtime still requires a usable ONNX model and `tokenizer.json`. Use `Model.ModelFile` when the asset set contains multiple ONNX graphs.

This format should be treated as a public protocol: future incompatible manifest semantics require explicit versioning rather than silently changing existing deployments.
