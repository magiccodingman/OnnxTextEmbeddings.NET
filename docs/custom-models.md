# Custom ONNX models

Custom models are supported, but the package does not claim that every arbitrary transformer export is automatically compatible.

A usable custom source needs:

- an ONNX graph selected unambiguously
- `tokenizer.json`
- `input_ids`
- `attention_mask`
- an embedding output shape the runtime can interpret as one vector per input
- any additional runtime data files referenced by the ONNX graph

The model should produce deterministic text embeddings and should be appropriate for cosine similarity. The current runtime normalizes returned vectors before storage/search when required by the resolved model contract.

Use a Hugging Face repository for the simplest distribution path, a local directory for offline provisioning, or an HTTP manifest for application-controlled hosting.

When replacing the model, treat the embedding-space fingerprint as the migration boundary. Persisted vectors may remain valid across packaging/revision changes only when the fingerprint remains the same.
