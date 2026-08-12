# Model artifacts, cache, and updates

OnnxTextEmbeddings.NET now delegates generic artifact acquisition/cache mechanics to the reusable `ModelArtifacts.NET` NuGet package while retaining all embedding-specific validation and identity policy.

## Ownership

`ModelArtifacts.NET` owns:

```text
source revision resolution
explicit artifact selection
download/retry/Retry-After handling
path safety
size/SHA-256 transfer verification
staging
cross-process cache locking
candidate snapshots
offline current-snapshot fallback
atomic promotion/discard
cleanup
```

OnnxTextEmbeddings.NET owns:

```text
Jasper/custom source presets
required ONNX/tokenizer selection
embedding manifest interpretation
model token limits
tokenizer construction
ONNX embedding contract validation
pooling/normalization
embedding-space fingerprint compatibility
```

## Activation sequence

A downloaded artifact is deliberately **not** active just because transfer verification succeeded.

```text
ModelArtifacts.ResolveCandidateAsync
          ↓
verified candidate snapshot
          ↓
OnnxTextEmbeddings resolves model/tokenizer files
          ↓
create tokenizer
          ↓
create OnnxModelRuntime + embedding executor
          ↓
run real validation inference
          ↓ success
ModelArtifacts.PromoteAsync
          ↓
swap active tokenizer/runtime
          ↓
dispose old runtime/tokenizer
          ↓
ModelArtifacts.CleanupAsync
```

If application validation fails, the candidate is discarded and the previous current snapshot is untouched.

## Offline behavior

When a remote source cannot be resolved but a valid current managed snapshot exists, ModelArtifacts returns that current snapshot as an offline fallback. The embedding service can therefore continue starting/serving with known-good artifacts during a Hugging Face/CDN/network outage.

Without any current snapshot, the source failure is surfaced normally.

## Hot swap

```csharp
bool changed = await embeddingService.UpdateModelAsync();
```

A new candidate is fully tokenizer/runtime validated before promotion. Only after it works does the service swap references. Requests already executing may finish on the previous `OnnxModelRuntime`; new requests use the replacement after the swap.

Old sessions are disposed before old managed snapshots are cleaned, avoiding common Windows locked-native-file failures.

## Artifact fingerprint vs embedding-space fingerprint

These are intentionally different responsibilities.

`ArtifactFingerprint` belongs to ModelArtifacts.NET and identifies the selected acquired files/content.

`EmbeddingSpaceFingerprint` belongs to OnnxTextEmbeddings.NET and is persisted with vectors to answer a stronger question: can these vectors legitimately be compared?

The refactor preserves the historical OnnxTextEmbeddings fingerprint algorithm over the embedding runtime asset subset so existing persisted vectors are not invalidated merely because generic cache ownership moved into ModelArtifacts.NET.

A genuine embedding-space fingerprint change still means persisted vectors need regeneration before comparison with new queries. Search throws `EmbeddingSpaceMismatchException` instead of returning invalid rankings.

## Multiple processes

Cross-process cache locking, atomic `current.json` activation, abandoned staging cleanup, and locked-file deletion retries are now supplied by ModelArtifacts.NET rather than duplicate code in this repository.
