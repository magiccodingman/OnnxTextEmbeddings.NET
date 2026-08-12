# Embedding dimension reduction

Embedding vectors may optionally be returned in fewer dimensions than were supplied.

Dimension reduction is separate from semantic aggregation and numeric/storage conversion:

```text
full-dimensional semantic representation
        ↓
dimension reduction
        ↓
normalize
        ↓
FP32 / FP16 / INT8 / INT4 storage conversion
```

The library never increases dimensionality or presents zero-padding as semantic recovery.

## Direct document chunks

Individual `TextEmbedding` records can now be reduced without aggregating a document:

```csharp
var reducedChunk = chunk.ReduceDimensions(
    1024,
    EmbeddingVectorFormat.Float32);
```

Source ranges, chunk metadata, text, and context remain attached to the same direct chunk. `DimensionReduction` records the transform and the identity receives the deterministic reduced-space fingerprint.

This is important for database backends with native dimensional limits. It preserves multi-vector retrieval rather than forcing a long document through `CombineToSingle()`.

Queries use the same operation:

```csharp
var reducedQuery = query.ReduceDimensions(1024);
```

When base fingerprint, source dimensions, profile, and target dimensions match, the reduced document/query fingerprints match.

## SRHT-v1

When no explicitly supported model-native lower-dimensional representation exists, `Auto` currently uses deterministic `SRHT-v1` (Subsampled Randomized Hadamard Transform).

Conceptually:

```text
source vector
    ↓
deterministic ±1 sign transform
    ↓
Walsh-Hadamard transform
    ↓
deterministic coordinate ordering/subsampling
    ↓
requested dimensions
    ↓
L2 normalize
```

For source dimensions that are not powers of two, the implementation internally zero-pads to the next power of two for the transform only. The public maximum remains the real supplied dimension.

For example:

```text
4095 source dimensions
        ↓
internal zero-pad to 4096
        ↓
SRHT-v1
        ↓
2048 output dimensions
```

A request for `8192` from those 4095 supplied dimensions still fails.

## Determinism

`SRHT-v1` is a persistence protocol, not an implementation-detail random projection.

Its sign choices and coordinate ordering are derived from SHA-256 domain-separated values defined by the profile and source dimensionality. They do not use `System.Random` or runtime-dependent seeded random behavior.

Changing those rules requires a new profile such as `SRHT-v2`.

## Reduced spaces are new embedding spaces

Dimension reduction changes the coordinate system. The output identity therefore receives a deterministic child fingerprint containing:

```text
base embedding-space fingerprint
SRHT-v1
source dimensions
output dimensions
```

Two 512-dimensional vectors are not automatically compatible merely because their lengths match.

These are different spaces:

```text
2048 -> SRHT-v1 -> 512
2048 -> future SRHT-v2 -> 512
2048 -> model-native Matryoshka -> 512
```

Numeric format does not change the embedding-space fingerprint; the coordinate transform does.

## Information loss

Reducing dimensions is mathematically lossy. A request such as `2048 -> 1024` may retain much more useful geometry than `2048 -> 64`, but both are permitted when mathematically valid.

The API guarantees deterministic compatible transforms, not equivalent retrieval quality at every target dimension.
