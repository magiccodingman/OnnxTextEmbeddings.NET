# Single-embedding aggregation

Long documents normally return several `TextEmbedding` records. Keeping those chunk embeddings is the preferred representation for retrieval because every chunk preserves its own semantic evidence.

Sometimes a downstream system requires exactly one vector. `CombineToSingle` provides a deterministic mathematical fallback for that case:

```csharp
IReadOnlyList<TextEmbedding> chunks = await embeddingService.EmbedDocumentAsync(text);
SingleEmbedding single = chunks.CombineToSingle();
```

This is **lossy semantic compression**. It is not equivalent to multi-vector retrieval and should not replace chunk-level search when chunk-level search is available.

## SemanticCoverage-v1

For two or more inputs, the default profile is `SemanticCoverage-v1`.

1. Validate common dimensions and embedding-space fingerprint.
2. Decode all source formats to FP32 working vectors.
3. L2-normalize each source vector.
4. Calculate unique original source-content mass.
5. Calculate exact pairwise cosine similarities.
6. Recenter similarity against the embedding-space neutral baseline.
7. Convert similarity to redundancy affinity.
8. Give repeated semantic regions diminishing influence.
9. Compute a weighted spherical aggregate.
10. Report coherence and minimum source similarity.
11. Optionally reduce dimensions.
12. Finally convert to the requested numeric/storage format.

The v1 affinity is:

```text
r = clamp((cosine - baseline) / (1 - baseline), 0, 1)
affinity = r^4
```

For each source embedding `i`:

```text
density[i] = mass[i] + sum(mass[j] * affinity[i,j])
weight[i] = mass[i] / sqrt(density[i])
```

The final native-dimensional aggregate is the normalized weighted sum of the normalized source vectors.

The default neutral-similarity baseline is `0` unless an explicit `NeutralSimilarityBaseline` is supplied. The actual baseline and profile constants used are persisted in `EmbeddingAggregationInfo`.

## Overlap-aware source mass

The combiner does not count repeated heading/synthetic model context as original source mass.

When normal library-generated chunks carry valid `Source.TokenRange` metadata, overlapping original tokens are apportioned symmetrically across every chunk that contains them. A token appearing in two overlapping chunks contributes `0.5` mass to each; a token appearing in three contributes `1/3` to each. The total therefore equals the unique original token union and does not depend on chunk order.

If usable common token ranges are unavailable, the fallback is each record's `Source.TokenCount`.

## Output metadata

A `SingleEmbedding` is intentionally distinct from `TextEmbedding`.

It records:

- `RepresentationKind` (`Direct` or `Aggregated`)
- source embedding count
- total original source-token count
- supplied/source dimensions
- aggregation profile/version
- source-mass method
- neutral similarity baseline
- affinity/redundancy exponents
- aggregation coherence
- minimum source similarity
- whether the numerical medoid fallback was used
- dimension-reduction profile, when applicable

This prevents downstream code from treating a many-chunk aggregate as if it were one direct chunk with ordinary chunk-capacity semantics.

## Coherence

The aggregation coherence is:

```text
coherence = ||weighted sum|| / sum(weights)
```

It lies in `0..1`. Values near one mean the source vectors point in similar semantic directions. Lower values mean one vector is a progressively weaker proxy for the complete chunk set.

The library intentionally does not label universal thresholds such as `0.8 = good` or `0.5 = bad`; appropriate thresholds depend on model and workload.

If relative coherence falls below the v1 numerical safety threshold (`1e-3`), the weighted sum is unsafe to normalize. The combiner deterministically returns the weighted semantic medoid instead and sets `FallbackUsed = true`.

## Output format

Aggregation calculations always occur in FP32. Numeric conversion happens last.

Default behavior:

```text
one input, no transform       -> preserve original vector
many inputs, same format      -> preserve common format
mixed source formats          -> FP32
explicit OutputFormat         -> caller wins
```

All existing formats are supported:

```text
FP32
FP16
INT8
INT4
```

Example:

```csharp
var single = chunks.CombineToSingle(new SingleEmbeddingOptions
{
    OutputDimensions = 512,
    OutputFormat = EmbeddingVectorFormat.Int8
});
```

That performs full-dimensional FP32 semantic aggregation first, then dimension reduction, normalization, and finally INT8 quantization.

## Single-input fast path

One input does not need semantic aggregation:

```csharp
var single = new[] { embedding }.CombineToSingle();
```

With no requested transformation, the underlying `EmbeddingVector` is returned unchanged and `RepresentationKind` is `Direct`.

Output dimension/format transformations still apply when requested.

## Validation

The combiner rejects:

- empty collections
- mismatched dimensions
- mismatched embedding-space fingerprints
- invalid vector payloads / NaN / infinity
- unusable zero-norm vectors
- requested dimensions less than one
- requested dimensions greater than the supplied dimensions
- invalid neutral-similarity baselines

Equal dimensionality alone is never considered proof of embedding-space compatibility.

## Persistence

`SingleEmbedding` supports the same JSON persistence style as the other protocol records:

```csharp
string json = EmbeddingSerializer.SerializeJson(single);
SingleEmbedding restored = EmbeddingSerializer.DeserializeSingleJson(json);
```

The contained `EmbeddingVector` can also use the existing binary vector envelope.
