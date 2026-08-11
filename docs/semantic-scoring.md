# DefaultV1 semantic scoring

DefaultV1 is intentionally explicit and versioned. It is not a hidden heuristic.

## 1. Raw similarity

For query vector `q` and chunk vector `c`:

```text
raw = cosine(q, c)
```

INT4, INT8, FP16, and FP32 vectors can be compared in mixed precision.

## 2. Length confidence

A tiny remainder chunk is useful evidence, but an equally similar near-full chunk generally contains more semantic evidence.

```text
coverage = clamp(chunkTokenCount / historicalTokenCapacity, 0, 1)
confidence = minConfidence + (1 - minConfidence) * sqrt(coverage)
adjusted = max(0, raw) * confidence
```

Default `minConfidence = 0.96`, so this is deliberately gentle. Length never rescues an unrelated chunk; it only provides a small confidence distinction between similarly relevant chunks.

## 3. Supporting evidence

Sort adjusted chunk scores descending. Let `best` be the strongest score. Only two additional scores may contribute:

```text
strength(best, support) = clamp(1 - ((best - support) / supportWindow), 0, 1)
bonus = (1 - best) * weight * strength * support
```

Defaults:

```text
supportWindow       = 0.12
secondSupportWeight = 0.25
thirdSupportWeight  = 0.10
```

The field score is:

```text
field = min(1, best + secondBonus + thirdBonus)
```

A weak support score outside the window contributes nothing. Repeating the same mediocre match 50 times cannot accumulate unlimited evidence.

## 4. Field weights

A semantic field weight `w` transforms its field score `s` as:

```text
weighted = 1 - (1 - clamp(s, 0, 1))^w
```

This keeps weighted confidence in 0..1 while allowing title/tags/body emphasis.

## 5. Item score

Weighted field scores are ordered strongest-first and passed through the same bounded evidence aggregator. The result remains in 0..1.

`SemanticScoringInfo("DefaultV1", 1)` is included in each result so persisted diagnostics can identify the exact scoring semantics used.
