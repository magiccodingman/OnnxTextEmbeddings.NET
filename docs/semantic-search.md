# Semantic search

The practical mental model is simple:

> Search uses an item's strongest semantic evidence. Irrelevant chunks do not drag the item downward. Nearby strong evidence may add a modest confidence boost.

## One-field search

```csharp
var results = await semanticSearch.SearchAsync(
    query,
    documents,
    document => document.Embeddings,
    new SemanticSearchRequest { Top = 10 });
```

`query` can be either a string or a precomputed `QueryEmbedding`. Precomputing is useful when the same query is applied to several candidate scopes because ONNX inference happens only once.

## Chunked documents

Every candidate `TextEmbedding` is scored separately. The best adjusted chunk establishes the field score. Only the next two strongest chunks can add bounded support; twenty mediocre chunks cannot win by sheer repetition.

By default each result returns the top three chunk matches. Set `IncludeAllChunkMatches = true` for diagnostics.

## Weighted fields

```csharp
var results = await semanticSearch.SearchFieldsAsync(
    query,
    pages,
    page =>
    [
        SemanticField.Create("title", page.TitleEmbeddings, 1.5f),
        SemanticField.Create("tags", page.TagEmbeddings, 1.2f),
        SemanticField.Create("body", page.BodyEmbeddings, 1.0f)
    ]);
```

A weight of zero disables the field. Negative weights are rejected. Weights transform field confidence rather than linearly multiplying raw cosine, keeping final scores in a stable 0..1 range.

## Result diagnostics

`SemanticSearchResult<T>` exposes:

- `Score` — final item score
- `BestMatch` — strongest chunk across fields
- `Fields` — field scores and weighted scores
- `Scoring` — scoring profile ID/version

Each `SemanticChunkMatch` exposes raw cosine, length confidence, adjusted similarity, and the original `TextEmbedding` record.

## Compatibility safety

Query and document fingerprints must match. A dimension match alone is not enough: comparing vectors from different embedding spaces is mathematically valid-looking but semantically meaningless, so the library throws `EmbeddingSpaceMismatchException`.
