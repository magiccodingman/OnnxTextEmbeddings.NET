# In-memory search

No persistence layer is required.

```csharp
record IndexedPage(Page Page, IReadOnlyList<TextEmbedding> Embeddings);

var indexed = new List<IndexedPage>();
foreach (var page in pages)
    indexed.Add(new(page, await embeddingService.EmbedAsync(page.Markdown)));

var results = await semanticSearch.SearchAsync(
    "restore a database backup",
    indexed,
    x => x.Embeddings);
```

This is ideal for hundreds or thousands of small records, game data, local tools, tests, and applications that already keep their working set in memory.

`ISemanticSearch` keeps only the requested Top-K item results in its priority queue. Chunk diagnostics are returned only for selected results.
