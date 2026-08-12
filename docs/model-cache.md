# Model cache and updates

The cache is snapshot-based rather than a mutable directory of half-downloaded files.

## Activation sequence

1. Resolve a candidate revision.
2. Acquire the cache lock.
3. Download into staging using streamed HTTP I/O.
4. Validate paths, expected lengths, and provided hashes.
5. Resolve model/tokenizer runtime files and compute the embedding-space fingerprint.
6. Load the candidate tokenizer and ONNX worker pool.
7. Promote the candidate snapshot atomically.
8. Swap the active runtime.
9. Dispose previous ONNX sessions/tokenizer.
10. Delete old snapshots.

A partial candidate is never the active snapshot.

## Failure behavior

If no working runtime exists, initialization failures surface to the caller and status becomes `Faulted`. If an update fails while a valid runtime is already serving requests, the existing runtime remains active and status records the update error.

## Hot swap

```csharp
bool changed = await embeddingService.UpdateModelAsync();
```

New requests use the new worker pool after the swap. Work already running may complete against the previous pool. Old sessions are disposed before physical snapshot deletion, preventing common Windows locked-file failures.

## Embedding-space changes

A changed fingerprint means application-persisted vectors must be regenerated before they can be compared with queries from the new space. Search throws `EmbeddingSpaceMismatchException` rather than returning invalid rankings.

## Multiple processes

Cache operations use a cross-process lock and retry delay. This prevents two application instances starting at the same time from racing to activate a partially written snapshot.
