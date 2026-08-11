# Chunking

Document chunking is structure-first:

```text
whole document
      ↓
Markdown sections
      ↓
paragraph boundaries
      ↓
sentence boundaries
      ↓
word boundaries
      ↓
token window fallback
```

A document that already fits the configured model-input ceiling remains one chunk.

## Markdown

ATX headings (`#` through `######`) create sections when they appear outside fenced code blocks. A split continuation can prepend a synthetic heading path such as:

```text
Operations > Backups

<original source text>
```

That synthetic context is fed to the model but is not claimed as original source. `TextEmbedding.Context`, `HeadingPath`, `ContextTokenCount`, and exact source ranges keep the distinction explicit.

## Source ranges

The tokenizer is run with offsets so every finalized chunk stores:

- UTF-16 character start/length
- token start/length
- source token count
- historical source-token capacity
- document token count

Those ranges point into the original input string, not a normalized copy.

## Overlap

`ChunkOverlapTokens` defaults to `0`. When enabled, continuation chunks borrow up to that many prior source tokens while reserving input capacity for the overlap. Overlap is scoped to the same Markdown section/heading context. This avoids pulling text from a prior heading into a new heading's semantic context.

## Token accounting

The final model input—including tokenizer special tokens and repeated heading context—must fit `DocumentChunkMaxTokens` and the model's hard maximum. Finalization validates this invariant and fails explicitly instead of silently truncating.
