# Architecture

The project now composes two reusable infrastructure packages instead of reimplementing their generic machinery.

```text
ModelArtifacts.NET
  source resolution / download / staging / cache / verification / promotion
              │
              ▼
OnnxTextEmbeddings.NET embedding policy
  Jasper/custom model selection
  required files + embedding-space identity
  tokenizer + chunking + tensor contract + pooling/normalization
              │
              ▼
OnnxModelRuntime.NET
  InferenceSession hosting / bounded queue / concurrency / scheduling / recovery
              │
              ▼
versioned TextEmbedding / QueryEmbedding
              │
       ┌──────┴──────┐
       │             │
 semantic search   lexical search
       │             │
       └──────┬──────┘
              ▼
        optional RRF hybrid
```

The ownership rule is intentionally simple:

- **ModelArtifacts.NET owns files.**
- **OnnxModelRuntime.NET owns generic ONNX execution orchestration.**
- **OnnxTextEmbeddings.NET owns what an embedding model means and how search results mean something.**
- **Database adapters own provider-native retrieval mechanics.**

## Model acquisition ownership

`ModelArtifacts.NET` owns revision resolution, explicit artifact selection, download retries, transfer integrity, staging, cross-process cache locking, candidate snapshots, offline fallback, atomic promotion/discard, and cleanup.

OnnxTextEmbeddings.NET still owns embedding-specific policy: Jasper presets, which model/tokenizer file is required, interpretation of `onnx-text-embeddings.json`, token limits, tokenizer construction, model/tensor validation, and embedding-space identity.

A newly downloaded candidate is not promoted merely because its bytes verified. OnnxTextEmbeddings creates the tokenizer and `OnnxModelRuntime`, runs a real validation inference through the embedding tensor/pooling path, and only then calls `ArtifactManager.PromoteAsync`. A failed candidate can therefore be discarded while the previous known-good snapshot stays current.

`ArtifactFingerprint` and `EmbeddingSpaceFingerprint` remain distinct concepts. The former is generic artifact identity; the latter is the compatibility contract for persisted vectors. The embedding package preserves its historical fingerprint calculation so adopting ModelArtifacts.NET does not silently invalidate existing vectors.

## Tokenization and request preparation

The tokenizer is created once per active model runtime. Source tokenization records offsets so chunks map directly back to original UTF-16 text. Application-wide document/query token limits can be overridden per call.

## ONNX runtime ownership

OnnxTextEmbeddings provides a small `EmbeddingOnnxExecutor : IOnnxModelExecutor<TokenizedModelInput,float[]>`. It owns only embedding-specific tensor behavior:

```text
input_ids / attention_mask / token_type_ids
output shape validation
mean pooling when required
L2 normalization
embedding dimension observation
```

`OnnxModelRuntime.NET` owns the generic machinery around that adapter: `InferenceSession` creation/disposal, one bounded global queue, per-instance concurrency, least-loaded scheduling, failure classification, draining/recovery, one-time recoverable retries, memory-pressure behavior, and runtime diagnostics.

Existing public inference option names remain stable and map directly into `OnnxModelRuntimeOptions`.

## Single-embedding post-processing

`CombineToSingle` remains entirely embedding-specific and runs after inference:

```text
TextEmbedding[]
   ↓
validate common embedding space / dimensions
   ↓
SemanticCoverage-v1
   ↓
optional deterministic SRHT-v1 dimension reduction
   ↓
optional FP32 / FP16 / INT8 / INT4 conversion
   ↓
SingleEmbedding
```

Semantic aggregation, coordinate-space reduction, and numeric/storage conversion remain independent because they lose different kinds of information.

## Search architecture

Core exposes three useful levels:

1. `ISemanticSearch` — existing semantic-only API and canonical `DefaultV1`.
2. `ILexicalSearch` — in-memory `BM25-v1` without requiring a model.
3. `IAdvancedSearch` / `SearchQuery` — composable global filters, stage filters, semantic/lexical retrieval stages, field weights, candidate budgets, RRF fusion, and optional post-filtering.

A `SearchQuery` is deliberately a retrieval plan rather than a `Semantic/Lexical/Hybrid` enum so future retriever kinds can be added without redesigning hybrid search.

## Database-native search

Database adapters push portable prefilters and retrieval work into their engines:

```text
SearchQuery
   ↓
global + stage portable filters
   ↓
provider-native semantic and/or lexical retrieval
   ↓
semantic candidates → core DefaultV1
lexical candidates  → provider-native rank
   ↓
core Reciprocal Rank Fusion
   ↓
optional post-filter
```

Official adapters:

```text
OnnxTextEmbeddings.NET.PgVector
  pgvector + PostgreSQL full text

OnnxTextEmbeddings.NET.SqliteVec
  sqlite-vec + FTS5 BM25

OnnxTextEmbeddings.NET.SqlServer
  SQL Server/Azure SQL VECTOR + Full-Text Search
```

Each adapter accepts portable filter ASTs through logical-to-physical column mappings while retaining `AdditionalWhereSql` and provider-native command configuration for intentional backend-specific predicates.

Raw lexical scores are never blended directly with semantic scores. Hybrid search uses rank positions through RRF because FTS5 BM25, PostgreSQL rank, SQL Server `RANK`, and semantic scores do not share a meaningful numeric scale.

## Stable direct records

The vector encoding and embedding record schema are explicitly versioned. Quantized vector bytes are self-describing. Document records preserve the model-space fingerprint and historical token capacity actually used when each chunk was created.

A direct `TextEmbedding` may carry deterministic `DimensionReduction` metadata. Reducing a direct chunk changes its vector coordinate space but does not make it an aggregate.

## Native AOT boundary

Core remains Native AOT compatible. The separate non-NuGet `OnnxTextEmbeddings.Native` facade publishes the managed implementation as a Native AOT shared library with a stable C ABI.

The native embedding facade continues to expose an embedding-specific ABI. Its generic model hosting underneath is now supplied by the Native-AOT-compatible `OnnxModelRuntime.NET` managed dependency rather than a duplicated worker-pool implementation.

## Storage boundary

Core has no persistence dependency. Portable embedding records can be stored in any application store. Native vector/full-text dependencies remain isolated in the optional database provider packages.
