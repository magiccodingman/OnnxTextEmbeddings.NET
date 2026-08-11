# Why Jasper Token Compression 600M is the default model

OnnxTextEmbeddings.NET defaults to **Jasper Token Compression 600M** because its architecture and measured CPU behavior line up unusually well with the goals of this project: strong local embeddings, practical CPU inference, long document chunks, small deployment footprints, and no separate embedding server.

This document separates two kinds of evidence:

- **Upstream model facts** from the Jasper model card and technical report.
- **Project measurements and observations** from the ONNX exports used by OnnxTextEmbeddings.NET.

The upstream model is [`infgrad/Jasper-Token-Compression-600M`](https://huggingface.co/infgrad/Jasper-Token-Compression-600M). The technical report is [`Jasper-Token-Compression-600M Technical Report`](https://arxiv.org/abs/2511.14405).

## What Jasper is

Jasper is a 0.6B-parameter bilingual embedding model built from the Qwen3 0.6B family. Its main architectural idea is **dynamic text token compression** before the expensive transformer path.

The upstream model card describes a simple compression module after the word-embedding layer. Text shorter than a threshold is left alone; longer text is compressed to a shorter latent token sequence before transformer processing. The published model also combines vector distillation with contrastive learning.

That design is particularly attractive for CPU embedding workloads because increasing the source text length does not force the expensive transformer portion of the model to grow at the same rate as the original input.

Upstream also documents an important limitation: the model was distilled on text lengths only up to **1024 tokens**, so degradation should be expected beyond that point.

## Why it fits this project

The project is intentionally aimed at ordinary .NET applications rather than dedicated GPU inference infrastructure. A good default model therefore needs more than benchmark quality alone. It should also have:

- strong CPU throughput;
- predictable behavior on substantial document chunks;
- a small enough footprint to ship and host comfortably;
- useful embedding quality after quantization;
- good concurrency behavior in one ONNX Runtime session;
- enough embedding width for high-quality retrieval and downstream transformations.

Jasper checks those boxes unusually well.

The ONNX exports used by this project produce **2048-dimensional embeddings** and are available as Dynamic INT8, INT4, and FP32 variants.

## Benchmark environment

The project benchmarks below were measured on:

| Property | Value |
|---|---|
| CPU | AMD Ryzen 9 7950X3D |
| CPU cores / threads | 16 / 32 |
| Runtime | ONNX Runtime 1.28.0 |
| Execution provider | CPUExecutionProvider |
| Embedding dimensions | 2048 |
| Maximum benchmarked input | 1024 tokens |

The single-request benchmark deliberately used **1 intra-op thread and 1 inter-op thread** so the intrinsic efficiency of each model artifact could be compared without allowing one request to consume the entire CPU.

The concurrency benchmark used **16 intra-op threads**, one shared ONNX Runtime session, 1024 tokens per request, and between one and eight simultaneous requests.

## Deployment comparison

For most CPU deployments, **Dynamic INT8 is the recommended Jasper variant**.

| Variant | Disk size | 1024-token latency | 1024-token throughput | Speed vs FP32 | Fidelity vs FP32 | Suggested model-process RAM |
|---|---:|---:|---:|---:|---:|---:|
| **Dynamic INT8** | **583.6 MB** | **84.6 ms** | **12,101 tok/s** | **3.41×** | ~0.988–0.992 cosine | **~1.25–1.5 GB** |
| **INT4** | **317.2 MB** | **137.3 ms** | **7,457 tok/s** | **2.10×** | ~0.981–0.986 cosine | **~1.0–1.25 GB** |
| **FP32** | **~2.3 GB** | **288.4 ms** | **3,551 tok/s** | 1.00× | Reference | **~3.5–4 GB** |

Dynamic INT8 ended up being the strongest general-purpose CPU choice: much smaller than FP32, substantially faster, and still very close to the FP32 embedding space.

INT4 is valuable when download size or memory footprint matters more than maximum CPU throughput and fidelity. It is smaller than INT8, but on this CPU the INT8 execution path is actually faster.

FP32 remains useful as the quality/reference export and for validation.

## Single-request scaling with input length

The most important result for this project is not only that Jasper is fast. It is that **latency grows unusually slowly as source context gets longer**.

### Dynamic INT8

| Tokens | Latency | Throughput | Speed vs FP32 |
|---:|---:|---:|---:|
| 32 | 44.4 ms | 721 tok/s | 3.14× |
| 128 | 48.6 ms | 2,633 tok/s | 3.20× |
| 512 | 63.2 ms | 8,104 tok/s | 3.35× |
| 1024 | **84.6 ms** | **12,101 tok/s** | **3.41×** |

For 32 times as many source tokens, measured INT8 inference latency increased from roughly **44 ms to 85 ms**, not anywhere close to 32 times. That is exactly the kind of scaling a chunk-oriented CPU embedding library wants.

The relative INT8 advantage also grew with longer requests rather than collapsing:

```text
32 tokens    ~3.14× FP32
128 tokens   ~3.20×
512 tokens   ~3.35×
1024 tokens  ~3.41×
```

Separate local comparisons against the original Qwen3 0.6B embedding path also recorded roughly **3.3× performance improvements** once inputs reached the approximately 256–512-token range and beyond. Those comparison runs were part of model-selection testing rather than the standardized ONNX artifact table above, so they are treated here as an additional project observation rather than a directly interchangeable benchmark series.

This long-context compute behavior is one of the main reasons Jasper was selected.

## Quality behavior by input length

The compute curve is only half the story. A model being able to *run* a long context efficiently does not mean embedding quality stays equally strong forever.

Local project testing found the following practical pattern when comparing Jasper against the original model behavior it was trained from:

- through roughly **756 tokens**, quality held up extremely well;
- around **1024 tokens**, the observed degradation was still very small;
- **beyond 1024 tokens**, long-tail retrieval quality began to fall off much more aggressively;
- longer inputs remained usable, but the quality trade became substantially less attractive.

That observation matches the upstream training limitation: Jasper's authors state that they only distilled text up to **1024 tokens** and expect degradation beyond that length.

For that reason, OnnxTextEmbeddings.NET deliberately uses **1024 tokens as the default document-chunk and query ceiling** for Jasper.

This is a quality-oriented default, not a claim that the runtime is incapable of longer input. The API supports per-call overrides, but 1024 is the point where the project's measurements and the upstream training regime agree that the model remains in its strongest operating range.

For long documents, the preferred strategy is therefore:

```text
large document
   ↓
structure-aware chunks of <= 1024 tokens
   ↓
high-quality Jasper embeddings for each chunk
   ↓
multi-vector retrieval
```

If a downstream system explicitly requires one vector, the library's `CombineToSingle()` abstraction can combine those well-behaved chunk embeddings afterward instead of relying on one extremely long Jasper input.

## Quantization fidelity

FP32 is the reference output for these measurements.

| Variant | Median cosine similarity vs FP32 | Interpretation |
|---|---:|---|
| FP32 | 1.0 | Reference |
| **Dynamic INT8** | **~0.988–0.992** | Very close to FP32 |
| INT4 | ~0.981–0.986 | More visible quantization drift |

This is why the package defaults to downloading **Jasper INT8** even though returned embedding vectors themselves default to FP32 for compatibility.

Model-weight precision and returned-vector storage precision are independent concepts.

## Concurrency behavior

The project also tested ONNX Runtime concurrency using one shared model session rather than duplicating model weights for every simultaneous request.

The measured older INT8 architecture build produced:

| Concurrent requests | Median request latency | Total throughput | Total token throughput | Extra RSS above loaded model |
|---:|---:|---:|---:|---:|
| 1 | 35.2 ms | 28.41 emb/s | 29,097 tok/s | 46.0 MiB |
| 2 | 42.0 ms | 47.47 emb/s | 48,612 tok/s | 90.3 MiB |
| 3 | 48.8 ms | 59.46 emb/s | 60,883 tok/s | 134.4 MiB |
| 4 | 58.5 ms | 63.21 emb/s | 64,730 tok/s | 178.5 MiB |
| 5 | 64.0 ms | 71.56 emb/s | 73,274 tok/s | 229.4 MiB |
| 6 | 86.3 ms | 67.08 emb/s | 68,690 tok/s | 272.6 MiB |
| 7 | 91.0 ms | 74.68 emb/s | 76,472 tok/s | 313.2 MiB |
| 8 | 100.3 ms | **76.03 emb/s** | **77,856 tok/s** | **353.3 MiB** |

This concurrency series was measured with an older approximately 1.02 GiB INT8 artifact rather than the final 583.6 MB Dynamic INT8 export, so it should be read as evidence about **architecture/session scaling**, not as the final published Dynamic INT8 throughput number.

It established several useful properties for the library design:

- one shared ONNX Runtime session can service simultaneous requests;
- duplicate model copies are not required simply to obtain request concurrency;
- quantized execution scaled much better than FP32 in this workload;
- request workspace grew reasonably predictably with concurrency.

The measured INT8 workspace increase was roughly **40–45 MiB per simultaneous 1024-token request**, with allocator/runtime variation.

### FP32 concurrency

FP32 saturated considerably earlier:

| Concurrent requests | Median latency | Total throughput |
|---:|---:|---:|
| 1 | 88.4 ms | 11.30 emb/s |
| 2 | 107.5 ms | 17.66 emb/s |
| 3 | 137.2 ms | 21.11 emb/s |
| 4 | 161.6 ms | **23.74 emb/s** |
| 5 | 204.7 ms | 22.98 emb/s |
| 6 | 247.8 ms | 23.36 emb/s |
| 7 | 304.3 ms | 22.04 emb/s |
| 8 | 324.1 ms | 22.74 emb/s |

For FP32, concurrency above approximately four mostly absorbed queue pressure rather than adding throughput.

## Memory and deployment planning

Three memory concepts matter independently:

1. model file/download size;
2. loaded model memory;
3. active inference workspace.

Practical project budgets are:

| Variant | Suggested process budget | Comfortable rounded allocation |
|---|---:|---:|
| INT4 | ~1.0 GB | ~1.25 GB |
| Dynamic INT8 | ~1.25 GB | ~1.5 GB |
| FP32 | ~3.5 GB | ~4 GB |

These are **model-process** budgets. They do not include the operating system, database, vector index, unrelated application state, or other services.

The INT8 and INT4 recommendations are intentionally conservative where final-artifact concurrency memory has not yet been directly remeasured.

## Why INT8 is the package default

For this project's target environment, Dynamic INT8 gives the best overall balance:

- **583.6 MB** model artifact;
- **~84.6 ms** one-thread latency at 1024 tokens on the benchmark CPU;
- **~12,100 source tokens/sec** at 1024 tokens;
- approximately **3.4×** FP32 throughput at that length;
- very high correspondence to FP32 embeddings;
- strong shared-session concurrency characteristics;
- a practical roughly **1.25–1.5 GB** model-process budget;
- unusually good compute scaling as document chunks get longer.

That combination is why Jasper felt unusually close to the ideal model for OnnxTextEmbeddings.NET: it is small enough to deploy casually, fast enough to make CPU-only embedding practical, and its token-compression architecture specifically attacks the long-input cost that often makes document embedding expensive.

The main compromise is also well understood: quality becomes less trustworthy once input length moves substantially beyond the model's 1024-token training regime. The library addresses that by preferring high-quality <=1024-token chunks rather than simply pushing more context through one embedding call.

## Reproducibility notes

The benchmark values in this document describe the project's ONNX exports and the stated Ryzen 9 7950X3D / ONNX Runtime setup. They should not be treated as universal performance numbers for every CPU.

In particular:

- Dynamic INT8 single-request values refer to the final `model-int8-dynamic.onnx` artifact.
- The INT8 concurrency series currently refers to the older approximately 1.02 GiB `model-int8.onnx` artifact.
- INT4 and Dynamic INT8 deployment memory budgets include conservative planning estimates where the final artifacts have not yet been run through every isolated memory/concurrency measurement.
- quality observations around the 756/1024/post-1024 boundary come from project model-selection experiments and should be considered empirical guidance rather than an upstream formal benchmark.

For most applications, the practical recommendation remains simple: **use Jasper Dynamic INT8, keep the default 1024-token ceiling unless you have measured a reason to change it, and let the library chunk longer documents.**
