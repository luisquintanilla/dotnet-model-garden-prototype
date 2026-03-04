# DotnetAILab.ModelGarden.Embeddings.GteSmall

> Local, on-device sentence embeddings using the gte-small model — a strong general-purpose text embedding model by Alibaba DAMO Academy — via the Microsoft.Extensions.AI `IEmbeddingGenerator` interface.

## Overview

**Text embeddings** convert human-readable text into fixed-length vectors (arrays of numbers). Once text is represented as a vector, you can mathematically compare how similar two pieces of text are using **cosine similarity** — a value between −1 and 1 where 1 means identical meaning, 0 means unrelated, and −1 means opposite.

```
"Renewable energy is growing fast"  →  [0.015, -0.037, 0.062, ..., 0.028]  (384 numbers)
"Solar and wind power are booming"  →  [0.013, -0.034, 0.059, ..., 0.031]  (384 numbers)
                                                   cosine similarity ≈ 0.89 (very similar!)
```

**GTE (General Text Embeddings) small** is an embedding model developed by Alibaba DAMO Academy. It was trained on a large-scale corpus using a multi-stage contrastive learning approach and performs strongly across a wide range of text embedding benchmarks — often matching or outperforming larger models. GTE is a versatile all-rounder: it handles semantic search, clustering, classification, and similarity tasks with consistently strong results.

This package downloads the ONNX model automatically on first use, caches it locally, and runs inference entirely on your machine with no cloud calls or API keys required.

## Model Details

| Property | Value |
|---|---|
| **Model ID** | `thenlper/gte-small` |
| **Architecture** | BERT-based (small) |
| **Embedding Dimensions** | 384 |
| **Pooling Strategy** | Mean pooling |
| **Normalization** | L2-normalized |
| **Model Size** | ~127 MB (ONNX) |
| **Max Tokens** | 512 |
| **License** | MIT |
| **Source** | [Hugging Face](https://huggingface.co/thenlper/gte-small) |

## Installation

```shell
dotnet add package DotnetAILab.ModelGarden.Embeddings.GteSmall
```

> **Note:** The ONNX model binary (~127 MB) is automatically downloaded from Hugging Face on first use and cached locally. No manual download is required.

### Dependencies

This package depends on:

- `ModelPackages` (0.1.0-preview.14) — Model download, caching, and verification
- `MLNet.TextInference.Onnx` (0.1.0-preview.1) — ONNX Runtime–based text inference
- `Microsoft.Extensions.AI` — The `IEmbeddingGenerator` abstraction

## Quick Start

### Generate Embeddings

```csharp
using DotnetAILab.ModelGarden.Embeddings.GteSmall;

// Create the embedding generator (downloads model on first call)
await using var generator = await GteSmallModel.CreateEmbeddingGeneratorAsync();

// Generate an embedding for a single string
var embedding = await generator.GenerateEmbeddingAsync("Renewable energy is the future of power generation.");

// The embedding vector (384 floats)
ReadOnlyMemory<float> vector = embedding.Vector;
Console.WriteLine($"Dimensions: {vector.Length}"); // 384
```

### Compute Cosine Similarity

```csharp
using DotnetAILab.ModelGarden.Embeddings.GteSmall;
using Microsoft.Extensions.AI;

await using var generator = await GteSmallModel.CreateEmbeddingGeneratorAsync();

var embeddings = await generator.GenerateAsync(new[]
{
    "Renewable energy is growing fast",
    "Solar and wind power are booming",
    "The recipe calls for two cups of flour"
});

float[] v1 = embeddings[0].Vector.ToArray();
float[] v2 = embeddings[1].Vector.ToArray();
float[] v3 = embeddings[2].Vector.ToArray();

Console.WriteLine($"Energy topics:     {CosineSimilarity(v1, v2):F4}");  // ~0.89 (similar)
Console.WriteLine($"Energy vs Cooking: {CosineSimilarity(v1, v3):F4}");  // ~0.07 (unrelated)

static float CosineSimilarity(float[] a, float[] b)
{
    float dot = 0, normA = 0, normB = 0;
    for (int i = 0; i < a.Length; i++)
    {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (MathF.Sqrt(normA) * MathF.Sqrt(normB));
}
```

> **Tip:** Since embeddings from this model are L2-normalized, the cosine similarity simplifies to just the dot product of the two vectors.

## API Reference

### `GteSmallModel.CreateEmbeddingGeneratorAsync`

Creates an `IEmbeddingGenerator<string, Embedding<float>>` backed by the local ONNX model.

```csharp
public static async Task<IEmbeddingGenerator<string, Embedding<float>>> CreateEmbeddingGeneratorAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `options` | `ModelOptions?` | Optional. Configuration for model download (cache directory, proxy, etc.). |
| `ct` | `CancellationToken` | Optional. Cancellation token for the async operation. |

**Returns:** `Task<IEmbeddingGenerator<string, Embedding<float>>>` — An embedding generator that accepts strings and produces `Embedding<float>` results.

**Behavior:**
- On first call, downloads the ONNX model from Hugging Face (~127 MB) and caches it locally.
- Subsequent calls use the cached model.
- No special prefixes are needed — just pass your raw text.
- The returned generator implements `IAsyncDisposable` — use `await using` for proper cleanup.

### `GteSmallModel.EnsureModelAsync`

Downloads and caches the model without creating a generator. Useful for pre-warming.

```csharp
public static Task<string> EnsureModelAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

**Returns:** The local file path to the cached ONNX model.

### `GteSmallModel.GetModelInfoAsync`

Returns metadata about the model (ID, size, file paths).

```csharp
public static Task<ModelInfo> GetModelInfoAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

### `GteSmallModel.VerifyModelAsync`

Verifies the integrity of the cached model file using SHA-256.

```csharp
public static Task VerifyModelAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

### The `IEmbeddingGenerator` Interface

This model implements `IEmbeddingGenerator<string, Embedding<float>>` from `Microsoft.Extensions.AI`. Key methods:

| Method | Description |
|---|---|
| `GenerateEmbeddingAsync(string)` | Generate a single embedding from a string. |
| `GenerateAsync(IEnumerable<string>)` | Generate embeddings for a batch of strings. |

This is the same standard interface used across all embedding providers in the .NET ecosystem, so you can swap models without changing your application code.

## Inputs & Outputs

### Input

- **Type:** `string` (plain text)
- **Max token length:** ~512 tokens (longer text is truncated)
- **Language:** Primarily English, with some multilingual capability

### Output

- **Type:** `Embedding<float>`
- **Dimensions:** 384
- **Normalization:** L2-normalized (unit vectors, magnitude = 1.0)
- **Vector property:** `embedding.Vector` returns `ReadOnlyMemory<float>`

### Understanding Cosine Similarity

Since the output vectors are L2-normalized, **cosine similarity equals the dot product**:

| Score | Meaning |
|---|---|
| **0.85–1.0** | Very similar / near-duplicate |
| **0.60–0.85** | Related / same topic |
| **0.30–0.60** | Loosely related |
| **0.0–0.30** | Unrelated |

> These ranges are approximate and depend on your domain. Always evaluate on your own data.

## Use Cases

- **Semantic Search** — Find documents that match a query by meaning, not just keywords.
- **Retrieval-Augmented Generation (RAG)** — Retrieve relevant context to feed into an LLM.
- **Text Clustering** — Group similar documents, articles, or support tickets automatically.
- **Duplicate Detection** — Identify near-duplicate content across a corpus.
- **Text Classification** — Use embeddings as features for downstream classifiers.
- **Recommendation Systems** — Recommend content based on semantic similarity.

## Choosing the Right Embedding Model

The .NET AI Model Garden includes four text embedding models. All produce 384-dimensional, normalized vectors and can be used interchangeably through the `IEmbeddingGenerator` interface:

| Model | Package | Best For | Pooling | Notes |
|---|---|---|---|---|
| **all-MiniLM-L6-v2** | `Embeddings.AllMiniLM` | General purpose | Mean | Most popular, great default choice |
| **bge-small-en-v1.5** | `Embeddings.BgeSmallEn` | Retrieval / RAG | CLS | Optimized for search workloads |
| **e5-small-v2** | `Embeddings.E5Small` | Query-passage matching | Mean | Use `query:` / `passage:` prefixes |
| **gte-small** | `Embeddings.GteSmall` | General text | Mean | ⭐ Strong all-rounder by Alibaba DAMO |

**When to choose GteSmall:**
- You need a **strong all-rounder** that performs well across many tasks without any special configuration.
- You don't want to deal with special prefixes (unlike E5) or worry about pooling strategy trade-offs (unlike BGE).
- You want consistently strong performance on benchmarks like MTEB (Massive Text Embedding Benchmark).
- You're comparing several models and want a reliable "second opinion" alongside AllMiniLM.

## Limitations

- **Max token length:** Input is truncated to ~512 tokens. For longer documents, consider chunking strategies.
- **English-focused:** While it has some multilingual capability, the model performs best on English text.
- **Static embeddings:** The model does not update or learn from your data at runtime.
- **CPU inference:** Runs on CPU via ONNX Runtime. For GPU acceleration, additional configuration may be needed.
- **First-run download:** The model must be downloaded on first use (~127 MB). Plan for this in air-gapped or restricted environments.

## Example: Building a Semantic Search System

```csharp
using DotnetAILab.ModelGarden.Embeddings.GteSmall;
using Microsoft.Extensions.AI;

// 1. Create the generator
await using var generator = await GteSmallModel.CreateEmbeddingGeneratorAsync();

// 2. Index your documents
string[] documents = new[]
{
    "Electric vehicles are becoming more affordable and widespread.",
    "The James Webb Space Telescope captured stunning images of distant galaxies.",
    "Quantum computing may revolutionize cryptography and drug discovery.",
    "Mediterranean diets are linked to improved heart health.",
    "Large language models can generate human-like text from prompts."
};

var docEmbeddings = await generator.GenerateAsync(documents);

// 3. Search by query
string query = "advances in AI and natural language processing";
var queryEmbedding = await generator.GenerateEmbeddingAsync(query);
float[] queryVec = queryEmbedding.Vector.ToArray();

// 4. Rank by cosine similarity
var results = documents
    .Select((doc, i) => new
    {
        Document = doc,
        Score = CosineSimilarity(queryVec, docEmbeddings[i].Vector.ToArray())
    })
    .OrderByDescending(r => r.Score)
    .ToList();

Console.WriteLine($"Query: \"{query}\"\n");
foreach (var r in results)
    Console.WriteLine($"  {r.Score:F4}  {r.Document}");

static float CosineSimilarity(float[] a, float[] b)
{
    float dot = 0;
    for (int i = 0; i < a.Length; i++) dot += a[i] * b[i];
    return dot; // Vectors are L2-normalized, so dot product = cosine similarity
}
```

## Related Models

- **[DotnetAILab.ModelGarden.Embeddings.AllMiniLM](../DotnetAILab.ModelGarden.Embeddings.AllMiniLM/)** — all-MiniLM-L6-v2, the most popular general-purpose sentence embedding model.
- **[DotnetAILab.ModelGarden.Embeddings.BgeSmallEn](../DotnetAILab.ModelGarden.Embeddings.BgeSmallEn/)** — BGE small English, optimized for retrieval and RAG workloads.
- **[DotnetAILab.ModelGarden.Embeddings.E5Small](../DotnetAILab.ModelGarden.Embeddings.E5Small/)** — E5 small v2, query-passage embedding model with prefix conventions.
- **[DotnetAILab.ModelGarden.AudioEmbedding.CLAP](../../audioembeddings/DotnetAILab.ModelGarden.AudioEmbedding.CLAP/)** — CLAP audio embeddings for audio-text similarity tasks.

## References

- [gte-small on Hugging Face](https://huggingface.co/thenlper/gte-small)
- [Towards General Text Embeddings with Multi-stage Contrastive Learning (paper)](https://arxiv.org/abs/2308.03281)
- [Alibaba DAMO Academy](https://damo.alibaba.com/)
- [Microsoft.Extensions.AI Documentation](https://learn.microsoft.com/en-us/dotnet/ai/conceptual/microsoft-extensions-ai)
