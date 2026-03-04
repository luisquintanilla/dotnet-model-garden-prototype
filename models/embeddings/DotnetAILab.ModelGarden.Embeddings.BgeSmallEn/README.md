# DotnetAILab.ModelGarden.Embeddings.BgeSmallEn

> Local, on-device sentence embeddings using the bge-small-en-v1.5 model — optimized for retrieval and RAG workloads — via the Microsoft.Extensions.AI `IEmbeddingGenerator` interface.

## Overview

**Text embeddings** convert human-readable text into fixed-length vectors (arrays of numbers). Once text is represented as a vector, you can mathematically compare how similar two pieces of text are using **cosine similarity** — a value between −1 and 1 where 1 means identical meaning, 0 means unrelated, and −1 means opposite.

```
"How to reset my password"     →  [0.023, -0.041, 0.067, ..., 0.019]  (384 numbers)
"Steps for password recovery"  →  [0.021, -0.038, 0.064, ..., 0.021]  (384 numbers)
                                                cosine similarity ≈ 0.91 (very similar!)
```

**BGE (BAAI General Embedding) small English v1.5** is an embedding model developed by the Beijing Academy of Artificial Intelligence (BAAI), specifically optimized for **retrieval tasks**. It uses **CLS token pooling** (instead of mean pooling), which tends to perform better for search and ranking scenarios. If you're building a RAG pipeline or semantic search system, BGE is an excellent choice.

This package downloads the ONNX model automatically on first use, caches it locally, and runs inference entirely on your machine with no cloud calls or API keys required.

## Model Details

| Property | Value |
|---|---|
| **Model ID** | `BAAI/bge-small-en-v1.5` |
| **Architecture** | BERT-based (small) |
| **Embedding Dimensions** | 384 |
| **Pooling Strategy** | CLS token |
| **Normalization** | L2-normalized |
| **Model Size** | ~127 MB (ONNX) |
| **Max Tokens** | 512 |
| **License** | MIT |
| **Source** | [Hugging Face](https://huggingface.co/BAAI/bge-small-en-v1.5) |

## Installation

```shell
dotnet add package DotnetAILab.ModelGarden.Embeddings.BgeSmallEn
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
using DotnetAILab.ModelGarden.Embeddings.BgeSmallEn;

// Create the embedding generator (downloads model on first call)
await using var generator = await BgeSmallEnModel.CreateEmbeddingGeneratorAsync();

// Generate an embedding for a single string
var embedding = await generator.GenerateEmbeddingAsync("How to reset my password");

// The embedding vector (384 floats)
ReadOnlyMemory<float> vector = embedding.Vector;
Console.WriteLine($"Dimensions: {vector.Length}"); // 384
```

### Compute Cosine Similarity

```csharp
using DotnetAILab.ModelGarden.Embeddings.BgeSmallEn;
using Microsoft.Extensions.AI;

await using var generator = await BgeSmallEnModel.CreateEmbeddingGeneratorAsync();

var embeddings = await generator.GenerateAsync(new[]
{
    "How to reset my password",
    "Steps for password recovery",
    "Best restaurants in Seattle"
});

float[] v1 = embeddings[0].Vector.ToArray();
float[] v2 = embeddings[1].Vector.ToArray();
float[] v3 = embeddings[2].Vector.ToArray();

Console.WriteLine($"Password queries: {CosineSimilarity(v1, v2):F4}");  // ~0.91 (similar)
Console.WriteLine($"Password vs Food: {CosineSimilarity(v1, v3):F4}");  // ~0.08 (unrelated)

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

### `BgeSmallEnModel.CreateEmbeddingGeneratorAsync`

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
- Uses **CLS token pooling**, which extracts the embedding from the `[CLS]` token rather than averaging all token embeddings. This strategy is optimized for retrieval tasks.
- The returned generator implements `IAsyncDisposable` — use `await using` for proper cleanup.

### `BgeSmallEnModel.EnsureModelAsync`

Downloads and caches the model without creating a generator. Useful for pre-warming.

```csharp
public static Task<string> EnsureModelAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

**Returns:** The local file path to the cached ONNX model.

### `BgeSmallEnModel.GetModelInfoAsync`

Returns metadata about the model (ID, size, file paths).

```csharp
public static Task<ModelInfo> GetModelInfoAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

### `BgeSmallEnModel.VerifyModelAsync`

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
- **Language:** English

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

- **Semantic Search** — Find documents that match a query by meaning, not just keywords. BGE is especially strong here.
- **Retrieval-Augmented Generation (RAG)** — Retrieve relevant context to feed into an LLM. BGE was designed with this use case in mind.
- **Question Answering** — Match user questions to relevant passages in a knowledge base.
- **Duplicate Detection** — Identify near-duplicate texts, support tickets, or articles.
- **Re-ranking** — Use similarity scores to re-rank candidate results from a first-pass search.

## Choosing the Right Embedding Model

The .NET AI Model Garden includes four text embedding models. All produce 384-dimensional, normalized vectors and can be used interchangeably through the `IEmbeddingGenerator` interface:

| Model | Package | Best For | Pooling | Notes |
|---|---|---|---|---|
| **all-MiniLM-L6-v2** | `Embeddings.AllMiniLM` | General purpose | Mean | Most popular, great default choice |
| **bge-small-en-v1.5** | `Embeddings.BgeSmallEn` | Retrieval / RAG | CLS | ⭐ Optimized for search workloads |
| **e5-small-v2** | `Embeddings.E5Small` | Query-passage matching | Mean | Use `query:` / `passage:` prefixes |
| **gte-small** | `Embeddings.GteSmall` | General text | Mean | Strong all-rounder by Alibaba DAMO |

**When to choose BgeSmallEn:**
- Your primary use case is **search or retrieval** (finding the most relevant documents for a query).
- You're building a **RAG pipeline** and need high-quality passage retrieval.
- You want a model that uses **CLS pooling**, which some benchmarks show performs better for retrieval tasks.

## Limitations

- **English only:** This model is trained specifically on English text and will not perform well on other languages.
- **Max token length:** Input is truncated to ~512 tokens. For longer documents, consider chunking strategies.
- **Static embeddings:** The model does not update or learn from your data at runtime.
- **CPU inference:** Runs on CPU via ONNX Runtime. For GPU acceleration, additional configuration may be needed.
- **First-run download:** The model must be downloaded on first use (~127 MB). Plan for this in air-gapped or restricted environments.
- **CLS pooling trade-off:** CLS pooling excels at retrieval but may slightly underperform mean pooling on some clustering or STS (Semantic Textual Similarity) benchmarks.

## Example: Building a Semantic Search System

```csharp
using DotnetAILab.ModelGarden.Embeddings.BgeSmallEn;
using Microsoft.Extensions.AI;

// 1. Create the generator
await using var generator = await BgeSmallEnModel.CreateEmbeddingGeneratorAsync();

// 2. Index your knowledge base
string[] knowledgeBase = new[]
{
    "To reset your password, go to Settings > Security > Change Password.",
    "Our return policy allows returns within 30 days of purchase.",
    "Free shipping is available on orders over $50.",
    "Contact customer support at support@example.com for billing issues.",
    "Two-factor authentication can be enabled in your account settings."
};

var docEmbeddings = await generator.GenerateAsync(knowledgeBase);

// 3. Search by user query
string userQuery = "I forgot my password, how do I change it?";
var queryEmbedding = await generator.GenerateEmbeddingAsync(userQuery);
float[] queryVec = queryEmbedding.Vector.ToArray();

// 4. Rank by cosine similarity
var results = knowledgeBase
    .Select((doc, i) => new
    {
        Document = doc,
        Score = CosineSimilarity(queryVec, docEmbeddings[i].Vector.ToArray())
    })
    .OrderByDescending(r => r.Score)
    .ToList();

Console.WriteLine($"Query: \"{userQuery}\"\n");
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
- **[DotnetAILab.ModelGarden.Embeddings.E5Small](../DotnetAILab.ModelGarden.Embeddings.E5Small/)** — E5 small v2, query-passage embedding model with prefix conventions.
- **[DotnetAILab.ModelGarden.Embeddings.GteSmall](../DotnetAILab.ModelGarden.Embeddings.GteSmall/)** — GTE small, strong general-purpose embeddings by Alibaba DAMO.
- **[DotnetAILab.ModelGarden.AudioEmbedding.CLAP](../../audioembeddings/DotnetAILab.ModelGarden.AudioEmbedding.CLAP/)** — CLAP audio embeddings for audio-text similarity tasks.

## References

- [bge-small-en-v1.5 on Hugging Face](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [C-Pack: Packaged Resources To Advance General Chinese Embedding (paper)](https://arxiv.org/abs/2309.07597)
- [BAAI FlagEmbedding GitHub](https://github.com/FlagOpen/FlagEmbedding)
- [Microsoft.Extensions.AI Documentation](https://learn.microsoft.com/en-us/dotnet/ai/conceptual/microsoft-extensions-ai)
