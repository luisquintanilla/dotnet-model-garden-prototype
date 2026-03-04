# DotnetAILab.ModelGarden.Embeddings.AllMiniLM

> Local, on-device sentence embeddings using the all-MiniLM-L6-v2 model — the most popular general-purpose text embedding model — via the Microsoft.Extensions.AI `IEmbeddingGenerator` interface.

## Overview

**Text embeddings** convert human-readable text into fixed-length vectors (arrays of numbers). Once text is represented as a vector, you can mathematically compare how similar two pieces of text are using **cosine similarity** — a value between −1 and 1 where 1 means identical meaning, 0 means unrelated, and −1 means opposite.

```
"The cat sat on the mat"  →  [0.012, -0.034, 0.056, ..., 0.078]  (384 numbers)
"A kitten rested on a rug" →  [0.011, -0.031, 0.059, ..., 0.075]  (384 numbers)
                                              cosine similarity ≈ 0.92 (very similar!)
```

**all-MiniLM-L6-v2** is the most widely used sentence embedding model in the open-source ecosystem. It is a distilled version of Microsoft's MiniLM architecture, fine-tuned on over 1 billion sentence pairs. Despite its small size (~86 MB), it provides strong performance across a wide range of tasks — semantic search, clustering, duplicate detection, and more.

This package downloads the ONNX model automatically on first use, caches it locally, and runs inference entirely on your machine with no cloud calls or API keys required.

## Model Details

| Property | Value |
|---|---|
| **Model ID** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Architecture** | MiniLM (6 layers, 384 hidden) |
| **Embedding Dimensions** | 384 |
| **Pooling Strategy** | Mean pooling |
| **Normalization** | L2-normalized |
| **Model Size** | ~86 MB (ONNX) |
| **Max Tokens** | 256 |
| **License** | Apache 2.0 |
| **Source** | [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |

## Installation

```shell
dotnet add package DotnetAILab.ModelGarden.Embeddings.AllMiniLM
```

> **Note:** The ONNX model binary (~86 MB) is automatically downloaded from Hugging Face on first use and cached locally. No manual download is required.

### Dependencies

This package depends on:

- `ModelPackages` (0.1.0-preview.14) — Model download, caching, and verification
- `MLNet.TextInference.Onnx` (0.1.0-preview.1) — ONNX Runtime–based text inference
- `Microsoft.Extensions.AI` — The `IEmbeddingGenerator` abstraction

## Quick Start

### Generate Embeddings

```csharp
using DotnetAILab.ModelGarden.Embeddings.AllMiniLM;

// Create the embedding generator (downloads model on first call)
await using var generator = await AllMiniLMModel.CreateEmbeddingGeneratorAsync();

// Generate an embedding for a single string
var embedding = await generator.GenerateEmbeddingAsync("The weather is lovely today.");

// The embedding vector (384 floats)
ReadOnlyMemory<float> vector = embedding.Vector;
Console.WriteLine($"Dimensions: {vector.Length}"); // 384
```

### Compute Cosine Similarity

```csharp
using DotnetAILab.ModelGarden.Embeddings.AllMiniLM;
using Microsoft.Extensions.AI;

await using var generator = await AllMiniLMModel.CreateEmbeddingGeneratorAsync();

var embeddings = await generator.GenerateAsync(new[]
{
    "The cat sat on the mat",
    "A kitten rested on a rug",
    "The stock market rose sharply"
});

float[] v1 = embeddings[0].Vector.ToArray();
float[] v2 = embeddings[1].Vector.ToArray();
float[] v3 = embeddings[2].Vector.ToArray();

Console.WriteLine($"Cat vs Kitten:  {CosineSimilarity(v1, v2):F4}");  // ~0.85 (similar)
Console.WriteLine($"Cat vs Stocks:  {CosineSimilarity(v1, v3):F4}");  // ~0.10 (unrelated)

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

### `AllMiniLMModel.CreateEmbeddingGeneratorAsync`

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
- On first call, downloads the ONNX model from Hugging Face (~86 MB) and caches it locally.
- Subsequent calls use the cached model.
- The returned generator implements `IAsyncDisposable` — use `await using` for proper cleanup.

### `AllMiniLMModel.EnsureModelAsync`

Downloads and caches the model without creating a generator. Useful for pre-warming.

```csharp
public static Task<string> EnsureModelAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

**Returns:** The local file path to the cached ONNX model.

### `AllMiniLMModel.GetModelInfoAsync`

Returns metadata about the model (ID, size, file paths).

```csharp
public static Task<ModelInfo> GetModelInfoAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

### `AllMiniLMModel.VerifyModelAsync`

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
- **Max token length:** ~256 tokens (longer text is truncated)
- **Language:** Multilingual, but best performance on English

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
- **Duplicate Detection** — Identify near-duplicate texts, support tickets, or articles.
- **Clustering** — Group similar documents together automatically.
- **Recommendation Systems** — Recommend content based on semantic similarity.
- **Text Classification** — Use embeddings as features for downstream classifiers.

## Choosing the Right Embedding Model

The .NET AI Model Garden includes four text embedding models. All produce 384-dimensional, normalized vectors and can be used interchangeably through the `IEmbeddingGenerator` interface:

| Model | Package | Best For | Pooling | Notes |
|---|---|---|---|---|
| **all-MiniLM-L6-v2** | `Embeddings.AllMiniLM` | General purpose | Mean | ⭐ Most popular, great default choice |
| **bge-small-en-v1.5** | `Embeddings.BgeSmallEn` | Retrieval / RAG | CLS | Optimized for search workloads |
| **e5-small-v2** | `Embeddings.E5Small` | Query-passage matching | Mean | Use `query:` / `passage:` prefixes |
| **gte-small** | `Embeddings.GteSmall` | General text | Mean | Strong all-rounder by Alibaba DAMO |

**When to choose AllMiniLM:**
- You need a battle-tested, widely-used default.
- Your use case spans multiple tasks (search, clustering, classification).
- You want the largest community and most benchmarks to compare against.

## Limitations

- **Max token length:** Input is truncated to ~256 tokens. For longer documents, consider chunking strategies.
- **English-optimized:** While multilingual, the model performs best on English text.
- **Static embeddings:** The model does not update or learn from your data at runtime.
- **CPU inference:** Runs on CPU via ONNX Runtime. For GPU acceleration, additional configuration may be needed.
- **First-run download:** The model must be downloaded on first use (~86 MB). Plan for this in air-gapped or restricted environments.

## Example: Building a Semantic Search System

```csharp
using DotnetAILab.ModelGarden.Embeddings.AllMiniLM;
using Microsoft.Extensions.AI;

// 1. Create the generator
await using var generator = await AllMiniLMModel.CreateEmbeddingGeneratorAsync();

// 2. Index your documents
string[] documents = new[]
{
    "Python is a popular programming language",
    "The Eiffel Tower is in Paris, France",
    "Machine learning models require training data",
    "The Great Wall of China is visible from space",
    "Neural networks are inspired by biological brains"
};

var docEmbeddings = await generator.GenerateAsync(documents);

// 3. Search by query
string query = "deep learning and AI";
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

- **[DotnetAILab.ModelGarden.Embeddings.BgeSmallEn](../DotnetAILab.ModelGarden.Embeddings.BgeSmallEn/)** — BGE small English, optimized for retrieval and RAG workloads.
- **[DotnetAILab.ModelGarden.Embeddings.E5Small](../DotnetAILab.ModelGarden.Embeddings.E5Small/)** — E5 small v2, query-passage embedding model with prefix conventions.
- **[DotnetAILab.ModelGarden.Embeddings.GteSmall](../DotnetAILab.ModelGarden.Embeddings.GteSmall/)** — GTE small, strong general-purpose embeddings by Alibaba DAMO.
- **[DotnetAILab.ModelGarden.AudioEmbedding.CLAP](../../audioembeddings/DotnetAILab.ModelGarden.AudioEmbedding.CLAP/)** — CLAP audio embeddings for audio-text similarity tasks.

## Versioning

This package's NuGet version tracks the upstream model version. The current version **2.0.0** corresponds to **[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)**, where "v2" refers to the model architecture version from sentence-transformers.

| NuGet Version | Upstream Model |
|---------------|---------------|
| 2.0.x | all-MiniLM-L6-**v2** |

When the upstream model releases a new version, a new major version of this package will be published with an updated model binary.

## References

- [all-MiniLM-L6-v2 on Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Sentence-Transformers Documentation](https://www.sbert.net/)
- [Microsoft.Extensions.AI Documentation](https://learn.microsoft.com/en-us/dotnet/ai/conceptual/microsoft-extensions-ai)
- [MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers (paper)](https://arxiv.org/abs/2002.10957)
