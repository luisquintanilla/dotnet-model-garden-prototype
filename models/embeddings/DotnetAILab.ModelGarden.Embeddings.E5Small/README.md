# DotnetAILab.ModelGarden.Embeddings.E5Small

> Local, on-device sentence embeddings using the e5-small-v2 model — a query-passage embedding model that benefits from task-specific prefixes — via the Microsoft.Extensions.AI `IEmbeddingGenerator` interface.

## Overview

**Text embeddings** convert human-readable text into fixed-length vectors (arrays of numbers). Once text is represented as a vector, you can mathematically compare how similar two pieces of text are using **cosine similarity** — a value between −1 and 1 where 1 means identical meaning, 0 means unrelated, and −1 means opposite.

```
"query: What is the capital of France?"   →  [0.018, -0.029, 0.071, ..., 0.044]  (384 numbers)
"passage: Paris is the capital of France" →  [0.016, -0.027, 0.068, ..., 0.041]  (384 numbers)
                                                         cosine similarity ≈ 0.93 (very similar!)
```

**E5 (EmbEddings from bidirEctional Encoder rEpresentations) small v2** is a text embedding model from Microsoft Research that was trained with a novel approach: contrastive learning on both labeled and unlabeled data. Its key differentiator is the **prefix convention** — prepending `"query: "` to search queries and `"passage: "` to documents/passages improves retrieval performance significantly. This makes E5 particularly well-suited for asymmetric search scenarios where queries and documents have different characteristics.

This package downloads the ONNX model automatically on first use, caches it locally, and runs inference entirely on your machine with no cloud calls or API keys required.

## Model Details

| Property | Value |
|---|---|
| **Model ID** | `intfloat/e5-small-v2` |
| **Architecture** | BERT-based (small) |
| **Embedding Dimensions** | 384 |
| **Pooling Strategy** | Mean pooling |
| **Normalization** | L2-normalized |
| **Model Size** | ~127 MB (ONNX) |
| **Max Tokens** | 512 |
| **License** | MIT |
| **Source** | [Hugging Face](https://huggingface.co/intfloat/e5-small-v2) |

## Installation

```shell
dotnet add package DotnetAILab.ModelGarden.Embeddings.E5Small
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
using DotnetAILab.ModelGarden.Embeddings.E5Small;

// Create the embedding generator (downloads model on first call)
await using var generator = await E5SmallModel.CreateEmbeddingGeneratorAsync();

// Generate an embedding for a query (note the "query: " prefix!)
var embedding = await generator.GenerateEmbeddingAsync("query: What is machine learning?");

// The embedding vector (384 floats)
ReadOnlyMemory<float> vector = embedding.Vector;
Console.WriteLine($"Dimensions: {vector.Length}"); // 384
```

### ⚠️ Important: Use Prefixes for Best Results

E5 models are trained with specific prefixes that significantly improve retrieval performance:

- **`"query: "`** — Prepend to search queries, questions, or short lookup strings.
- **`"passage: "`** — Prepend to documents, paragraphs, or passages being indexed.

```csharp
// For search queries
var queryEmb = await generator.GenerateEmbeddingAsync("query: How does photosynthesis work?");

// For documents/passages being indexed
var passageEmb = await generator.GenerateEmbeddingAsync("passage: Photosynthesis is the process by which plants convert sunlight into energy.");
```

> The model works without prefixes too, but you'll get noticeably better retrieval accuracy with them.

### Compute Cosine Similarity

```csharp
using DotnetAILab.ModelGarden.Embeddings.E5Small;
using Microsoft.Extensions.AI;

await using var generator = await E5SmallModel.CreateEmbeddingGeneratorAsync();

var embeddings = await generator.GenerateAsync(new[]
{
    "query: What causes rain?",
    "passage: Rain is caused by water evaporating, condensing into clouds, and falling as precipitation.",
    "passage: The stock market experienced a sharp decline today."
});

float[] query = embeddings[0].Vector.ToArray();
float[] relevant = embeddings[1].Vector.ToArray();
float[] unrelated = embeddings[2].Vector.ToArray();

Console.WriteLine($"Query vs Relevant:  {CosineSimilarity(query, relevant):F4}");  // ~0.90 (match!)
Console.WriteLine($"Query vs Unrelated: {CosineSimilarity(query, unrelated):F4}"); // ~0.10 (no match)

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

### `E5SmallModel.CreateEmbeddingGeneratorAsync`

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
- Remember to prepend `"query: "` or `"passage: "` to your inputs for best results.
- The returned generator implements `IAsyncDisposable` — use `await using` for proper cleanup.

### `E5SmallModel.EnsureModelAsync`

Downloads and caches the model without creating a generator. Useful for pre-warming.

```csharp
public static Task<string> EnsureModelAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

**Returns:** The local file path to the cached ONNX model.

### `E5SmallModel.GetModelInfoAsync`

Returns metadata about the model (ID, size, file paths).

```csharp
public static Task<ModelInfo> GetModelInfoAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

### `E5SmallModel.VerifyModelAsync`

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

- **Type:** `string` (plain text, optionally with `"query: "` or `"passage: "` prefix)
- **Max token length:** ~512 tokens (longer text is truncated)
- **Language:** English (primarily), with some multilingual capability
- **Prefix convention:**
  - `"query: "` for search queries, questions
  - `"passage: "` for documents, paragraphs, knowledge base entries

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

- **Asymmetric Semantic Search** — Short queries matched against longer document passages. E5's prefix convention is designed for exactly this.
- **Retrieval-Augmented Generation (RAG)** — Retrieve relevant context to feed into an LLM with high precision.
- **Question Answering** — Match user questions (`query:`) to knowledge base entries (`passage:`).
- **Document Retrieval** — Index a large corpus and find the most relevant documents for any query.
- **Semantic Similarity** — Compare text pairs for similarity (use the same prefix for symmetric comparisons).

## Choosing the Right Embedding Model

The .NET AI Model Garden includes four text embedding models. All produce 384-dimensional, normalized vectors and can be used interchangeably through the `IEmbeddingGenerator` interface:

| Model | Package | Best For | Pooling | Notes |
|---|---|---|---|---|
| **all-MiniLM-L6-v2** | `Embeddings.AllMiniLM` | General purpose | Mean | Most popular, great default choice |
| **bge-small-en-v1.5** | `Embeddings.BgeSmallEn` | Retrieval / RAG | CLS | Optimized for search workloads |
| **e5-small-v2** | `Embeddings.E5Small` | Query-passage matching | Mean | ⭐ Use `query:` / `passage:` prefixes |
| **gte-small** | `Embeddings.GteSmall` | General text | Mean | Strong all-rounder by Alibaba DAMO |

**When to choose E5Small:**
- Your use case involves **asymmetric search** — short queries matched against longer passages.
- You can easily add **`"query: "` / `"passage: "` prefixes** to your inputs (the key to getting the best performance from E5).
- You want a model from **Microsoft Research** with strong academic backing and benchmarks.

## Limitations

- **Prefix dependency:** For best results, you must prepend `"query: "` or `"passage: "` to your inputs. Without prefixes, performance may degrade compared to other models.
- **English-focused:** Primarily trained on English text. For multilingual use cases, consider other options.
- **Max token length:** Input is truncated to ~512 tokens. For longer documents, consider chunking strategies.
- **Static embeddings:** The model does not update or learn from your data at runtime.
- **CPU inference:** Runs on CPU via ONNX Runtime. For GPU acceleration, additional configuration may be needed.
- **First-run download:** The model must be downloaded on first use (~127 MB). Plan for this in air-gapped or restricted environments.

## Example: Building a Semantic Search System

```csharp
using DotnetAILab.ModelGarden.Embeddings.E5Small;
using Microsoft.Extensions.AI;

// 1. Create the generator
await using var generator = await E5SmallModel.CreateEmbeddingGeneratorAsync();

// 2. Index your knowledge base (using "passage: " prefix)
string[] rawDocuments = new[]
{
    "Python is a popular programming language created by Guido van Rossum.",
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
    "Machine learning models require training data to learn patterns.",
    "The Great Wall of China stretches over 13,000 miles.",
    "Neural networks are computing systems inspired by biological brains."
};

// Add passage prefix for indexing
var prefixedDocs = rawDocuments.Select(d => $"passage: {d}").ToArray();
var docEmbeddings = await generator.GenerateAsync(prefixedDocs);

// 3. Search by user query (using "query: " prefix)
string userQuery = "deep learning and artificial intelligence";
var queryEmbedding = await generator.GenerateEmbeddingAsync($"query: {userQuery}");
float[] queryVec = queryEmbedding.Vector.ToArray();

// 4. Rank by cosine similarity
var results = rawDocuments
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
- **[DotnetAILab.ModelGarden.Embeddings.BgeSmallEn](../DotnetAILab.ModelGarden.Embeddings.BgeSmallEn/)** — BGE small English, optimized for retrieval and RAG workloads.
- **[DotnetAILab.ModelGarden.Embeddings.GteSmall](../DotnetAILab.ModelGarden.Embeddings.GteSmall/)** — GTE small, strong general-purpose embeddings by Alibaba DAMO.
- **[DotnetAILab.ModelGarden.AudioEmbedding.CLAP](../../audioembeddings/DotnetAILab.ModelGarden.AudioEmbedding.CLAP/)** — CLAP audio embeddings for audio-text similarity tasks.

## Versioning

This package's NuGet version tracks the upstream model version. The current version **2.0.0** corresponds to **[e5-small-v2](https://huggingface.co/intfloat/e5-small-v2)**, where "v2" is the version designated by intfloat.

| NuGet Version | Upstream Model |
|---------------|---------------|
| 2.0.x | e5-small-**v2** |

When the upstream model releases a new version, a new major version of this package will be published with an updated model binary.

## References

- [e5-small-v2 on Hugging Face](https://huggingface.co/intfloat/e5-small-v2)
- [Text Embeddings by Weakly-Supervised Contrastive Pre-training (paper)](https://arxiv.org/abs/2212.03533)
- [Microsoft Research — E5 Embedding Models](https://github.com/microsoft/unilm/tree/master/e5)
- [Microsoft.Extensions.AI Documentation](https://learn.microsoft.com/en-us/dotnet/ai/conceptual/microsoft-extensions-ai)
