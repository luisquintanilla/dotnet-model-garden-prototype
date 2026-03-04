# BGE Reranker Base

> Cross-encoder reranking model by BAAI for scoring query-document relevance — use it to reorder retrieval results in a RAG pipeline.

## Overview

**What is reranking?** In a typical search or retrieval-augmented generation (RAG) pipeline, an initial retrieval step (often using embedding similarity) returns a set of candidate documents. These candidates are fast to retrieve but may not be perfectly ordered by relevance. A **reranker** takes each (query, document) pair and produces a fine-grained relevance score, allowing you to re-sort candidates so the most relevant documents appear first.

**Cross-encoder vs. bi-encoder.** Embedding models (bi-encoders) encode queries and documents independently, then compare vectors. Cross-encoders like BGE Reranker process the query and document *together* through the full transformer, enabling deeper interaction between the two texts. This makes cross-encoders significantly more accurate for relevance scoring, but slower — which is why they are used as a second-stage reranker on a small candidate set rather than for initial retrieval over millions of documents.

**Why BGE Reranker Base?** Developed by the Beijing Academy of Artificial Intelligence (BAAI), BGE Reranker Base delivers strong relevance scoring across general-purpose retrieval benchmarks. It is a solid default choice when you need high-quality reranking and can afford a ~1.1 GB model download.

## Model Details

| Property | Value |
|---|---|
| **Model ID** | [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) |
| **Architecture** | Cross-encoder (XLM-RoBERTa based) |
| **Task** | Reranking / relevance scoring |
| **Max Token Length** | 512 tokens |
| **ONNX File Size** | ~1.11 GB |
| **Source** | Hugging Face |
| **License** | MIT |
| **NuGet Package** | `DotnetAILab.ModelGarden.Reranking.BgeReranker` |

## Installation

Add the NuGet package to your project:

```shell
dotnet add package DotnetAILab.ModelGarden.Reranking.BgeReranker
```

The ONNX model binary (~1.11 GB) is **automatically downloaded** on first use and cached locally. No manual download is required.

### Dependencies

| Package | Version |
|---|---|
| `ModelPackages` | 0.1.0-preview.14 |
| `MLNet.TextInference.Onnx` | 0.1.0-preview.1 |

## Quick Start

```csharp
using DotnetAILab.ModelGarden.Reranking.BgeReranker;
using Microsoft.ML;

// Create the reranker (downloads model on first call)
var reranker = await BgeRerankerModel.CreateRerankerAsync();

var mlContext = new MLContext();

// Prepare query-document pairs
var query = "What is the capital of France?";
var documents = new[]
{
    "Paris is the capital and most populous city of France.",
    "Berlin is the capital of Germany.",
    "France is a country in Western Europe.",
    "The Eiffel Tower is located in Paris."
};

// Score each document against the query
var pairs = documents.Select(doc => new { Query = query, Document = doc });
var dataView = mlContext.Data.LoadFromEnumerable(pairs);
var scored = reranker.Transform(dataView);

// Read relevance scores
var scores = mlContext.Data
    .CreateEnumerable<ScoredResult>(scored, reuseRowObject: false)
    .ToList();

// Sort by score descending for reranked order
var reranked = scores
    .OrderByDescending(s => s.Score)
    .ToList();

foreach (var item in reranked)
    Console.WriteLine($"Score: {item.Score:F4}  Document: {item.Document}");

public class ScoredResult
{
    public string Document { get; set; } = "";
    public float Score { get; set; }
}
```

## API Reference

### `BgeRerankerModel.CreateRerankerAsync`

```csharp
public static async Task<OnnxRerankerTransformer> CreateRerankerAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

Creates a reranker transformer backed by the local ONNX model. Downloads the model on first call; cached thereafter.

**Parameters:**
- `options` — Optional `ModelOptions` for controlling download behavior (cache directory, etc.)
- `ct` — Cancellation token

**Returns:** `OnnxRerankerTransformer` — an ML.NET transformer that scores (query, document) pairs.

**Configuration:**
- `MaxTokenLength`: 512
- `BatchSize`: 8

### `BgeRerankerModel.EnsureModelAsync`

```csharp
public static Task<string> EnsureModelAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

Downloads the model if not already cached and returns the local file path to the ONNX model.

### `BgeRerankerModel.GetModelInfoAsync`

Returns metadata about the model package (ID, source, file sizes).

### `BgeRerankerModel.VerifyModelAsync`

Verifies the integrity of the cached model using SHA-256 checksums.

## Inputs & Outputs

### Input

| Field | Type | Description |
|---|---|---|
| `Query` | `string` | The search query or question |
| `Document` | `string` | A candidate document to score against the query |

Inputs are tokenized and truncated to **512 tokens**. Longer texts are silently truncated.

### Output

| Field | Type | Description |
|---|---|---|
| `Score` | `float` | Relevance score for the (query, document) pair |

**Higher scores indicate greater relevance.** Sort descending by score to get reranked results.

> **Note:** Scores are not probabilities — they are unbounded logits. They are meaningful for *ordering* documents relative to a single query, not for comparing across different queries.

## Use Cases in RAG Pipeline

Reranking fits between retrieval and generation in a RAG pipeline:

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐     ┌────────────┐
│  Query   │────▶│  Retriever   │────▶│  Reranker    │────▶│ Generator  │
│          │     │ (Embeddings) │     │ (BGE/Cross)  │     │   (LLM)    │
└──────────┘     └──────────────┘     └──────────────┘     └────────────┘
                  Returns top-K         Rescores &           Uses top-N
                  candidates by         reorders by          documents as
                  vector similarity     true relevance       context
```

1. **Retrieve** — Use an embedding model (e.g., `AllMiniLM`, `BgeSmallEn`) to find the top 50–100 candidate documents by vector similarity.
2. **Rerank** — Pass each (query, candidate) pair through the cross-encoder reranker. This produces accurate relevance scores.
3. **Generate** — Feed the top 3–5 reranked documents as context to an LLM (e.g., `Phi3Mini`) for answer generation.

This two-stage approach gives you the **speed** of embedding retrieval with the **accuracy** of cross-encoder scoring.

## Choosing Between Reranking Models

| Property | BGE Reranker Base | MS MARCO MiniLM-L6-v2 |
|---|---|---|
| **Model Size** | ~1.11 GB | ~91 MB |
| **Speed** | Slower | **Faster** (~12× smaller) |
| **Accuracy** | **Higher** | Good |
| **Base Architecture** | XLM-RoBERTa | MiniLM (6 layers) |
| **Best For** | High-quality reranking, multilingual | Low-latency, resource-constrained |
| **Multilingual** | Yes | English-focused |

**Choose BGE Reranker** when accuracy matters most and you can afford a larger model.
**Choose MS MARCO MiniLM** when speed and memory are critical or you only need English reranking.

## Limitations

- **512-token limit** — Longer documents are truncated. Consider chunking long documents before reranking.
- **Latency** — Cross-encoders score each (query, document) pair individually. Reranking 100 documents will be slower than reranking 10. Keep the candidate set to a reasonable size (typically 20–100).
- **Not for initial retrieval** — Cross-encoders must process each pair; they cannot efficiently search over millions of documents. Use embedding-based retrieval first.
- **Scores are relative** — Do not compare scores across different queries; they are only meaningful for ordering documents within a single query.

## Example: Full RAG Pipeline with Reranking

```csharp
using DotnetAILab.ModelGarden.Embeddings.AllMiniLM;
using DotnetAILab.ModelGarden.Reranking.BgeReranker;
using DotnetAILab.ModelGarden.TextGeneration.Phi3Mini;
using Microsoft.ML;

// Step 1: Retrieve candidates using embeddings
var embedder = await AllMiniLMModel.CreateEmbedderAsync();
// ... embed query and documents, find top-K by cosine similarity ...

// Step 2: Rerank the top candidates
var reranker = await BgeRerankerModel.CreateRerankerAsync();
var mlContext = new MLContext();

var pairs = topCandidates.Select(doc => new { Query = query, Document = doc });
var dataView = mlContext.Data.LoadFromEnumerable(pairs);
var reranked = reranker.Transform(dataView);

// Step 3: Take top-N reranked documents as context
var topDocs = mlContext.Data
    .CreateEnumerable<ScoredResult>(reranked, reuseRowObject: false)
    .OrderByDescending(s => s.Score)
    .Take(3)
    .Select(s => s.Document);

// Step 4: Generate answer with context
var generator = await Phi3MiniModel.CreateTextGeneratorAsync();
var context = string.Join("\n\n", topDocs);
var prompt = $"Context:\n{context}\n\nQuestion: {query}\nAnswer:";
// ... pass prompt to generator ...
```

## Related Models

| Model | Task | Package |
|---|---|---|
| AllMiniLM | Embeddings (retrieval) | `DotnetAILab.ModelGarden.Embeddings.AllMiniLM` |
| BGE Small EN | Embeddings (retrieval) | `DotnetAILab.ModelGarden.Embeddings.BgeSmallEn` |
| MS MARCO MiniLM | Reranking (alternative) | `DotnetAILab.ModelGarden.Reranking.MsMarcoMiniLM` |
| Phi-3 Mini | Text generation | `DotnetAILab.ModelGarden.TextGeneration.Phi3Mini` |

## References

- [BAAI/bge-reranker-base on Hugging Face](https://huggingface.co/BAAI/bge-reranker-base)
- [C-Pack: Packaged Resources To Advance General Chinese Embedding (arXiv)](https://arxiv.org/abs/2309.07597)
- [FlagEmbedding GitHub Repository](https://github.com/FlagOpen/FlagEmbedding)
