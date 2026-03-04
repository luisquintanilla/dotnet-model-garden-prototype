# MS MARCO MiniLM-L6-v2 — Reranking

> Lightweight cross-encoder for query-document relevance scoring, trained on the MS MARCO passage ranking dataset — fast and efficient reranking for search and RAG pipelines.

## Overview

**What is reranking?** After an initial retrieval step returns candidate documents (typically via embedding similarity), a **reranker** rescores each (query, document) pair with a more accurate model. This lets you re-sort candidates so the most relevant documents float to the top, significantly improving answer quality in downstream tasks like RAG.

**Cross-encoder architecture.** Unlike bi-encoder embedding models that encode queries and documents separately, cross-encoders process the query and document *together* through the full transformer. This joint processing captures deeper semantic interactions, producing more accurate relevance judgments — at the cost of higher per-pair latency.

**Why MS MARCO MiniLM-L6-v2?** This model distills the cross-encoder approach into a compact 6-layer MiniLM architecture (~91 MB), making it roughly **12× smaller** than alternatives like BGE Reranker Base. Trained on Microsoft's MS MARCO passage ranking dataset (8.8M+ passages), it delivers strong English relevance scoring at a fraction of the compute cost. Choose this model when speed and memory efficiency matter more than maximum accuracy.

## Model Details

| Property | Value |
|---|---|
| **Model ID** | [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) |
| **Architecture** | Cross-encoder (MiniLM, 6 layers) |
| **Task** | Reranking / relevance scoring |
| **Training Data** | MS MARCO Passage Ranking |
| **Max Token Length** | 512 tokens |
| **ONNX File Size** | ~91 MB |
| **Source** | Hugging Face |
| **License** | Apache 2.0 |
| **NuGet Package** | `DotnetAILab.ModelGarden.Reranking.MsMarcoMiniLM` |

## Installation

Add the NuGet package to your project:

```shell
dotnet add package DotnetAILab.ModelGarden.Reranking.MsMarcoMiniLM
```

The ONNX model binary (~91 MB) is **automatically downloaded** on first use and cached locally. No manual download is required.

### Dependencies

| Package | Version |
|---|---|
| `ModelPackages` | 0.1.0-preview.14 |
| `MLNet.TextInference.Onnx` | 0.1.0-preview.1 |

## Quick Start

```csharp
using DotnetAILab.ModelGarden.Reranking.MsMarcoMiniLM;
using Microsoft.ML;

// Create the reranker (downloads model on first call)
var reranker = await MsMarcoMiniLMModel.CreateRerankerAsync();

var mlContext = new MLContext();

// Prepare query-document pairs
var query = "How does photosynthesis work?";
var documents = new[]
{
    "Photosynthesis converts sunlight, water, and CO2 into glucose and oxygen in plants.",
    "The mitochondria is the powerhouse of the cell.",
    "Plants use chlorophyll to absorb light energy during photosynthesis.",
    "The water cycle involves evaporation, condensation, and precipitation."
};

// Score each document against the query
var pairs = documents.Select(doc => new { Query = query, Document = doc });
var dataView = mlContext.Data.LoadFromEnumerable(pairs);
var scored = reranker.Transform(dataView);

// Read relevance scores and sort descending
var results = mlContext.Data
    .CreateEnumerable<ScoredResult>(scored, reuseRowObject: false)
    .OrderByDescending(s => s.Score)
    .ToList();

foreach (var item in results)
    Console.WriteLine($"Score: {item.Score:F4}  Document: {item.Document}");

public class ScoredResult
{
    public string Document { get; set; } = "";
    public float Score { get; set; }
}
```

## API Reference

### `MsMarcoMiniLMModel.CreateRerankerAsync`

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

### `MsMarcoMiniLMModel.EnsureModelAsync`

```csharp
public static Task<string> EnsureModelAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

Downloads the model if not already cached and returns the local file path to the ONNX model.

### `MsMarcoMiniLMModel.GetModelInfoAsync`

Returns metadata about the model package (ID, source, file sizes).

### `MsMarcoMiniLMModel.VerifyModelAsync`

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
│          │     │ (Embeddings) │     │ (MiniLM/CE)  │     │   (LLM)    │
└──────────┘     └──────────────┘     └──────────────┘     └────────────┘
                  Returns top-K         Rescores &           Uses top-N
                  candidates by         reorders by          documents as
                  vector similarity     true relevance       context
```

1. **Retrieve** — Use an embedding model (e.g., `AllMiniLM`, `BgeSmallEn`) to find the top 50–100 candidate documents by vector similarity.
2. **Rerank** — Pass each (query, candidate) pair through the cross-encoder reranker. MS MARCO MiniLM is particularly fast for this step.
3. **Generate** — Feed the top 3–5 reranked documents as context to an LLM (e.g., `Phi3Mini`) for answer generation.

This two-stage approach gives you the **speed** of embedding retrieval with the **accuracy** of cross-encoder scoring.

## Choosing Between Reranking Models

| Property | MS MARCO MiniLM-L6-v2 | BGE Reranker Base |
|---|---|---|
| **Model Size** | **~91 MB** | ~1.11 GB |
| **Speed** | **Faster** (~12× smaller) | Slower |
| **Accuracy** | Good | **Higher** |
| **Base Architecture** | MiniLM (6 layers) | XLM-RoBERTa |
| **Training Data** | MS MARCO passages | General retrieval |
| **Best For** | Low-latency, English search | High-quality, multilingual |
| **Multilingual** | English-focused | Yes |

**Choose MS MARCO MiniLM** when you need fast reranking, have memory constraints, or only work with English text.
**Choose BGE Reranker** when accuracy is the top priority or you need multilingual support.

## Limitations

- **512-token limit** — Longer documents are truncated. Consider chunking long documents before reranking.
- **English-focused** — Trained primarily on English MS MARCO data. For multilingual reranking, consider BGE Reranker Base.
- **Latency** — Cross-encoders score each (query, document) pair individually. Keep the candidate set to a reasonable size (typically 20–100).
- **Not for initial retrieval** — Cross-encoders must process each pair; they cannot efficiently search over millions of documents. Use embedding-based retrieval first.
- **Scores are relative** — Do not compare scores across different queries; they are only meaningful for ordering documents within a single query.
- **Lower accuracy ceiling** — The compact 6-layer architecture trades some accuracy for speed compared to larger rerankers.

## Example: RAG Pipeline with Reranking

```csharp
using DotnetAILab.ModelGarden.Embeddings.AllMiniLM;
using DotnetAILab.ModelGarden.Reranking.MsMarcoMiniLM;
using DotnetAILab.ModelGarden.TextGeneration.Phi3Mini;
using Microsoft.ML;

// Step 1: Retrieve candidates using embeddings
var embedder = await AllMiniLMModel.CreateEmbedderAsync();
// ... embed query and documents, find top-K by cosine similarity ...

// Step 2: Rerank the top candidates
var reranker = await MsMarcoMiniLMModel.CreateRerankerAsync();
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
| BGE Reranker Base | Reranking (alternative) | `DotnetAILab.ModelGarden.Reranking.BgeReranker` |
| Phi-3 Mini | Text generation | `DotnetAILab.ModelGarden.TextGeneration.Phi3Mini` |

## References

- [cross-encoder/ms-marco-MiniLM-L-6-v2 on Hugging Face](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
- [MS MARCO: A Human Generated MAchine Reading COmprehension Dataset](https://microsoft.github.io/msmarco/)
- [Sentence-Transformers Cross-Encoders Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
