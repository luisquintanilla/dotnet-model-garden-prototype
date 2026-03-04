# DotnetAILab.ModelGarden.QA.RobertaSquad2

> High-accuracy extractive question answering powered by RoBERTa-base, fine-tuned on SQuAD 2.0, running locally via ONNX.

## Overview

**Extractive question answering** is the task of finding the exact answer span within a given context passage for a question. Unlike generative QA (which synthesizes new text), extractive QA locates and returns a substring directly from the provided context—making answers traceable and grounded.

This package wraps a **RoBERTa-base** model fine-tuned on [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) (Stanford Question Answering Dataset). SQuAD 2.0 extends the original SQuAD by including over 50,000 unanswerable questions, training the model to know when the context does not contain a valid answer.

RoBERTa (Robustly Optimized BERT Approach) improves on BERT through longer training, larger batches, and dynamic masking. The base variant offers strong accuracy on reading comprehension tasks and is well-suited for applications where answer quality is the top priority, even at the cost of higher latency and memory usage compared to distilled alternatives.

The ONNX model binary (~473 MB) is automatically downloaded from Hugging Face on first use and cached locally.

## Model Details

| Property | Value |
|---|---|
| **Model** | RoBERTa-base |
| **Fine-tuned on** | SQuAD 2.0 |
| **Task** | Extractive Question Answering |
| **Format** | ONNX |
| **Model size** | ~473 MB |
| **Max context tokens** | 384 |
| **Max answer tokens** | 30 |
| **Language** | English |
| **Source** | [lquint/roberta-base-squad2-onnx](https://huggingface.co/lquint/roberta-base-squad2-onnx) |
| **Framework** | .NET 10 / ML.NET |

## Installation

Add the NuGet package to your project:

```shell
dotnet add package DotnetAILab.ModelGarden.QA.RobertaSquad2
```

### Dependencies

| Package | Version |
|---|---|
| `ModelPackages` | 0.1.0-preview.14 |
| `MLNet.TextInference.Onnx` | 0.1.0-preview.1 |

## Quick Start

```csharp
using DotnetAILab.ModelGarden.QA.RobertaSquad2;
using Microsoft.ML;

// Create the QA transformer (downloads model on first use)
var qa = await RobertaSquad2Model.CreateQaAsync();

var mlContext = new MLContext();

var input = mlContext.Data.LoadFromEnumerable(new[]
{
    new
    {
        Context = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
                + "in Paris, France. It was constructed from 1887 to 1889 as the centerpiece "
                + "of the 1889 World's Fair.",
        Question = "When was the Eiffel Tower built?"
    }
});

var output = qa.Transform(input);

// Read the predicted answer
var answers = mlContext.Data
    .CreateEnumerable<QaAnswer>(output, reuseRowObject: false)
    .ToList();

Console.WriteLine(answers[0].Answer);
// → "from 1887 to 1889"
```

> **Note:** The first call to `CreateQaAsync` downloads the ONNX model (~473 MB) from Hugging Face. Subsequent calls use the local cache.

## API Reference

### `RobertaSquad2Model.CreateQaAsync(...)` → `OnnxQaTransformer`

Creates a fitted ML.NET transformer for question answering, backed by the local ONNX model.

```csharp
public static async Task<OnnxQaTransformer> CreateQaAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `options` | `ModelOptions?` | Optional configuration for model download/caching behavior. |
| `ct` | `CancellationToken` | Cancellation token for the async operation. |

**Returns:** `Task<OnnxQaTransformer>` — A fitted transformer that accepts `IDataView` rows with `Context` and `Question` columns and produces answer predictions.

### `RobertaSquad2Model.EnsureModelAsync(...)`

Downloads (if needed) and returns the local file path to the cached ONNX model.

```csharp
public static Task<string> EnsureModelAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

### `RobertaSquad2Model.GetModelInfoAsync(...)`

Returns metadata about the model package.

```csharp
public static Task<ModelInfo> GetModelInfoAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

### `RobertaSquad2Model.VerifyModelAsync(...)`

Verifies the integrity of the cached model file (SHA-256 check).

```csharp
public static Task VerifyModelAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

## Inputs & Outputs

### Input

| Field | Type | Description |
|---|---|---|
| `Context` | `string` | The passage of text containing the potential answer. |
| `Question` | `string` | The question to answer based on the context. |

### Output

| Field | Type | Description |
|---|---|---|
| `Answer` | `string` | The extracted answer span from the context. |
| `Score` | `float` | Confidence score for the predicted answer. |

### Constraints

- **Max context length:** 384 tokens. Longer passages are truncated; consider chunking long documents.
- **Max answer length:** 30 tokens. Answers exceeding this are clipped.
- **Unanswerable questions:** When the context does not contain an answer, the model may return an empty string or a low-confidence score. Check the `Score` field to filter uncertain predictions.

## Use Cases

- **FAQ systems** — Match user questions against a knowledge base of context passages and return precise answers.
- **Document search** — Find specific answers within large documents by splitting them into chunks and querying each.
- **Customer support automation** — Extract answers from support documentation to respond to customer queries.
- **Reading comprehension** — Build educational tools that test or demonstrate reading comprehension.
- **Knowledge base querying** — Power internal tools that answer questions from company wikis, manuals, or policy documents.

## Choosing Between QA Models

| Model | Speed | Accuracy | Size | Best For |
|---|---|---|---|---|
| **MiniLMSquad2** | ⚡ Faster | Good | ~127 MB | High-throughput, latency-sensitive applications |
| **RobertaSquad2** | Slower | ✅ Better | ~473 MB | Accuracy-critical applications |

Choose **RobertaSquad2** when answer accuracy is the top priority and you can tolerate higher latency and memory usage. Choose [MiniLMSquad2](../DotnetAILab.ModelGarden.QA.MiniLMSquad2/) when you need fast inference, are running on constrained hardware, or are processing many queries in parallel.

## Limitations

- **Extractive only** — The answer must exist as a contiguous span within the provided context. The model cannot synthesize, paraphrase, or generate new text.
- **No multi-hop reasoning** — The model cannot combine information from multiple passages or perform logical inference across separate pieces of text.
- **Context window** — Limited to ~384 tokens. Documents longer than this must be split into overlapping chunks.
- **English only** — The model was trained on English text and does not support other languages.

## Example: Building a Document Q&A System

```csharp
using DotnetAILab.ModelGarden.QA.RobertaSquad2;
using Microsoft.ML;

// 1. Initialize the QA model
var qa = await RobertaSquad2Model.CreateQaAsync();
var mlContext = new MLContext();

// 2. Chunk your document into passages that fit within 384 tokens
string[] documentChunks = GetDocumentChunks(myDocument, maxTokens: 384);

// 3. Query each chunk with the user's question
string userQuestion = "What is the return policy?";

var inputs = documentChunks.Select(chunk => new
{
    Context = chunk,
    Question = userQuestion
}).ToArray();

var data = mlContext.Data.LoadFromEnumerable(inputs);
var results = qa.Transform(data);

var answers = mlContext.Data
    .CreateEnumerable<QaAnswer>(results, reuseRowObject: false)
    .ToList();

// 4. Pick the best answer by confidence score
var bestAnswer = answers
    .OrderByDescending(a => a.Score)
    .FirstOrDefault();

if (bestAnswer != null && bestAnswer.Score > 0.5f)
{
    Console.WriteLine($"Answer: {bestAnswer.Answer} (confidence: {bestAnswer.Score:P1})");
}
else
{
    Console.WriteLine("No confident answer found in the document.");
}
```

## Related Models

- **[DotnetAILab.ModelGarden.QA.MiniLMSquad2](../DotnetAILab.ModelGarden.QA.MiniLMSquad2/)** — MiniLM-uncased fine-tuned on SQuAD 2.0. Smaller and faster; recommended for high-throughput and latency-sensitive scenarios.

## References

- [SQuAD 2.0: Know What You Don't Know](https://arxiv.org/abs/1806.03822) — Rajpurkar et al., 2018
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — Liu et al., 2019
- [Hugging Face model card: lquint/roberta-base-squad2-onnx](https://huggingface.co/lquint/roberta-base-squad2-onnx)
- [ML.NET Documentation](https://learn.microsoft.com/dotnet/machine-learning/)
