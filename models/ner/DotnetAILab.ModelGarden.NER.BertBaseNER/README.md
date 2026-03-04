# DotnetAILab.ModelGarden.NER.BertBaseNER

> BERT-base model fine-tuned for Named Entity Recognition (NER) on English text, identifying persons, organizations, locations, and miscellaneous entities.

## Overview

**Named Entity Recognition (NER)** is the task of identifying and classifying named entities in text into predefined categories such as person names, organizations, locations, and more. This model package wraps the [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER) ONNX model with a simple .NET facade that auto-downloads and caches the model on first use.

The model uses the **BIO tagging scheme** (Beginning, Inside, Outside):
- **B-** prefix marks the **beginning** of an entity span (e.g., `B-PER` for the first token of a person's name)
- **I-** prefix marks tokens **inside** (continuation of) an entity span (e.g., `I-PER` for subsequent tokens)
- **O** marks tokens that are **not** part of any entity

For example, in _"John Smith works at Microsoft"_:
```
John   → B-PER    (beginning of a person entity)
Smith  → I-PER    (inside/continuation of person entity)
works  → O        (not an entity)
at     → O        (not an entity)
Microsoft → B-ORG (beginning of an organization entity)
```

## Model Details

| Property | Value |
|---|---|
| **Model** | [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER) |
| **Architecture** | BERT base (cased) |
| **Format** | ONNX |
| **Model Size** | ~431 MB |
| **Language** | English only |
| **Max Token Length** | 128 |
| **Entity Types** | PER, ORG, LOC, MISC |
| **Tagging Scheme** | BIO (B-/I-/O) |
| **Labels** | `O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`, `B-MISC`, `I-MISC` |
| **NuGet Package** | `DotnetAILab.ModelGarden.NER.BertBaseNER` |
| **Target Framework** | .NET 10.0 |

## Installation

```shell
dotnet add package DotnetAILab.ModelGarden.NER.BertBaseNER
```

### Dependencies

| Package | Version |
|---|---|
| `ModelPackages` | 0.1.0-preview.14 |
| `MLNet.TextInference.Onnx` | 0.1.0-preview.1 |

## Quick Start

```csharp
using DotnetAILab.ModelGarden.NER.BertBaseNER;
using Microsoft.ML;

// Create the NER transformer (downloads model on first use)
var ner = await BertBaseNERModel.CreateNerAsync();

// Prepare input
var mlContext = new MLContext();
var data = mlContext.Data.LoadFromEnumerable(new[]
{
    new { Text = "John Smith works at Microsoft in Seattle." }
});

// Run NER inference
var predictions = ner.Transform(data);

// Read results
var textColumn = predictions.GetColumn<string>("Text").ToArray();
var entityColumn = predictions.GetColumn<string>("Entities").ToArray();

Console.WriteLine($"Text: {textColumn[0]}");
Console.WriteLine($"Entities: {entityColumn[0]}");
// Expected entities: (PER: "John Smith"), (ORG: "Microsoft"), (LOC: "Seattle")
```

## API Reference

### `BertBaseNERModel.CreateNerAsync(...) → OnnxNerTransformer`

Creates a NER transformer backed by the local ONNX model. Downloads the model on first call; cached thereafter.

```csharp
public static async Task<OnnxNerTransformer> CreateNerAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `options` | `ModelOptions?` | Optional. Configuration for model download (cache directory, proxy, etc.) |
| `ct` | `CancellationToken` | Optional. Cancellation token for the async operation |

**Returns:** `Task<OnnxNerTransformer>` — A fitted ML.NET transformer that accepts text input and produces entity annotations.

### `BertBaseNERModel.EnsureModelAsync(...) → string`

Downloads the ONNX model if not already cached and returns the local file path.

```csharp
public static Task<string> EnsureModelAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

### `BertBaseNERModel.Labels`

Static array of BIO entity labels used by this model:

```csharp
public static readonly string[] Labels =
    ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"];
```

### `BertBaseNERModel.GetModelInfoAsync(...) → ModelInfo`

Returns metadata about the model (source, size, hash).

### `BertBaseNERModel.VerifyModelAsync(...)`

Verifies the integrity of the cached model file against the manifest SHA-256 hash.

## Inputs & Outputs

### Input

- **Text** (`string`): A sentence or passage of English text to analyze.

### Output

- **Named entities**, each with:
  - **Type**: The entity category (`PER`, `ORG`, `LOC`, `MISC`)
  - **Text span**: The matched text from the input
  - **Position**: Start and end token positions in the input

### BIO Format Example

Given the input:

```
"John works at Microsoft in Seattle"
```

The model produces token-level labels:

| Token | Label |
|---|---|
| John | B-PER |
| works | O |
| at | O |
| Microsoft | B-ORG |
| in | O |
| Seattle | B-LOC |

These are decoded into entity spans:

| Entity Type | Text | Description |
|---|---|---|
| **PER** | "John" | Person name |
| **ORG** | "Microsoft" | Organization name |
| **LOC** | "Seattle" | Location name |

## Entity Types

| Label | Entity Type | Description | Examples |
|---|---|---|---|
| **PER** | Person | Names of people | _John Smith_, _Marie Curie_, _Dr. Johnson_ |
| **ORG** | Organization | Companies, institutions, agencies | _Microsoft_, _United Nations_, _MIT_ |
| **LOC** | Location | Physical locations, geopolitical areas | _Seattle_, _France_, _Mount Everest_ |
| **MISC** | Miscellaneous | Named entities that don't fit other categories | _English_ (language), _Nobel Prize_, _World War II_ |

## Use Cases

- **Information Extraction** — Pull structured entity data from unstructured text
- **Document Processing** — Automatically tag and index documents by the entities they mention
- **Knowledge Graph Construction** — Extract entity nodes and relationships from text corpora
- **Content Indexing** — Enrich search indexes with entity metadata for improved retrieval
- **Resume Parsing** — Extract candidate names (PER), employers (ORG), and education institutions (ORG)
- **Address Extraction** — Identify locations (LOC) from correspondence or forms
- **News Analysis** — Track mentions of people, organizations, and places across articles

## Limitations

- **English only** — This model is trained on English text and will not produce reliable results on other languages. For multilingual support, see [MultilingualNER](../DotnetAILab.ModelGarden.NER.MultilingualNER/).
- **Fixed entity types** — Only recognizes PER, ORG, LOC, and MISC. Custom entity types cannot be added without retraining.
- **Context window** — Maximum input length is 128 tokens. Longer texts are truncated; consider splitting text into sentences or paragraphs.
- **No DATE entity** — Dates, times, and temporal expressions are not recognized as entities by this model. Use [MultilingualNER](../DotnetAILab.ModelGarden.NER.MultilingualNER/) if DATE extraction is needed.
- **Subword tokenization** — BERT tokenizes words into subword pieces, which can affect entity boundary alignment on unusual or domain-specific terms.
- **Model size** — The ONNX model is ~431 MB and is downloaded on first use. Ensure sufficient disk space and bandwidth.

## Example: Extracting Entities from a News Article

```csharp
using DotnetAILab.ModelGarden.NER.BertBaseNER;
using Microsoft.ML;

var ner = await BertBaseNERModel.CreateNerAsync();
var mlContext = new MLContext();

var articles = new[]
{
    new { Text = "Apple CEO Tim Cook announced a new partnership with NASA to develop AI tools at their headquarters in Cupertino." },
    new { Text = "The United Nations Secretary-General António Guterres addressed climate change at a summit in Geneva." },
    new { Text = "Researchers at Stanford University published findings on the Amazon rainforest ecosystem." }
};

var data = mlContext.Data.LoadFromEnumerable(articles);
var results = ner.Transform(data);

var texts = results.GetColumn<string>("Text").ToArray();
var entities = results.GetColumn<string>("Entities").ToArray();

for (int i = 0; i < texts.Length; i++)
{
    Console.WriteLine($"Article: {texts[i]}");
    Console.WriteLine($"Entities: {entities[i]}");
    Console.WriteLine();
}

// Expected output includes:
// Article 1: PER: "Tim Cook", ORG: "Apple", ORG: "NASA", LOC: "Cupertino"
// Article 2: ORG: "United Nations", PER: "António Guterres", LOC: "Geneva"
// Article 3: ORG: "Stanford University", LOC: "Amazon"
```

## Related Models

| Model | Package | Description |
|---|---|---|
| **Multilingual NER** | [`DotnetAILab.ModelGarden.NER.MultilingualNER`](../DotnetAILab.ModelGarden.NER.MultilingualNER/) | Multilingual BERT NER supporting English, German, Dutch, Spanish, and more. Recognizes PER, ORG, LOC, and **DATE** entities. |

## References

- **Hugging Face Model**: [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER)
- **BERT Paper**: Devlin et al., _"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"_ (2019) — [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- **CoNLL-2003 Dataset**: Tjong Kim Sang and De Meulder, _"Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition"_ (2003)
- **ML.NET Documentation**: [https://learn.microsoft.com/dotnet/machine-learning/](https://learn.microsoft.com/dotnet/machine-learning/)
