# DotnetAILab.ModelGarden.NER.MultilingualNER

> Multilingual BERT model fine-tuned for Named Entity Recognition (NER) across multiple languages, identifying persons, organizations, locations, and dates.

## Overview

**Named Entity Recognition (NER)** is the task of identifying and classifying named entities in text into predefined categories such as person names, organizations, locations, and more. This model package wraps the [Davlan/bert-base-multilingual-cased-ner-hrl](https://huggingface.co/Davlan/bert-base-multilingual-cased-ner-hrl) ONNX model with a simple .NET facade that auto-downloads and caches the model on first use.

Unlike the English-only [BertBaseNER](../DotnetAILab.ModelGarden.NER.BertBaseNER/), this model supports **multiple languages** including English, German, Dutch, Spanish, French, Chinese, Arabic, and more. It also recognizes **DATE** entities instead of MISC entities.

The model uses the **BIO tagging scheme** (Beginning, Inside, Outside):
- **B-** prefix marks the **beginning** of an entity span (e.g., `B-PER` for the first token of a person's name)
- **I-** prefix marks tokens **inside** (continuation of) an entity span (e.g., `I-PER` for subsequent tokens)
- **O** marks tokens that are **not** part of any entity

For example, in _"Angela Merkel besuchte Berlin am 5. Mai"_ (German):
```
Angela   → B-PER    (beginning of a person entity)
Merkel   → I-PER    (inside/continuation of person entity)
besuchte → O        (not an entity)
Berlin   → B-LOC    (beginning of a location entity)
am       → O        (not an entity)
5.       → B-DATE   (beginning of a date entity)
Mai      → I-DATE   (inside/continuation of date entity)
```

## Model Details

| Property | Value |
|---|---|
| **Model** | [Davlan/bert-base-multilingual-cased-ner-hrl](https://huggingface.co/Davlan/bert-base-multilingual-cased-ner-hrl) |
| **Architecture** | BERT base multilingual (cased) |
| **Format** | ONNX |
| **Model Size** | ~709 MB |
| **Languages** | English, German, Dutch, Spanish, French, Chinese, Arabic, and others |
| **Max Token Length** | 128 |
| **Entity Types** | PER, ORG, LOC, DATE |
| **Tagging Scheme** | BIO (B-/I-/O) |
| **Labels** | `O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`, `B-DATE`, `I-DATE` |
| **NuGet Package** | `DotnetAILab.ModelGarden.NER.MultilingualNER` |
| **Target Framework** | .NET 10.0 |

## Installation

```shell
dotnet add package DotnetAILab.ModelGarden.NER.MultilingualNER
```

### Dependencies

| Package | Version |
|---|---|
| `ModelPackages` | 0.1.0-preview.14 |
| `MLNet.TextInference.Onnx` | 0.1.0-preview.1 |

## Quick Start

```csharp
using DotnetAILab.ModelGarden.NER.MultilingualNER;
using Microsoft.ML;

// Create the NER transformer (downloads model on first use)
var ner = await MultilingualNERModel.CreateNerAsync();

// Prepare input
var mlContext = new MLContext();
var data = mlContext.Data.LoadFromEnumerable(new[]
{
    new { Text = "Angela Merkel visited the United Nations in New York on January 15th." }
});

// Run NER inference
var predictions = ner.Transform(data);

// Read results
var textColumn = predictions.GetColumn<string>("Text").ToArray();
var entityColumn = predictions.GetColumn<string>("Entities").ToArray();

Console.WriteLine($"Text: {textColumn[0]}");
Console.WriteLine($"Entities: {entityColumn[0]}");
// Expected entities: (PER: "Angela Merkel"), (ORG: "United Nations"),
//                    (LOC: "New York"), (DATE: "January 15th")
```

## API Reference

### `MultilingualNERModel.CreateNerAsync(...) → OnnxNerTransformer`

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

### `MultilingualNERModel.EnsureModelAsync(...) → string`

Downloads the ONNX model if not already cached and returns the local file path.

```csharp
public static Task<string> EnsureModelAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

### `MultilingualNERModel.Labels`

Static array of BIO entity labels used by this model:

```csharp
public static readonly string[] Labels =
    ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-DATE", "I-DATE"];
```

### `MultilingualNERModel.GetModelInfoAsync(...) → ModelInfo`

Returns metadata about the model (source, size, hash).

### `MultilingualNERModel.VerifyModelAsync(...)`

Verifies the integrity of the cached model file against the manifest SHA-256 hash.

## Inputs & Outputs

### Input

- **Text** (`string`): A sentence or passage of text in any supported language.

### Output

- **Named entities**, each with:
  - **Type**: The entity category (`PER`, `ORG`, `LOC`, `DATE`)
  - **Text span**: The matched text from the input
  - **Position**: Start and end token positions in the input

### BIO Format Example

Given the input:

```
"Marie Curie trabajó en la Universidad de París el 3 de noviembre"
```

The model produces token-level labels:

| Token | Label |
|---|---|
| Marie | B-PER |
| Curie | I-PER |
| trabajó | O |
| en | O |
| la | O |
| Universidad | B-ORG |
| de | I-ORG |
| París | I-ORG |
| el | O |
| 3 | B-DATE |
| de | I-DATE |
| noviembre | I-DATE |

These are decoded into entity spans:

| Entity Type | Text | Description |
|---|---|---|
| **PER** | "Marie Curie" | Person name |
| **ORG** | "Universidad de París" | Organization name |
| **DATE** | "3 de noviembre" | Date expression |

## Entity Types

| Label | Entity Type | Description | Examples |
|---|---|---|---|
| **PER** | Person | Names of people | _Angela Merkel_, _José García_, _王明_ |
| **ORG** | Organization | Companies, institutions, agencies | _United Nations_, _Siemens_, _Universidad de Barcelona_ |
| **LOC** | Location | Physical locations, geopolitical areas | _Berlin_, _New York_, _río Amazonas_ |
| **DATE** | Date | Dates, temporal expressions | _January 15th_, _3 de noviembre_, _2024_ |

> **Note:** This model recognizes **DATE** entities instead of the **MISC** (miscellaneous) type found in [BertBaseNER](../DotnetAILab.ModelGarden.NER.BertBaseNER/). If you need MISC entity recognition (e.g., languages, events, nationalities), use BertBaseNER instead.

## Supported Languages

This model is trained on high-resource languages (HRL) and provides strong NER performance across:

| Language | Code | Example |
|---|---|---|
| English | en | _"John works at Google in London"_ |
| German | de | _"Angela Merkel besuchte Berlin"_ |
| Dutch | nl | _"Willem werkt bij Philips in Amsterdam"_ |
| Spanish | es | _"María trabaja en Barcelona"_ |
| French | fr | _"Jean travaille à Paris"_ |
| Chinese | zh | _"王明在北京大学工作"_ |
| Arabic | ar | _"أحمد يعمل في القاهرة"_ |

> **Note:** Performance may vary by language and entity type. Languages with more training data (English, German, Spanish) tend to have higher accuracy.

## Use Cases

- **Multilingual Information Extraction** — Extract structured entity data from documents in multiple languages
- **International Document Processing** — Tag and index multilingual documents by entities
- **Knowledge Graph Construction** — Build cross-lingual knowledge graphs from diverse text sources
- **Content Indexing** — Enrich multilingual search indexes with entity metadata
- **Event Timeline Extraction** — Combine DATE entities with PER/ORG/LOC to build event timelines
- **Resume Parsing** — Extract names (PER), employers (ORG), and dates (DATE) from multilingual CVs
- **Address Extraction** — Identify locations (LOC) from international correspondence
- **News Monitoring** — Track entities across multilingual news feeds

## Limitations

- **Fixed entity types** — Only recognizes PER, ORG, LOC, and DATE. Custom entity types cannot be added without retraining.
- **No MISC entity** — Unlike BertBaseNER, this model does not recognize miscellaneous entities (languages, events, nationalities). Use [BertBaseNER](../DotnetAILab.ModelGarden.NER.BertBaseNER/) if MISC recognition is needed.
- **Context window** — Maximum input length is 128 tokens. Longer texts are truncated; consider splitting text into sentences or paragraphs.
- **Language support varies by entity type** — DATE recognition may be less accurate in languages with fewer training examples.
- **Subword tokenization** — Multilingual BERT tokenizes words into subword pieces, which can affect entity boundary alignment, especially for non-Latin scripts.
- **Model size** — The ONNX model is ~709 MB (larger than BertBaseNER due to multilingual vocabulary) and is downloaded on first use. Ensure sufficient disk space and bandwidth.
- **Low-resource languages** — While the underlying multilingual BERT supports 104 languages, NER fine-tuning was performed on high-resource languages only. Performance on low-resource languages is not guaranteed.

## Example: Extracting Entities from Multilingual News

```csharp
using DotnetAILab.ModelGarden.NER.MultilingualNER;
using Microsoft.ML;

var ner = await MultilingualNERModel.CreateNerAsync();
var mlContext = new MLContext();

var articles = new[]
{
    // English
    new { Text = "President Biden met with EU leaders in Brussels on March 24th." },
    // German
    new { Text = "Bundeskanzler Scholz besuchte die Europäische Kommission in Brüssel am 15. März." },
    // Spanish
    new { Text = "El presidente Sánchez visitó la ONU en Nueva York el 10 de enero." },
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
// English:  PER: "Biden", ORG: "EU", LOC: "Brussels", DATE: "March 24th"
// German:   PER: "Scholz", ORG: "Europäische Kommission", LOC: "Brüssel", DATE: "15. März"
// Spanish:  PER: "Sánchez", ORG: "ONU", LOC: "Nueva York", DATE: "10 de enero"
```

## Related Models

| Model | Package | Description |
|---|---|---|
| **BERT Base NER** | [`DotnetAILab.ModelGarden.NER.BertBaseNER`](../DotnetAILab.ModelGarden.NER.BertBaseNER/) | English-only BERT NER recognizing PER, ORG, LOC, and **MISC** entities. Smaller model (~431 MB) with strong English performance. |

### Choosing Between NER Models

| Feature | BertBaseNER | MultilingualNER |
|---|---|---|
| **Languages** | English only | English, German, Dutch, Spanish, French, Chinese, Arabic, + more |
| **Entity Types** | PER, ORG, LOC, **MISC** | PER, ORG, LOC, **DATE** |
| **Model Size** | ~431 MB | ~709 MB |
| **Best For** | English-only workloads needing MISC entities | Multilingual workloads or DATE extraction |

## References

- **Hugging Face Model**: [Davlan/bert-base-multilingual-cased-ner-hrl](https://huggingface.co/Davlan/bert-base-multilingual-cased-ner-hrl)
- **Paper**: Adelani et al., _"MasakhaNER: Named Entity Recognition for African Languages"_ (2021) — [arXiv:2103.11811](https://arxiv.org/abs/2103.11811)
- **BERT Multilingual**: Devlin et al., _"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"_ (2019) — [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- **ML.NET Documentation**: [https://learn.microsoft.com/dotnet/machine-learning/](https://learn.microsoft.com/dotnet/machine-learning/)
