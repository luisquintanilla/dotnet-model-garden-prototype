# DotnetAILab.ModelGarden.Classification.EmotionRoBERTa

> Multi-label emotion classification for text using RoBERTa fine-tuned on the GoEmotions dataset — detecting 28 nuanced emotion categories in a single pass.

## Overview

Understanding human emotion in text goes far beyond simple positive/negative sentiment. A customer review might express both **gratitude** and **disappointment** simultaneously; a social media post could mix **excitement** with **nervousness**. The EmotionRoBERTa model package brings this level of emotional granularity to .NET applications by wrapping a RoBERTa model fine-tuned on Google's GoEmotions dataset — the largest manually-annotated dataset of 58,000 Reddit comments labeled across 27 emotions plus neutral.

This package leverages the RoBERTa (Robustly Optimized BERT Pretraining Approach) architecture, which improves on BERT through dynamic masking, larger mini-batches, and training on more data. Because this is a **multi-label** classifier, a single text input can be associated with multiple emotions simultaneously, reflecting the real-world complexity of human expression. The model outputs confidence scores for all 28 labels, allowing you to threshold or rank emotions as needed for your application.

Like all packages in the .NET Model Garden, the heavy ONNX model binary (~499 MB) downloads transparently on first use and is cached locally. Your NuGet package stays lightweight — you get a simple, strongly-typed C# API with no Python dependencies, no Docker containers, and no external services required.

## Model Details

| Property | Value |
|---|---|
| **Architecture** | RoBERTa-base |
| **Task** | Multi-label text classification |
| **Training Dataset** | GoEmotions (58k Reddit comments) |
| **Labels** | 28 (27 emotions + neutral) |
| **Max Token Length** | 128 |
| **Model Format** | ONNX |
| **Model Size** | ~499 MB |
| **HuggingFace Repo** | [`lquint/roberta-base-go_emotions-onnx`](https://huggingface.co/lquint/roberta-base-go_emotions-onnx) |
| **NuGet Package** | `DotnetAILab.ModelGarden.Classification.EmotionRoBERTa` |
| **Target Framework** | .NET 10.0 |

## Installation

```bash
dotnet add package DotnetAILab.ModelGarden.Classification.EmotionRoBERTa
```

**Dependencies** (pulled in automatically):

- `ModelPackages` (0.1.0-preview.14) — model download, caching, and verification
- `MLNet.TextInference.Onnx` (0.1.0-preview.1) — ONNX text classification runtime

## Quick Start

```csharp
using DotnetAILab.ModelGarden.Classification.EmotionRoBERTa;

// Create classifier — model auto-downloads on first call (~499 MB), cached thereafter
var classifier = await EmotionRoBERTaModel.CreateClassifierAsync();

// Classify text
var results = classifier.Classify(["I'm so grateful for your help, this is amazing!"]);

Console.WriteLine($"Predicted: {results[0].PredictedLabel}");
Console.WriteLine($"Confidence: {results[0].Confidence:P1}");
```

## API Reference

### `EmotionRoBERTaModel` (static class)

**Namespace:** `DotnetAILab.ModelGarden.Classification.EmotionRoBERTa`

#### Properties

```csharp
public static readonly string[] Labels;
```

The 28 classification labels: `admiration`, `amusement`, `anger`, `annoyance`, `approval`, `caring`, `confusion`, `curiosity`, `desire`, `disappointment`, `disapproval`, `disgust`, `embarrassment`, `excitement`, `fear`, `gratitude`, `grief`, `joy`, `love`, `nervousness`, `optimism`, `pride`, `realization`, `relief`, `remorse`, `sadness`, `surprise`, `neutral`.

#### Methods

```csharp
public static async Task<OnnxTextClassificationTransformer> CreateClassifierAsync(
    ModelOptions? options = null,
    CancellationToken ct = default);
```

Creates a text classification transformer backed by the ONNX model. Downloads the model on first call; cached for subsequent calls. The returned `OnnxTextClassificationTransformer` provides a `Classify()` method for inference.

```csharp
public static Task<string> EnsureModelAsync(
    ModelOptions? options = null,
    CancellationToken ct = default);
```

Downloads the model (if not cached) and returns the local file path to the ONNX model. Useful if you need direct access to the model file for custom pipelines.

```csharp
public static Task<ModelInfo> GetModelInfoAsync(
    ModelOptions? options = null,
    CancellationToken ct = default);
```

Returns metadata about the model package (ID, revision, file sizes, SHA256 hashes).

```csharp
public static Task VerifyModelAsync(
    ModelOptions? options = null,
    CancellationToken ct = default);
```

Verifies the integrity of the cached model file against its expected SHA256 hash. Throws if verification fails.

### Classifier Usage

```csharp
// Single text
var results = classifier.Classify(["Some text to classify"]);

// Batch classification
var results = classifier.Classify(["Text one", "Text two", "Text three"]);
```

Each result exposes:
- `PredictedLabel` — the top predicted emotion label (e.g., `"gratitude"`)
- `Confidence` — confidence score for the predicted label (0.0–1.0)

## Inputs & Outputs

### Input

| Parameter | Type | Description |
|---|---|---|
| `texts` | `string[]` | Array of text strings to classify. Each text is tokenized up to 128 tokens. |

Texts longer than 128 tokens are truncated. The model processes inputs in batches of 8 for efficiency.

### Output

Each classification result includes:

| Field | Type | Description |
|---|---|---|
| `PredictedLabel` | `string` | The top-scoring emotion label from the 28 categories |
| `Confidence` | `float` | Confidence score (0.0–1.0) for the predicted label |

### Label Descriptions

| Label | Meaning |
|---|---|
| `admiration` | Finding something impressive or worthy of respect |
| `amusement` | Finding something funny or being entertained |
| `anger` | A strong feeling of displeasure or hostility |
| `annoyance` | Mild irritation or displeasure |
| `approval` | Agreeing with or endorsing something |
| `caring` | Showing concern or empathy for others |
| `confusion` | Lack of understanding; being perplexed |
| `curiosity` | Desire to learn or know more |
| `desire` | A strong feeling of wanting something |
| `disappointment` | Sadness from unmet expectations |
| `disapproval` | Disagreement or objection |
| `disgust` | Strong disapproval or revulsion |
| `embarrassment` | Self-conscious discomfort |
| `excitement` | Enthusiastic eagerness |
| `fear` | Feeling afraid or anxious |
| `gratitude` | Feeling thankful or appreciative |
| `grief` | Deep sorrow, especially from loss |
| `joy` | A feeling of happiness and delight |
| `love` | Strong affection or deep caring |
| `nervousness` | Worry or anxiety about something |
| `optimism` | Hopefulness about the future |
| `pride` | Satisfaction from achievements |
| `realization` | Sudden understanding or insight |
| `relief` | Comfort after anxiety or distress |
| `remorse` | Regret or guilt over a past action |
| `sadness` | Feeling unhappy or sorrowful |
| `surprise` | Unexpected discovery or event |
| `neutral` | No strong emotion detected |

## Use Cases

1. **Customer Feedback Analysis** — Go beyond star ratings. Detect *why* customers feel a certain way: gratitude for good service, annoyance with shipping, excitement about a feature.

2. **Social Media Monitoring** — Track emotional reactions to brand mentions, product launches, or campaigns across 28 emotion categories for nuanced brand health dashboards.

3. **Chatbot Emotion Awareness** — Enable conversational AI to detect user emotions and respond empathetically — escalating to a human agent when anger or disappointment is detected.

4. **Content Moderation Triage** — Flag content expressing disgust, anger, or fear for human review, while auto-approving content with neutral or positive emotions.

5. **Mental Health & Wellness Apps** — Monitor journaling entries for emotional patterns over time, detecting shifts toward sadness, grief, or nervousness that may warrant intervention.

6. **Employee Engagement Surveys** — Analyze open-ended survey responses to understand the emotional undertone beyond what structured questions capture.

7. **Educational Feedback** — Analyze student discussion forums to detect confusion (needs more explanation), excitement (topic resonates), or disappointment (unmet expectations).

## Limitations & Considerations

- **Training Data Bias**: The model was trained on Reddit comments, which skew toward informal English, younger demographics, and Western cultural norms. Emotion expression varies across cultures and communities.
- **Multi-label Nuance**: While the model supports multi-label classification internally, the facade API returns the single top-scoring label. For applications requiring multiple emotion detection, consider working with the raw model output.
- **Sarcasm & Irony**: Like most text classifiers, the model may misinterpret sarcastic text (e.g., "Oh great, another meeting" classified as `approval` instead of `annoyance`).
- **Token Limit**: Texts are truncated at 128 tokens (~100 words). For longer documents, consider splitting into sentences and aggregating results.
- **Language**: The model is trained on English text only. Non-English input will produce unreliable results.
- **First-run Download**: The ~499 MB model downloads on first use. Ensure adequate disk space and network access in deployment environments.

## Example: Customer Feedback Analysis

```csharp
using DotnetAILab.ModelGarden.Classification.EmotionRoBERTa;

var classifier = await EmotionRoBERTaModel.CreateClassifierAsync();

var reviews = new[]
{
    "I absolutely love this product! Best purchase I've made all year.",
    "Shipping took forever and the box arrived damaged. Very frustrating.",
    "The product is fine, nothing special. Does what it says.",
    "I was really excited to try this but it broke after two days. So disappointed.",
    "Thank you for the quick replacement — your support team is amazing!"
};

var results = classifier.Classify(reviews);

for (int i = 0; i < reviews.Length; i++)
{
    Console.WriteLine($"Review: \"{reviews[i]}\"");
    Console.WriteLine($"  Emotion: {results[i].PredictedLabel} ({results[i].Confidence:P1})");
    Console.WriteLine();
}

// Example output:
// Review: "I absolutely love this product! Best purchase I've made all year."
//   Emotion: love (92.3%)
//
// Review: "Shipping took forever and the box arrived damaged. Very frustrating."
//   Emotion: annoyance (87.1%)
//
// Review: "The product is fine, nothing special. Does what it says."
//   Emotion: neutral (74.5%)
//
// Review: "I was really excited to try this but it broke after two days. So disappointed."
//   Emotion: disappointment (85.8%)
//
// Review: "Thank you for the quick replacement — your support team is amazing!"
//   Emotion: gratitude (95.2%)
```

## Related Models

| Package | Task | When to Use |
|---|---|---|
| [`DotnetAILab.ModelGarden.Classification.SentimentDistilBERT`](../DotnetAILab.ModelGarden.Classification.SentimentDistilBERT/) | Binary sentiment (POSITIVE/NEGATIVE) | When you only need positive vs. negative polarity — faster and lighter (~268 MB) |
| [`DotnetAILab.ModelGarden.Classification.ZeroShotDeBERTa`](../DotnetAILab.ModelGarden.Classification.ZeroShotDeBERTa/) | Zero-shot classification with custom labels | When you need custom emotion categories or labels beyond the fixed 28 — no fine-tuning required |

## References

- **GoEmotions Dataset**: Demszky et al., ["GoEmotions: A Dataset of Fine-Grained Emotions"](https://arxiv.org/abs/2005.00547) (ACL 2020)
- **RoBERTa**: Liu et al., ["RoBERTa: A Robustly Optimized BERT Pretraining Approach"](https://arxiv.org/abs/1907.11692) (2019)
- **HuggingFace Model Card**: [`lquint/roberta-base-go_emotions-onnx`](https://huggingface.co/lquint/roberta-base-go_emotions-onnx)
- **ML.NET Documentation**: [ML.NET Overview](https://learn.microsoft.com/dotnet/machine-learning/)
