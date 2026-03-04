# DotnetAILab.ModelGarden.Classification.SentimentDistilBERT

> Binary sentiment classification (POSITIVE/NEGATIVE) using DistilBERT fine-tuned on the SST-2 benchmark — fast, lightweight, and production-ready.

## Overview

Sentiment analysis is one of the most widely deployed NLP tasks in production systems. Whether you're monitoring product reviews, analyzing customer support tickets, or building social media dashboards, the core question is often simple: *is this text positive or negative?* The SentimentDistilBERT model package provides a fast, accurate answer using DistilBERT — a distilled version of BERT that retains 97% of BERT's language understanding while being 60% faster and 40% smaller.

This model is fine-tuned on the Stanford Sentiment Treebank v2 (SST-2), a benchmark dataset of movie reviews labeled as positive or negative. Despite its training domain, the model generalizes well to many real-world text classification scenarios including product reviews, social media posts, and customer feedback. The binary classification approach keeps things simple: every input text receives a `POSITIVE` or `NEGATIVE` label with a confidence score.

As with all .NET Model Garden packages, the ONNX model binary (~268 MB) downloads automatically on first use and is cached locally. You get a clean C# API with no Python runtime, no REST API calls, and no GPU required — just add the NuGet package and start classifying.

## Model Details

| Property | Value |
|---|---|
| **Architecture** | DistilBERT (6-layer, 768-hidden, 12-heads) |
| **Task** | Binary text classification (sentiment) |
| **Training Dataset** | SST-2 (Stanford Sentiment Treebank v2) |
| **Labels** | 2: `NEGATIVE`, `POSITIVE` |
| **Max Token Length** | 128 |
| **Model Format** | ONNX |
| **Model Size** | ~268 MB |
| **HuggingFace Repo** | [`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) |
| **NuGet Package** | `DotnetAILab.ModelGarden.Classification.SentimentDistilBERT` |
| **Target Framework** | .NET 10.0 |

## Installation

```bash
dotnet add package DotnetAILab.ModelGarden.Classification.SentimentDistilBERT
```

**Dependencies** (pulled in automatically):

- `ModelPackages` (0.1.0-preview.14) — model download, caching, and verification
- `MLNet.TextInference.Onnx` (0.1.0-preview.1) — ONNX text classification runtime

## Quick Start

```csharp
using DotnetAILab.ModelGarden.Classification.SentimentDistilBERT;

// Create classifier — model auto-downloads on first call (~268 MB), cached thereafter
var classifier = await SentimentDistilBERTModel.CreateClassifierAsync();

// Classify text
var results = classifier.Classify(["This product is absolutely wonderful!"]);

Console.WriteLine($"Sentiment: {results[0].PredictedLabel}");
Console.WriteLine($"Confidence: {results[0].Confidence:P1}");
// Output: Sentiment: POSITIVE
//         Confidence: 99.8%
```

## API Reference

### `SentimentDistilBERTModel` (static class)

**Namespace:** `DotnetAILab.ModelGarden.Classification.SentimentDistilBERT`

#### Properties

```csharp
public static readonly string[] Labels;
```

The 2 classification labels: `NEGATIVE`, `POSITIVE`.

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
- `PredictedLabel` — either `"POSITIVE"` or `"NEGATIVE"`
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
| `PredictedLabel` | `string` | Either `"POSITIVE"` or `"NEGATIVE"` |
| `Confidence` | `float` | Confidence score (0.0–1.0) for the predicted label |

### Label Descriptions

| Label | Meaning |
|---|---|
| `POSITIVE` | The text expresses a favorable, approving, or optimistic sentiment |
| `NEGATIVE` | The text expresses an unfavorable, critical, or pessimistic sentiment |

> **Note**: Truly neutral text (e.g., "The meeting is at 3pm") will still receive one of the two labels. If neutral detection is important for your use case, consider thresholding on the confidence score — predictions with confidence below ~0.6 often indicate neutral or ambiguous text.

## Use Cases

1. **Product Review Analysis** — Automatically classify thousands of product reviews as positive or negative to identify trending issues or popular features without reading each one.

2. **Social Media Sentiment Tracking** — Monitor brand mentions on social platforms in real-time, aggregating sentiment over time to detect PR crises or positive viral moments.

3. **Customer Support Triage** — Route angry or negative tickets to senior agents while directing positive feedback to marketing or product teams.

4. **App Store Review Monitoring** — Classify app reviews at scale to quickly surface negative feedback after new releases and track sentiment trends version-over-version.

5. **Email & Survey Analysis** — Gauge overall sentiment in open-ended survey responses or customer emails to complement quantitative satisfaction scores.

6. **Financial News Sentiment** — Classify news headlines and articles to build sentiment signals for trading strategies or risk monitoring dashboards.

7. **Content Recommendation Filtering** — Deprioritize or flag content with negative sentiment in recommendation feeds to improve user experience.

## Limitations & Considerations

- **Binary Only**: This model classifies text as strictly POSITIVE or NEGATIVE. For nuanced emotion detection (joy, anger, sadness, etc.), use the [EmotionRoBERTa](../DotnetAILab.ModelGarden.Classification.EmotionRoBERTa/) package instead.
- **No Neutral Class**: The model always predicts one of two labels. Low-confidence predictions (< 0.6) can serve as a proxy for neutral, but this requires application-level thresholding.
- **English Only**: Trained on English text. Non-English input will produce unreliable results.
- **Training Domain**: Fine-tuned on movie review sentences (SST-2). While it generalizes well, domain-specific jargon (e.g., medical, legal) may reduce accuracy.
- **Token Limit**: Texts are truncated at 128 tokens (~100 words). For longer documents, consider classifying individual sentences or paragraphs and aggregating.
- **Sarcasm**: Sarcastic text ("What a *great* day to have my flight cancelled") may be misclassified as positive due to surface-level positive words.
- **First-run Download**: The ~268 MB model downloads on first use. Plan for this in CI/CD pipelines and air-gapped environments.

## Example: Product Review Dashboard

```csharp
using DotnetAILab.ModelGarden.Classification.SentimentDistilBERT;

var classifier = await SentimentDistilBERTModel.CreateClassifierAsync();

Console.WriteLine("Available labels: " + string.Join(", ", SentimentDistilBERTModel.Labels));

var reviews = new[]
{
    "This movie was absolutely wonderful!",
    "The food was terrible and the service was slow.",
    "I love programming in C# with ML.NET",
    "The weather is okay today.",
    "Worst experience ever. Will never come back.",
    "The staff was friendly and the room was clean."
};

var results = classifier.Classify(reviews);

int positive = 0, negative = 0;
for (int i = 0; i < reviews.Length; i++)
{
    Console.WriteLine($"\"{reviews[i]}\"");
    Console.WriteLine($"  → {results[i].PredictedLabel} (confidence: {results[i].Confidence:P1})");

    if (results[i].PredictedLabel == "POSITIVE") positive++;
    else negative++;
}

Console.WriteLine($"\nSummary: {positive} positive, {negative} negative");
Console.WriteLine($"Overall sentiment ratio: {(double)positive / reviews.Length:P0} positive");

// Example output:
// "This movie was absolutely wonderful!"
//   → POSITIVE (confidence: 99.9%)
// "The food was terrible and the service was slow."
//   → NEGATIVE (confidence: 99.8%)
// "I love programming in C# with ML.NET"
//   → POSITIVE (confidence: 99.9%)
// "The weather is okay today."
//   → POSITIVE (confidence: 65.2%)
// "Worst experience ever. Will never come back."
//   → NEGATIVE (confidence: 99.9%)
// "The staff was friendly and the room was clean."
//   → POSITIVE (confidence: 99.9%)
//
// Summary: 4 positive, 2 negative
// Overall sentiment ratio: 67% positive
```

## Related Models

| Package | Task | When to Use |
|---|---|---|
| [`DotnetAILab.ModelGarden.Classification.EmotionRoBERTa`](../DotnetAILab.ModelGarden.Classification.EmotionRoBERTa/) | 28-label emotion detection | When you need granular emotions (joy, anger, gratitude, etc.) beyond simple positive/negative |
| [`DotnetAILab.ModelGarden.Classification.ZeroShotDeBERTa`](../DotnetAILab.ModelGarden.Classification.ZeroShotDeBERTa/) | Zero-shot classification with custom labels | When you need custom sentiment categories (e.g., "very positive", "slightly negative", "neutral") without retraining |

## References

- **DistilBERT**: Sanh et al., ["DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"](https://arxiv.org/abs/1910.01108) (NeurIPS 2019 Workshop)
- **SST-2 Benchmark**: Socher et al., ["Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank"](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf) (EMNLP 2013)
- **HuggingFace Model Card**: [`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
- **ML.NET Documentation**: [ML.NET Overview](https://learn.microsoft.com/dotnet/machine-learning/)
