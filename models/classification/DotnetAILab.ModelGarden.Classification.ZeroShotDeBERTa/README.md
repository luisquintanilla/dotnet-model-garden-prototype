# DotnetAILab.ModelGarden.Classification.ZeroShotDeBERTa

> Zero-shot text classification with **any custom labels** — no fine-tuning, no training data, just describe your categories and classify. Powered by DeBERTa-v3 and natural language inference.

## Overview

Most classification models require you to choose from a fixed set of labels decided at training time. But what if you need to classify support tickets into categories specific to *your* product? Or moderate content with rules unique to *your* community? Zero-shot classification solves this by letting you define **any labels you want at inference time** — no model retraining, no labeled datasets, no ML expertise required. Just provide your text and a list of candidate labels, and the model scores each one.

This works through a clever application of **Natural Language Inference (NLI)**. Instead of directly predicting categories, the model evaluates whether a given text *entails* a hypothesis like "This text is about {label}." For each candidate label, the model produces three scores: **entailment** (the text supports the hypothesis), **contradiction** (the text contradicts it), and **neutral** (no clear relationship). The entailment score becomes the classification confidence for that label. This approach means the model can classify text into categories it has never seen during training.

The underlying DeBERTa-v3-base model is trained on three major NLI datasets — MNLI, FEVER, and ANLI — giving it robust natural language understanding across diverse domains. The ONNX model (~739 MB) downloads transparently on first use. Like all .NET Model Garden packages, you get a pure C# API with no Python dependencies, no external services, and no GPU required.

## Model Details

| Property | Value |
|---|---|
| **Architecture** | DeBERTa-v3-base (12-layer, 768-hidden, 12-heads) |
| **Task** | Zero-shot classification via natural language inference (NLI) |
| **Training Datasets** | MNLI + FEVER + ANLI |
| **NLI Labels** | 3: `contradiction`, `neutral`, `entailment` |
| **Max Token Length** | 256 |
| **Model Format** | ONNX |
| **Model Size** | ~739 MB |
| **HuggingFace Repo** | [`lquint/DeBERTa-v3-base-mnli-fever-anli-onnx`](https://huggingface.co/lquint/DeBERTa-v3-base-mnli-fever-anli-onnx) |
| **NuGet Package** | `DotnetAILab.ModelGarden.Classification.ZeroShotDeBERTa` |
| **Target Framework** | .NET 10.0 |

## Installation

```bash
dotnet add package DotnetAILab.ModelGarden.Classification.ZeroShotDeBERTa
```

**Dependencies** (pulled in automatically):

- `ModelPackages` (0.1.0-preview.14) — model download, caching, and verification
- `MLNet.TextInference.Onnx` (0.1.0-preview.1) — ONNX text classification runtime

## Quick Start

```csharp
using DotnetAILab.ModelGarden.Classification.ZeroShotDeBERTa;

// Create the NLI classifier — model auto-downloads on first call (~739 MB), cached thereafter
var classifier = await ZeroShotDeBERTaModel.CreateClassifierAsync();

// Classify: pair your text with a hypothesis for each candidate label
var text = "The new iPhone has an incredible camera and amazing battery life.";

// For zero-shot: create premise-hypothesis pairs like "text </s></s> This text is about {label}"
var candidates = new[] { "technology", "sports", "politics", "food" };
foreach (var label in candidates)
{
    var input = $"{text}</s></s>This text is about {label}.";
    var results = classifier.Classify([input]);
    Console.WriteLine($"  {label}: {results[0].PredictedLabel} ({results[0].Confidence:P1})");
}
```

## How Zero-Shot Classification Works

Traditional classifiers learn a fixed mapping from text → labels during training. Zero-shot classification takes a fundamentally different approach using **Natural Language Inference (NLI)**:

1. **Formulate a hypothesis** for each candidate label: `"This text is about {label}."`
2. **Pair** the input text (premise) with each hypothesis
3. **Run NLI inference** — the model predicts whether the premise *entails*, *contradicts*, or is *neutral* toward the hypothesis
4. **Use the entailment score** as the confidence that the text belongs to that category
5. **Rank** candidates by entailment score to find the best-matching label

```
Input:  "The stock market crashed 5% today amid trade war fears."

Hypothesis 1: "This text is about finance."      → entailment: 0.95 ✓
Hypothesis 2: "This text is about sports."        → contradiction: 0.92
Hypothesis 3: "This text is about technology."    → neutral: 0.68
```

This is powerful because the model's NLI training generalizes to *any* label you can describe in natural language — you're not limited to categories seen during training.

## API Reference

### `ZeroShotDeBERTaModel` (static class)

**Namespace:** `DotnetAILab.ModelGarden.Classification.ZeroShotDeBERTa`

#### Properties

```csharp
public static readonly string[] Labels;
```

The 3 NLI output labels: `contradiction`, `neutral`, `entailment`. These are the raw model outputs — for zero-shot classification, you'll primarily use the **entailment** score as the classification confidence.

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
// For NLI, pass premise-hypothesis pairs separated by </s></s>
var results = classifier.Classify(["premise text</s></s>hypothesis text"]);
```

Each result exposes:
- `PredictedLabel` — one of `"contradiction"`, `"neutral"`, or `"entailment"`
- `Confidence` — confidence score for the predicted label (0.0–1.0)

## Inputs & Outputs

### Input

| Parameter | Type | Description |
|---|---|---|
| `texts` | `string[]` | Array of premise-hypothesis pairs. For zero-shot: `"{text}</s></s>This text is about {label}."` |

For zero-shot classification, construct one input per candidate label. Texts are tokenized up to 256 tokens and processed in batches of 8.

### Output

Each classification result includes:

| Field | Type | Description |
|---|---|---|
| `PredictedLabel` | `string` | One of `"contradiction"`, `"neutral"`, or `"entailment"` |
| `Confidence` | `float` | Confidence score (0.0–1.0) for the predicted label |

### NLI Label Descriptions

| Label | Meaning | Zero-Shot Interpretation |
|---|---|---|
| `entailment` | The premise supports/implies the hypothesis | The text **matches** the candidate label |
| `neutral` | The premise neither supports nor contradicts the hypothesis | The text is **ambiguous** for this label |
| `contradiction` | The premise contradicts the hypothesis | The text **does not match** the candidate label |

## Use Cases

1. **Content Moderation** — Define categories like "harassment", "spam", "misinformation", "safe" and classify user-generated content instantly — no need to collect and label thousands of examples first.

2. **Topic Routing** — Route customer support tickets to the right team (billing, technical, shipping, general inquiry) using your exact department names as labels.

3. **Intent Detection for Chatbots** — Classify user messages into intents ("book_flight", "cancel_reservation", "check_status", "speak_to_agent") without building a training dataset for each new intent.

4. **Dynamic News Categorization** — Classify articles into arbitrary topic taxonomies that change with current events — no retraining needed when new categories emerge.

5. **Email Classification** — Sort incoming emails into custom folders or priority levels ("urgent", "informational", "action_required", "spam") tailored to your organization's workflow.

6. **Compliance & Policy Screening** — Check documents against custom policy categories that are unique to your industry or organization without building specialized models for each policy.

7. **Multilingual Topic Detection** — While the model is English-focused, the NLI approach can be adapted to classify text in languages where DeBERTa has some coverage, using English hypotheses.

8. **A/B Test Content Tagging** — Quickly tag marketing content with custom attributes ("emotional_appeal", "data_driven", "urgency", "social_proof") for A/B testing analysis.

## Limitations & Considerations

- **Latency vs. Flexibility Trade-off**: Zero-shot classification requires one NLI inference per candidate label. Classifying into 10 categories means 10 forward passes — this is slower than a single-pass fixed-label classifier for high-throughput scenarios.
- **Hypothesis Sensitivity**: Results are sensitive to how you phrase the hypothesis template. "This text is about {label}" works well generally, but domain-specific phrasing (e.g., "This email requires {action}") may improve accuracy.
- **English Primarily**: The model performs best on English text. While DeBERTa has some multilingual capability, accuracy degrades significantly for non-English languages.
- **No Fine-Grained Scores by Default**: The API returns the top NLI label and confidence. For full zero-shot ranking, you need to run inference for each candidate label and compare entailment scores across them.
- **Model Size**: At ~739 MB, this is the largest of the classification models. Ensure adequate disk space and plan for the initial download time.
- **Max Sequence Length**: Combined premise + hypothesis length is limited to 256 tokens. Very long texts may need truncation or splitting.
- **Semantic Understanding**: The model classifies based on semantic meaning, not keywords. "Apple released new products" will correctly classify as "technology" over "fruit", but edge cases exist.

## Example: Support Ticket Routing

```csharp
using DotnetAILab.ModelGarden.Classification.ZeroShotDeBERTa;

var classifier = await ZeroShotDeBERTaModel.CreateClassifierAsync();

var ticket = "I was charged twice for my subscription and I need a refund immediately.";
var departments = new[] { "billing", "technical_support", "shipping", "account_management", "general_inquiry" };

Console.WriteLine($"Ticket: \"{ticket}\"\n");
Console.WriteLine("Department Scores:");

string bestLabel = "";
float bestScore = 0f;

foreach (var dept in departments)
{
    // Construct the NLI premise-hypothesis pair
    var input = $"{ticket}</s></s>This text is about {dept.Replace('_', ' ')}.";
    var results = classifier.Classify([input]);

    var label = results[0].PredictedLabel;
    var confidence = results[0].Confidence;
    var indicator = label == "entailment" ? "✓" : " ";

    Console.WriteLine($"  {indicator} {dept,-25} → {label} ({confidence:P1})");

    // Track best entailment match
    if (label == "entailment" && confidence > bestScore)
    {
        bestScore = confidence;
        bestLabel = dept;
    }
}

Console.WriteLine($"\n→ Route to: {bestLabel} (confidence: {bestScore:P1})");

// Example output:
// Ticket: "I was charged twice for my subscription and I need a refund immediately."
//
// Department Scores:
//   ✓ billing                   → entailment (93.2%)
//     technical_support          → contradiction (88.5%)
//     shipping                   → contradiction (91.7%)
//   ✓ account_management         → entailment (62.1%)
//     general_inquiry            → neutral (54.3%)
//
// → Route to: billing (confidence: 93.2%)
```

## Related Models

| Package | Task | When to Use |
|---|---|---|
| [`DotnetAILab.ModelGarden.Classification.SentimentDistilBERT`](../DotnetAILab.ModelGarden.Classification.SentimentDistilBERT/) | Binary sentiment (POSITIVE/NEGATIVE) | When you specifically need sentiment polarity — much faster with a fixed 2-label classifier (~268 MB) |
| [`DotnetAILab.ModelGarden.Classification.EmotionRoBERTa`](../DotnetAILab.ModelGarden.Classification.EmotionRoBERTa/) | 28-label emotion detection | When you need fine-grained emotions from a fixed taxonomy — single-pass classification is faster for this specific task |

**When to choose zero-shot over fixed-label models:**
- Your labels are **unique to your domain** and not covered by existing models
- Your label taxonomy **changes frequently** and retraining isn't practical
- You're **prototyping** and want to iterate on categories quickly
- You need to classify into **many niche categories** simultaneously

## References

- **DeBERTa-v3**: He et al., ["DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing"](https://arxiv.org/abs/2111.09543) (ICLR 2023)
- **MNLI**: Williams et al., ["A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference"](https://arxiv.org/abs/1704.05426) (NAACL 2018)
- **ANLI**: Nie et al., ["Adversarial NLI: A New Benchmark for Natural Language Understanding"](https://arxiv.org/abs/1910.14599) (ACL 2020)
- **FEVER**: Thorne et al., ["FEVER: a Large-scale Dataset for Fact Extraction and VERification"](https://arxiv.org/abs/1803.05355) (NAACL 2018)
- **Zero-Shot Classification with NLI**: Yin et al., ["Benchmarking Zero-shot Text Classification"](https://arxiv.org/abs/1909.00161) (EMNLP 2019)
- **HuggingFace Model Card**: [`lquint/DeBERTa-v3-base-mnli-fever-anli-onnx`](https://huggingface.co/lquint/DeBERTa-v3-base-mnli-fever-anli-onnx)
- **ML.NET Documentation**: [ML.NET Overview](https://learn.microsoft.com/dotnet/machine-learning/)
