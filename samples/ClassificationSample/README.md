# Classification Sample

> Perform sentiment analysis on text using a fine-tuned [DistilBERT](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) model.

## Prerequisites

- [.NET 10 SDK (preview)](https://dotnet.microsoft.com/download/dotnet/10.0)
- Internet connection (the model downloads automatically on first use)

## What This Sample Demonstrates

Text classification assigns a label to a piece of text. This sample uses a DistilBERT model fine-tuned on the SST-2 dataset for **binary sentiment analysis** (POSITIVE / NEGATIVE). It shows how to:

1. Create a sentiment classifier with a single async call.
2. Classify multiple texts in a batch.
3. Read the predicted label and confidence score for each result.

## Running the Sample

```sh
cd dotnet-model-garden-prototype
dotnet run --project samples/ClassificationSample
```

## Expected Output

```
=== Model Garden: Classification Sample ===

Creating sentiment classifier (model downloads on first use)...
Classifier ready!

Available labels: NEGATIVE, POSITIVE

Classifying texts:
  "This movie was absolutely wonderful!"
    → POSITIVE (confidence: 100.0%)
  "The food was terrible and the service was slow."
    → NEGATIVE (confidence: 99.9%)
  "I love programming in C# with ML.NET"
    → POSITIVE (confidence: 61.3%)
  "The weather is okay today."
    → POSITIVE (confidence: 99.6%)

Done!
```

## Code Walkthrough

1. **Create the classifier** — `SentimentDistilBERTModel.CreateClassifierAsync()` downloads the ONNX model on first run and returns a ready-to-use classifier.
2. **Inspect available labels** — `SentimentDistilBERTModel.Labels` exposes the set of labels the model can predict (`NEGATIVE`, `POSITIVE`).
3. **Prepare input texts** — A string array of sentences with varying sentiment.
4. **Run classification** — `classifier.Classify(texts)` tokenizes each text, runs inference, and returns a result per input containing the predicted label and confidence.
5. **Display results** — Prints each text with its predicted sentiment and confidence percentage.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Sentiment Analysis** | A text classification task that determines whether a piece of text expresses positive or negative sentiment. |
| **Confidence Score** | The model's estimated probability (0–100 %) that its predicted label is correct. Higher values indicate stronger predictions. |
| **DistilBERT** | A smaller, faster variant of BERT that retains 97 % of BERT's language understanding while being 60 % faster. |

## Next Steps

- **Try different texts** — Experiment with product reviews, tweets, or customer feedback.
- **Threshold-based filtering** — Use the confidence score to flag uncertain predictions for human review.
- **Multi-class classification** — Look for model packages that support more than two labels (e.g., emotion detection).
- Explore the [EmbeddingSample](../EmbeddingSample/) to see how text can be represented as vectors instead of labels.

## Model Package References

- [`DotnetAILab.ModelGarden.Classification.SentimentDistilBERT`](../../models/classification/DotnetAILab.ModelGarden.Classification.SentimentDistilBERT/) — DistilBERT sentiment model package
- [Hugging Face: distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
