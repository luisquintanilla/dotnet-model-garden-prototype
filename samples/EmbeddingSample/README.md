# Embedding Sample

> Generate text embeddings and compute semantic similarity using the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model via the `Microsoft.Extensions.AI` `IEmbeddingGenerator` interface.

## Prerequisites

- [.NET 10 SDK (preview)](https://dotnet.microsoft.com/download/dotnet/10.0)
- Internet connection (the model downloads automatically on first use)

## What This Sample Demonstrates

Text embeddings map sentences into a high-dimensional vector space where semantically similar texts are close together. This sample:

1. Creates an `IEmbeddingGenerator<string, Embedding<float>>` backed by the **all-MiniLM-L6-v2** ONNX model.
2. Generates 384-dimensional embeddings for a set of input texts.
3. Computes **cosine similarity** between every pair to show how the model captures semantic relationships.

## Running the Sample

```sh
cd dotnet-model-garden-prototype
dotnet run --project samples/EmbeddingSample
```

## Expected Output

```
=== Model Garden: Embedding Sample ===

Creating embedding generator (model downloads on first use)...
Generator ready!

Generating embeddings for 4 texts...
Embeddings generated! Dimension: 384

Cosine Similarities:
  "The cat sat on the mat" vs "A kitten rested on the rug"
    → 0.8256 (similar meaning)
  "The cat sat on the mat" vs "Stock prices rose sharply today"
    → 0.0847 (different topics)

Done!
```

## Code Walkthrough

1. **Create the embedding generator** — `AllMiniLMModel.CreateEmbeddingGeneratorAsync()` downloads the ONNX model on first run and returns an `IEmbeddingGenerator<string, Embedding<float>>`.
2. **Prepare input texts** — A string array of sentences covering related and unrelated topics.
3. **Generate embeddings** — `generator.GenerateAsync(texts)` tokenizes each text and runs inference, producing one 384-dimensional float vector per input.
4. **Compute cosine similarity** — `TensorPrimitives.CosineSimilarity` from `System.Numerics.Tensors` measures how close two vectors are (1.0 = identical, 0.0 = orthogonal).
5. **Display results** — Prints all pairwise similarity scores so you can see which texts the model considers semantically related.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Text Embedding** | A fixed-length numeric vector that captures the semantic meaning of a sentence. |
| **Cosine Similarity** | A metric ranging from −1 to 1 that measures the angle between two vectors. Values closer to 1 indicate higher similarity. |
| **IEmbeddingGenerator** | The `Microsoft.Extensions.AI` abstraction for embedding generators, allowing model-agnostic code. |

## Next Steps

- **Try different texts** — Replace the sample sentences with your own to explore how the model handles different domains.
- **Build a semantic search engine** — Store embeddings in a vector database and retrieve the most similar documents for a query.
- **Compare models** — Swap in a different embedding model package to compare quality and performance.
- Explore the [AudioEmbeddingSample](../AudioEmbeddingSample/) for embedding audio data.

## Model Package References

- [`DotnetAILab.ModelGarden.Embeddings.AllMiniLM`](../../models/embeddings/DotnetAILab.ModelGarden.Embeddings.AllMiniLM/) — all-MiniLM-L6-v2 ONNX model package
- [Hugging Face: sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
