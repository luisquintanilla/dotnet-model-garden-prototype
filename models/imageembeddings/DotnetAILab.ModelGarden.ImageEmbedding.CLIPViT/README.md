# DotnetAILab.ModelGarden.ImageEmbedding.CLIPViT

CLIP ViT-Base image embedding model package. Generates 512-dim L2-normalized embeddings from images.

## Model Details

- **Model**: [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- **Task**: Image Embedding
- **Architecture**: CLIP ViT-Base (patch size 32)
- **Size**: ~353 MB (ONNX)
- **Output**: 512-dimensional L2-normalized float embeddings
- **Interface**: `IEmbeddingGenerator<MLImage, Embedding<float>>` (MEAI)

## Quick Start

```csharp
using DotnetAILab.ModelGarden.ImageEmbedding.CLIPViT;
using Microsoft.Extensions.AI;
using Microsoft.ML.Data;
using System.Numerics.Tensors;

var generator = await CLIPViTModel.CreateEmbeddingGeneratorAsync();
using var img1 = MLImage.CreateFromFile("cat.jpg");
using var img2 = MLImage.CreateFromFile("dog.jpg");

var embeddings = await generator.GenerateAsync(new[] { img1, img2 });
float similarity = TensorPrimitives.CosineSimilarity(
    embeddings[0].Vector.Span, embeddings[1].Vector.Span);
Console.WriteLine($"Similarity: {similarity:F4}");
```

## API

| Method | Description |
|--------|-------------|
| `CreateEmbeddingGeneratorAsync()` | Creates an MEAI embedding generator (downloads model on first use) |
| `EnsureFilesAsync()` | Downloads model files and returns local paths |
| `GetModelInfoAsync()` | Returns model metadata |
| `VerifyModelAsync()` | Verifies cached model integrity |
```
