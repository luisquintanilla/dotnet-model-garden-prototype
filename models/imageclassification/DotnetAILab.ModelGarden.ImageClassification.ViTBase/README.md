# DotnetAILab.ModelGarden.ImageClassification.ViTBase

ViT-Base image classification model package. Classifies images into 1000 ImageNet categories.

## Model Details

- **Model**: [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)
- **Task**: Image Classification
- **Architecture**: Vision Transformer (ViT-Base, patch size 16, input 224×224)
- **Size**: ~346 MB (ONNX)
- **Output**: Top-5 ImageNet class predictions with probabilities

## Quick Start

```csharp
using DotnetAILab.ModelGarden.ImageClassification.ViTBase;
using Microsoft.ML.Data;

var classifier = await ViTBaseModel.CreateClassifierAsync();
using var image = MLImage.CreateFromFile("photo.jpg");
var results = classifier.Classify(image);

foreach (var (label, probability) in results)
    Console.WriteLine($"{label}: {probability:P1}");
```

## API

| Method | Description |
|--------|-------------|
| `CreateClassifierAsync()` | Creates an image classifier (downloads model on first use) |
| `EnsureFilesAsync()` | Downloads model files and returns local paths |
| `GetModelInfoAsync()` | Returns model metadata |
| `VerifyModelAsync()` | Verifies cached model integrity |
