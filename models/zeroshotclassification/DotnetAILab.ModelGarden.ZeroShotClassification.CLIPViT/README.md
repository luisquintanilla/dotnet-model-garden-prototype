# DotnetAILab.ModelGarden.ZeroShotClassification.CLIPViT

CLIP ViT-Base zero-shot image classification model package. Classifies images against arbitrary text labels without task-specific training.

## Model Details

- **Model**: [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- **Task**: Zero-Shot Image Classification
- **Architecture**: CLIP (Vision Transformer + Text Transformer)
- **Size**: ~608 MB total (vision model + text model + tokenizer files)
- **Output**: Probability distribution over user-defined text labels

## Quick Start

```csharp
using DotnetAILab.ModelGarden.ZeroShotClassification.CLIPViT;
using Microsoft.ML.Data;

var labels = new[] { "a photo of a cat", "a photo of a dog", "a photo of a bird" };
var classifier = await ZeroShotCLIPViTModel.CreateClassifierAsync(labels);
using var image = MLImage.CreateFromFile("photo.jpg");
var results = classifier.Classify(image);

foreach (var (label, probability) in results)
    Console.WriteLine($"{label}: {probability:P1}");
```

## API

| Method | Description |
|--------|-------------|
| `CreateClassifierAsync(labels)` | Creates a zero-shot classifier for given labels (downloads model on first use) |
| `EnsureFilesAsync()` | Downloads model files and returns local paths |
| `GetModelInfoAsync()` | Returns model metadata |
| `VerifyModelAsync()` | Verifies cached model integrity |
```
