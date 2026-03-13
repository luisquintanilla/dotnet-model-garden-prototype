# DotnetAILab.ModelGarden.SegmentAnything.SAM2HieraTiny

SAM2 Hiera-Tiny segment anything model package. Segments any object in an image given point or box prompts.

## Model Details

- **Model**: [facebook/sam2-hiera-tiny](https://huggingface.co/facebook/sam2-hiera-tiny)
- **Task**: Promptable Image Segmentation
- **Architecture**: SAM2 Hiera-Tiny (encoder + decoder)
- **Size**: ~126 MB total (encoder + decoder)
- **Output**: Binary segmentation masks with IoU scores
- **Prompts**: Point coordinates or bounding box

## Quick Start

```csharp
using DotnetAILab.ModelGarden.SegmentAnything.SAM2HieraTiny;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.SegmentAnything;

var transformer = await SAM2HieraTinyModel.CreateTransformerAsync();
using var image = MLImage.CreateFromFile("photo.jpg");

var prompt = SegmentAnythingPrompt.FromPoint(256f, 256f);
var result = transformer.Segment(image, prompt);
Console.WriteLine($"Masks: {result.NumMasks}, Best IoU: {result.GetBestIoU():F4}");
```

## API

| Method | Description |
|--------|-------------|
| `CreateTransformerAsync()` | Creates a SAM2 transformer (downloads model on first use) |
| `EnsureFilesAsync()` | Downloads model files and returns local paths |
| `GetModelInfoAsync()` | Returns model metadata |
| `VerifyModelAsync()` | Verifies cached model integrity |
