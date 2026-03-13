# DotnetAILab.ModelGarden.ImageSegmentation.SegFormerB0

SegFormer-B0 image segmentation model package. Segments images into semantic regions (ADE20K 150 classes).

## Model Details

- **Model**: [nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
- **Task**: Semantic Image Segmentation
- **Architecture**: SegFormer-B0
- **Size**: ~15 MB (ONNX)
- **Output**: Pixel-level class masks (150 ADE20K categories)

## Quick Start

```csharp
using DotnetAILab.ModelGarden.ImageSegmentation.SegFormerB0;
using Microsoft.ML.Data;

var segmenter = await SegFormerB0Model.CreateSegmenterAsync();
using var image = MLImage.CreateFromFile("photo.jpg");
var mask = segmenter.Segment(image);

Console.WriteLine($"Mask: {mask.Width}x{mask.Height}, {mask.ClassIds.Distinct().Count()} classes");
```

## API

| Method | Description |
|--------|-------------|
| `CreateSegmenterAsync()` | Creates a segmenter (downloads model on first use) |
| `EnsureFilesAsync()` | Downloads model files and returns local paths |
| `GetModelInfoAsync()` | Returns model metadata |
| `VerifyModelAsync()` | Verifies cached model integrity |
