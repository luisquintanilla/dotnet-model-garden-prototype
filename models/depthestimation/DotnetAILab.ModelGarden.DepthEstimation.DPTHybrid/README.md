# DotnetAILab.ModelGarden.DepthEstimation.DPTHybrid

DPT-Hybrid (MiDaS) depth estimation model package. Estimates relative depth from monocular images.

## Model Details

- **Model**: [Intel/dpt-hybrid-midas](https://huggingface.co/Intel/dpt-hybrid-midas)
- **Task**: Monocular Depth Estimation
- **Architecture**: DPT-Hybrid (MiDaS)
- **Size**: ~533 MB (ONNX)
- **Output**: Dense depth maps with relative depth values

## Quick Start

```csharp
using DotnetAILab.ModelGarden.DepthEstimation.DPTHybrid;
using Microsoft.ML.Data;

var estimator = await DPTHybridModel.CreateEstimatorAsync();
using var image = MLImage.CreateFromFile("photo.jpg");
var depthMap = estimator.Estimate(image);

Console.WriteLine($"Depth: {depthMap.Width}x{depthMap.Height}, range [{depthMap.MinDepth:F2}, {depthMap.MaxDepth:F2}]");
```

## API

| Method | Description |
|--------|-------------|
| `CreateEstimatorAsync()` | Creates a depth estimator (downloads model on first use) |
| `EnsureFilesAsync()` | Downloads model files and returns local paths |
| `GetModelInfoAsync()` | Returns model metadata |
| `VerifyModelAsync()` | Verifies cached model integrity |
