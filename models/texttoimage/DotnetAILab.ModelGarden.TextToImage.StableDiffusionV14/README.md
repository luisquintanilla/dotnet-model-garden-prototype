# DotnetAILab.ModelGarden.TextToImage.StableDiffusionV14

Stable Diffusion v1.4 text-to-image generation model package. Generates images from text prompts.

## Model Details

- **Model**: [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)
- **Task**: Text-to-Image Generation
- **Architecture**: Stable Diffusion (UNet + VAE + Text Encoder)
- **Size**: ~4 GB total (text encoder + UNet + VAE decoder + tokenizer)
- **Output**: 512x512 generated images
- **Runtime**: ONNX GenAI

## Quick Start

```csharp
using DotnetAILab.ModelGarden.TextToImage.StableDiffusionV14;

var generator = await StableDiffusionV14Model.CreateGeneratorAsync();
using var image = generator.Generate("a cat sitting on a beach at sunset", seed: 42);
Console.WriteLine($"Generated: {image.Width}x{image.Height}");
```

## API

| Method | Description |
|--------|-------------|
| `CreateGeneratorAsync()` | Creates an image generator (downloads model on first use, ~4 GB) |
| `EnsureFilesAsync()` | Downloads model files and returns local paths |
| `GetModelInfoAsync()` | Returns model metadata |
| `VerifyModelAsync()` | Verifies cached model integrity |
