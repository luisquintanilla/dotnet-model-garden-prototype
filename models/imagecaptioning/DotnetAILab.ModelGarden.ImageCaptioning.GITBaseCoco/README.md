# DotnetAILab.ModelGarden.ImageCaptioning.GITBaseCoco

GIT-Base (COCO) image captioning model package. Generates natural language captions for images.

## Model Details

- **Model**: [microsoft/git-base-coco](https://huggingface.co/microsoft/git-base-coco)
- **Task**: Image Captioning
- **Architecture**: GIT (Generative Image-to-Text Transformer)
- **Size**: ~707 MB total (encoder + decoder + vocab)
- **Output**: Natural language captions
- **Interfaces**: `OnnxImageCaptioningTransformer` (direct) or `IChatClient` (MEAI)

## Quick Start

```csharp
using DotnetAILab.ModelGarden.ImageCaptioning.GITBaseCoco;
using Microsoft.ML.Data;

var captioner = await GITBaseCocoModel.CreateCaptionerAsync();
using var image = MLImage.CreateFromFile("photo.jpg");
var caption = captioner.GenerateCaption(image);
Console.WriteLine($"Caption: {caption}");
```

## API

| Method | Description |
|--------|-------------|
| `CreateCaptionerAsync()` | Creates an image captioner (downloads model on first use) |
| `CreateChatClientAsync()` | Creates an MEAI IChatClient for conversational image understanding |
| `EnsureFilesAsync()` | Downloads model files and returns local paths |
| `GetModelInfoAsync()` | Returns model metadata |
| `VerifyModelAsync()` | Verifies cached model integrity |
