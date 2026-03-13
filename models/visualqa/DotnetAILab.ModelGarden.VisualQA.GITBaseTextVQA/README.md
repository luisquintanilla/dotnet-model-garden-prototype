# DotnetAILab.ModelGarden.VisualQA.GITBaseTextVQA

GIT-Base (TextVQA) visual question answering model package. Answers questions about image content.

## Model Details

- **Model**: [microsoft/git-base-textvqa](https://huggingface.co/microsoft/git-base-textvqa)
- **Task**: Visual Question Answering
- **Architecture**: GIT (Generative Image-to-Text Transformer), fine-tuned for VQA
- **Size**: ~709 MB total (encoder + decoder + vocab)
- **Output**: Natural language answers to questions about images
- **Interfaces**: `OnnxImageCaptioningTransformer` (direct) or `IChatClient` (MEAI)

## Quick Start

```csharp
using DotnetAILab.ModelGarden.VisualQA.GITBaseTextVQA;
using Microsoft.ML.Data;

var transformer = await GITBaseTextVQAModel.CreateTransformerAsync();
using var image = MLImage.CreateFromFile("photo.jpg");
var answer = transformer.AnswerQuestion(image, "What is in this image?");
Console.WriteLine($"Answer: {answer}");
```

## API

| Method | Description |
|--------|-------------|
| `CreateTransformerAsync()` | Creates a VQA transformer (downloads model on first use) |
| `CreateChatClientAsync()` | Creates an MEAI IChatClient for conversational VQA |
| `EnsureFilesAsync()` | Downloads model files and returns local paths |
| `GetModelInfoAsync()` | Returns model metadata |
| `VerifyModelAsync()` | Verifies cached model integrity |
