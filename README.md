# .NET Model Garden Prototype

A collection of pre-packaged AI model NuGet packages for .NET. Each package wraps a HuggingFace model as a lightweight NuGet — the heavy ONNX binary downloads transparently on first use.

## Quick Start

```bash
# Install a model package
dotnet add package DotnetAILab.ModelGarden.Embeddings.AllMiniLM
```

```csharp
using DotnetAILab.ModelGarden.Embeddings.AllMiniLM;

// One-liner — model auto-downloads on first call, cached thereafter
var generator = await AllMiniLMModel.CreateEmbeddingGeneratorAsync();
var embeddings = await generator.GenerateAsync(new[] { "Hello world" });
```

## Available Model Packages

### Embeddings

| Package | HuggingFace Model | Facade Method |
|---|---|---|
| `DotnetAILab.ModelGarden.Embeddings.AllMiniLM` | all-MiniLM-L6-v2 | `CreateEmbeddingGeneratorAsync()` |
| `DotnetAILab.ModelGarden.Embeddings.BgeSmallEn` | bge-small-en-v1.5 | `CreateEmbeddingGeneratorAsync()` |
| `DotnetAILab.ModelGarden.Embeddings.E5Small` | e5-small-v2 | `CreateEmbeddingGeneratorAsync()` |
| `DotnetAILab.ModelGarden.Embeddings.GteSmall` | gte-small | `CreateEmbeddingGeneratorAsync()` |

### Classification

| Package | HuggingFace Model | Facade Method |
|---|---|---|
| `DotnetAILab.ModelGarden.Classification.SentimentDistilBERT` | distilbert-base-uncased-finetuned-sst-2-english | `CreateClassifierAsync()` |
| `DotnetAILab.ModelGarden.Classification.EmotionRoBERTa` | roberta-base-go_emotions | `CreateClassifierAsync()` |
| `DotnetAILab.ModelGarden.Classification.ZeroShotDeBERTa` | DeBERTa-v3-base-mnli-fever-anli | `CreateClassifierAsync()` |

### Reranking

| Package | HuggingFace Model | Facade Method |
|---|---|---|
| `DotnetAILab.ModelGarden.Reranking.BgeReranker` | bge-reranker-base | `CreateRerankerAsync()` |
| `DotnetAILab.ModelGarden.Reranking.MsMarcoMiniLM` | ms-marco-MiniLM-L-6-v2 | `CreateRerankerAsync()` |

### Named Entity Recognition

| Package | HuggingFace Model | Facade Method |
|---|---|---|
| `DotnetAILab.ModelGarden.NER.BertBaseNER` | bert-base-NER | `CreateNerAsync()` |
| `DotnetAILab.ModelGarden.NER.MultilingualNER` | bert-base-multilingual-cased-ner-hrl | `CreateNerAsync()` |

### Question Answering

| Package | HuggingFace Model | Facade Method |
|---|---|---|
| `DotnetAILab.ModelGarden.QA.MiniLMSquad2` | minilm-uncased-squad2 | `CreateQaAsync()` |
| `DotnetAILab.ModelGarden.QA.RobertaSquad2` | roberta-base-squad2 | `CreateQaAsync()` |

### Text Generation

| Package | HuggingFace Model | Facade Method |
|---|---|---|
| `DotnetAILab.ModelGarden.TextGeneration.Phi3Mini` | Phi-3-mini-4k-instruct-onnx | `CreateTextGeneratorAsync()` |

## Architecture

```
Consumer App
    │ NuGet PackageReference
    ▼
Model Package (this repo)
    │ Contains: model-manifest.json + vocab.txt + static facade
    │ NuGet PackageReferences ▼
    ├── ModelPackages (Core SDK) — fetch, cache, verify model binaries
    └── MLNet.TextInference.Onnx — tokenization, ONNX inference, post-processing
```

Each model package:
- **Contains only metadata** (~few KB) — no ONNX binaries in the NuGet
- **Auto-downloads** the model from HuggingFace on first use
- **Caches locally** under the user's app data directory
- **Verifies integrity** via SHA256 hash
- **Supports source redirection** via `model-sources.json` or environment variables

## NuGet Source Setup

Model packages and their dependencies are published to GitHub Packages. Add this to your `nuget.config`:

```xml
<packageSources>
  <add key="github" value="https://nuget.pkg.github.com/luisquintanilla/index.json" />
</packageSources>
```

## Customizing Model Source

Override where the model binary is fetched from:

```bash
# Environment variable
set MODELPACKAGES_SOURCE=company-mirror

# Or add model-sources.json next to your .csproj
```

## Upstream Dependencies

| Package | Source Repo |
|---|---|
| `ModelPackages` | [model-packages-prototype](https://github.com/luisquintanilla/model-packages-prototype) |
| `MLNet.TextInference.Onnx` | [mlnet-text-inference-custom-transforms](https://github.com/luisquintanilla/mlnet-text-inference-custom-transforms) |
| `MLNet.TextGeneration.OnnxGenAI` | [mlnet-text-inference-custom-transforms](https://github.com/luisquintanilla/mlnet-text-inference-custom-transforms) |
| `MLNet.Embeddings.Onnx` | [mlnet-embedding-custom-transforms](https://github.com/luisquintanilla/mlnet-embedding-custom-transforms) |

## License

MIT
