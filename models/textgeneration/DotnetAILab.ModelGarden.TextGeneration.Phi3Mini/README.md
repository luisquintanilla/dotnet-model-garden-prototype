# Phi-3 Mini 4K — Text Generation

> Microsoft Phi-3 Mini with 4K context window, INT4 quantized for efficient local CPU inference via ONNX Runtime GenAI.

## Overview

**What is Phi-3?** Phi-3 is Microsoft's family of small language models (SLMs) designed to deliver strong performance at a fraction of the size and cost of larger models. Phi-3 Mini is the smallest variant at 3.8 billion parameters, yet it achieves competitive quality on reasoning, coding, and language understanding benchmarks.

**What is INT4 quantization?** This package uses the INT4 (4-bit integer) quantized version of Phi-3 Mini. Quantization compresses the model's floating-point weights into 4-bit integers, reducing the model size from ~7.6 GB (FP16) to ~2.7 GB and enabling faster inference on CPUs without dedicated GPU hardware. The trade-off is a small reduction in output quality compared to the full-precision model.

**How does autoregressive generation work?** The model generates text one token at a time. Given an input prompt, it predicts the most likely next token, appends it to the sequence, and repeats until it reaches a stop condition (max length, end-of-sequence token, etc.). This package uses ONNX Runtime GenAI to run this loop efficiently on CPU.

## Model Details

| Property | Value |
|---|---|
| **Model ID** | [`microsoft/Phi-3-mini-4k-instruct-onnx`](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx) |
| **Parameters** | 3.8B (original) |
| **Quantization** | INT4 (RTN, block-32, acc-level-4) |
| **Context Window** | 4,096 tokens |
| **Total Download** | **~2.7 GB** (6 files) |
| **Runtime** | ONNX Runtime GenAI (CPU) |
| **Source** | Hugging Face |
| **License** | MIT |
| **NuGet Package** | `DotnetAILab.ModelGarden.TextGeneration.Phi3Mini` |

### Model Files

| File | Size | Description |
|---|---|---|
| `phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx` | ~231 KB | ONNX graph definition |
| `phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx.data` | ~2.72 GB | Quantized model weights |
| `genai_config.json` | ~1.5 KB | GenAI runtime configuration |
| `tokenizer.model` | ~500 KB | SentencePiece tokenizer |
| `tokenizer_config.json` | ~3.4 KB | Tokenizer configuration |
| `special_tokens_map.json` | ~600 B | Special token definitions |

## Installation

Add the NuGet package to your project:

```shell
dotnet add package DotnetAILab.ModelGarden.TextGeneration.Phi3Mini
```

All 6 model files (~2.7 GB total) are **automatically downloaded** on first use and cached locally. No manual download is required.

> **Note:** The first call to `CreateTextGeneratorAsync` may take several minutes depending on your network speed due to the ~2.7 GB download.

### Dependencies

| Package | Version |
|---|---|
| `ModelPackages` | 0.1.0-preview.14 |
| `MLNet.TextGeneration.OnnxGenAI` | 0.1.0-preview.1 |

## Quick Start

```csharp
using DotnetAILab.ModelGarden.TextGeneration.Phi3Mini;
using Microsoft.ML;

// Create the text generator (downloads model on first call)
var generator = await Phi3MiniModel.CreateTextGeneratorAsync();

var mlContext = new MLContext();

// Prepare input
var input = new[] { new TextData { Text = "Explain quantum computing in simple terms:" } };
var dataView = mlContext.Data.LoadFromEnumerable(input);

// Generate text
var result = generator.Transform(dataView);
var output = mlContext.Data
    .CreateEnumerable<TextData>(result, reuseRowObject: false)
    .First();

Console.WriteLine(output.Text);

public class TextData
{
    public string Text { get; set; } = "";
}
```

## API Reference

### `Phi3MiniModel.CreateTextGeneratorAsync`

```csharp
public static async Task<OnnxTextGenerationTransformer> CreateTextGeneratorAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

Creates a text generation transformer backed by the local ONNX model directory. Downloads all model files on first call; cached thereafter.

**Parameters:**
- `options` — Optional `ModelOptions` for controlling download behavior (cache directory, etc.)
- `ct` — Cancellation token

**Returns:** `OnnxTextGenerationTransformer` — an ML.NET transformer that generates text continuations.

**Generation Parameters:**

| Parameter | Value | Description |
|---|---|---|
| `MaxLength` | 256 | Maximum number of tokens to generate |
| `Temperature` | 0.7 | Controls randomness (lower = more deterministic) |
| `TopP` | 0.9 | Nucleus sampling threshold |

### `Phi3MiniModel.EnsureModelAsync`

```csharp
public static async Task<string> EnsureModelAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

Downloads all model files if not already cached and returns the local directory path containing the model.

### `Phi3MiniModel.GetModelInfoAsync`

Returns metadata about the model package (ID, source, file sizes).

### `Phi3MiniModel.VerifyModelAsync`

Verifies the integrity of cached model files using SHA-256 checksums.

> **⚠️ Known Issue:** The SHA-256 hash for the `.onnx.data` file is currently empty in the model manifest. Integrity verification for this file is pending. See [Issue #9](../../issues/9).

## Inputs & Outputs

### Input

| Field | Type | Description |
|---|---|---|
| `Text` | `string` | The prompt or instruction for the model |

For best results with Phi-3 Mini Instruct, use the chat template format:

```
<|user|>
Your question or instruction here<|end|>
<|assistant|>
```

### Output

| Field | Type | Description |
|---|---|---|
| `Text` | `string` | Generated text continuation |

The model generates up to **256 tokens** per call. Output may stop earlier if the model produces an end-of-sequence token.

## Use Cases

- **Chatbots & Assistants** — Build conversational interfaces that run entirely locally without API calls.
- **Code Generation** — Generate code snippets, function implementations, or explanations.
- **Summarization** — Condense long text into shorter summaries (within the 4K context window).
- **Question Answering** — Answer questions given context (pairs well with reranking models for RAG).
- **Content Creation** — Draft emails, documentation, or creative writing.

> **Key advantage:** This is a fully local/offline model. No API keys, no network calls, no usage fees. Your data never leaves your machine.

## Limitations

- **Large download (~2.7 GB)** — The initial model download is substantial. Ensure adequate disk space and a stable network connection.
- **CPU inference is slow** — Autoregressive generation on CPU takes several seconds per token. For interactive use cases, consider GPU-accelerated alternatives.
- **4K context window** — The model can process at most 4,096 tokens (input + output combined). Longer inputs will be truncated.
- **256-token generation limit** — The current configuration caps output at 256 tokens. Longer outputs require multiple calls or configuration changes.
- **INT4 quality trade-off** — 4-bit quantization reduces model quality slightly compared to FP16/FP32. Expect occasional degradation in complex reasoning or nuanced text.
- **No streaming** — The current API generates the full response synchronously. There is no token-by-token streaming callback.
- **SHA-256 verification gap** — The `.onnx.data` file hash is empty in the manifest, so integrity verification is incomplete for the model weights file. See [Issue #9](../../issues/9).

## Example: Chat Completion

```csharp
using DotnetAILab.ModelGarden.TextGeneration.Phi3Mini;
using Microsoft.ML;

var generator = await Phi3MiniModel.CreateTextGeneratorAsync();
var mlContext = new MLContext();

// Use Phi-3 instruct format for best results
var prompt = """
    <|user|>
    Write a haiku about machine learning.<|end|>
    <|assistant|>
    """;

var input = new[] { new TextData { Text = prompt } };
var dataView = mlContext.Data.LoadFromEnumerable(input);
var result = generator.Transform(dataView);

var output = mlContext.Data
    .CreateEnumerable<TextData>(result, reuseRowObject: false)
    .First();

Console.WriteLine(output.Text);

public class TextData
{
    public string Text { get; set; } = "";
}
```

## Example: RAG with Reranking + Generation

```csharp
using DotnetAILab.ModelGarden.Reranking.BgeReranker;
using DotnetAILab.ModelGarden.TextGeneration.Phi3Mini;
using Microsoft.ML;

// After retrieval and reranking (see reranking model READMEs)...
var topDocuments = new[] { "Document 1 content...", "Document 2 content..." };
var question = "What causes aurora borealis?";

var context = string.Join("\n\n", topDocuments);
var prompt = $"""
    <|user|>
    Answer the question based only on the following context:

    {context}

    Question: {question}<|end|>
    <|assistant|>
    """;

var generator = await Phi3MiniModel.CreateTextGeneratorAsync();
var mlContext = new MLContext();

var input = new[] { new TextData { Text = prompt } };
var dataView = mlContext.Data.LoadFromEnumerable(input);
var result = generator.Transform(dataView);

var answer = mlContext.Data
    .CreateEnumerable<TextData>(result, reuseRowObject: false)
    .First();

Console.WriteLine(answer.Text);

public class TextData
{
    public string Text { get; set; } = "";
}
```

## Related Models

| Model | Task | Package |
|---|---|---|
| BGE Reranker Base | Reranking | `DotnetAILab.ModelGarden.Reranking.BgeReranker` |
| MS MARCO MiniLM | Reranking | `DotnetAILab.ModelGarden.Reranking.MsMarcoMiniLM` |
| AllMiniLM | Embeddings | `DotnetAILab.ModelGarden.Embeddings.AllMiniLM` |
| BGE Small EN | Embeddings | `DotnetAILab.ModelGarden.Embeddings.BgeSmallEn` |

## References

- [microsoft/Phi-3-mini-4k-instruct-onnx on Hugging Face](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx)
- [Phi-3 Technical Report (arXiv)](https://arxiv.org/abs/2404.14219)
- [Microsoft Phi-3 Blog Post](https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/)
- [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai)
