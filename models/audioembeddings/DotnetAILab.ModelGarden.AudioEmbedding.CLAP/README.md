# CLAP — Audio Embeddings

> Contrastive Language-Audio Pretraining model for generating 512-dimensional audio embeddings.

## Overview

[CLAP](https://github.com/LAION-AI/CLAP) (Contrastive Language-Audio Pretraining) learns a shared embedding space for audio and text — similar to CLIP for images. This package exposes the **audio encoder** of the `laion/clap-htsat-unfused` model, which maps any audio clip to a **512-dimensional L2-normalized vector**.

These embeddings capture the semantic content of sounds: similar-sounding clips land close together in embedding space, while dissimilar sounds are far apart. Use them for similarity search, clustering, deduplication, or as features for downstream classifiers.

The ONNX model binary (~120 MB) is **not** shipped in the NuGet — it downloads transparently on first use and is cached locally.

## Model Details

| Property | Value |
|---|---|
| **Model ID** | `laion/clap-htsat-unfused` |
| **Architecture** | HTSAT (Hierarchical Token-Semantic Audio Transformer) encoder |
| **Embedding Dimension** | 512 |
| **Normalization** | L2-normalized (unit vectors) |
| **Pooling** | Mean pooling |
| **Size** | ~120 MB |
| **Format** | ONNX |
| **Sample Rate** | 16 kHz mono |
| **License** | Apache-2.0 |
| **Source** | [HuggingFace](https://huggingface.co/lquint/clap-htsat-unfused-onnx) / [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP) |

## Installation

```bash
dotnet add package DotnetAILab.ModelGarden.AudioEmbedding.CLAP
```

> **NuGet source** — this package is published to GitHub Packages. See the [root README](../../../README.md#nuget-source-setup) for source configuration.

## Quick Start

```csharp
using DotnetAILab.ModelGarden.AudioEmbedding.CLAP;
using Microsoft.Extensions.AI;
using MLNet.Audio.Core;
using System.Numerics.Tensors;

// Load two audio clips (16 kHz mono)
// var audio1 = AudioIO.LoadWav("guitar.wav");
// var audio2 = AudioIO.LoadWav("piano.wav");
var audio1 = new AudioData(samples1, sampleRate: 16000);
var audio2 = new AudioData(samples2, sampleRate: 16000);

// Create the embedding generator — model downloads (~120 MB) on first call
IEmbeddingGenerator<AudioData, Embedding<float>> generator =
    await CLAPModel.CreateEmbeddingGeneratorAsync();

// Generate embeddings
var embeddings = await generator.GenerateAsync([audio1, audio2]);

// Compute cosine similarity (embeddings are L2-normalized, so dot product = cosine sim)
float similarity = TensorPrimitives.CosineSimilarity(
    embeddings[0].Vector.Span,
    embeddings[1].Vector.Span);

Console.WriteLine($"Similarity: {similarity:F4}");
```

## API Reference

### `CLAPModel.CreateEmbeddingGeneratorAsync`

```csharp
public static async Task<IEmbeddingGenerator<AudioData, Embedding<float>>> CreateEmbeddingGeneratorAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

Creates an `IEmbeddingGenerator<AudioData, Embedding<float>>` backed by the CLAP ONNX model. This implements the **`Microsoft.Extensions.AI`** `IEmbeddingGenerator` interface, making it interoperable with the broader .NET AI ecosystem.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `options` | `ModelOptions?` | `null` | Override download source, cache path, etc. |
| `ct` | `CancellationToken` | `default` | Cancellation token. |

**Returns:** `IEmbeddingGenerator<AudioData, Embedding<float>>` — call `GenerateAsync(IList<AudioData>)` to produce embeddings.

**Internal configuration:**

| Setting | Value |
|---|---|
| Feature extractor | `MelSpectrogramExtractor` |
| Mel bins | 64 |
| FFT size | 512 |
| Hop length | 160 |
| Pooling | Mean pooling |
| Normalize | `true` (L2) |
| Sample rate | 16 000 Hz |

---

### Utility Methods

```csharp
// Pre-download / cache the model files without creating a generator
Task<ModelFiles> EnsureFilesAsync(ModelOptions? options = null, CancellationToken ct = default)

// Retrieve model metadata (ID, resolved source, file paths)
Task<ModelInfo> GetModelInfoAsync(ModelOptions? options = null, CancellationToken ct = default)

// Verify cached model integrity (SHA-256 + size)
Task VerifyModelAsync(ModelOptions? options = null, CancellationToken ct = default)
```

## Inputs & Outputs

### Input

| Property | Requirement |
|---|---|
| **Type** | `AudioData` (from `MLNet.Audio.Core`) |
| **Sample rate** | 16 kHz |
| **Channels** | Mono |
| **Format** | `float[]` PCM samples |

Audio is internally converted to a mel spectrogram (64 mel bins, FFT size 512, hop 160) before being fed to the HTSAT encoder.

### Output

`Embedding<float>` (from `Microsoft.Extensions.AI`) per input clip:

| Property | Description |
|---|---|
| `Vector` | `ReadOnlyMemory<float>` — 512-dimensional, L2-normalized |

Because the vectors are L2-normalized, **cosine similarity equals the dot product**, which simplifies distance calculations.

## Use Cases

- **Audio similarity search** — index embeddings in a vector database and find sounds that match a query clip.
- **Audio clustering / categorization** — group similar sounds (e.g., separate music, speech, and environmental noise).
- **Content-based audio retrieval** — search a media library by acoustic similarity rather than metadata.
- **Sound effect libraries** — find alternative takes or similar effects across large collections.
- **Duplicate / near-duplicate detection** — flag audio files that are acoustically identical or very similar.

## Limitations

- **Audio-only embeddings** — this package exposes the audio encoder only; the CLAP text encoder is not included. You cannot compare audio to text queries with this package alone.
- **Fixed mel configuration** — the mel spectrogram settings (64 bins, FFT 512, hop 160) are baked into the model and cannot be changed.
- **16 kHz mono only** — audio must be resampled before processing if recorded at a different rate.
- **512 dimensions** — the embedding vectors are moderately sized; applications with extreme storage constraints may need dimensionality reduction.

## Example: Building an Audio Search System

```csharp
using DotnetAILab.ModelGarden.AudioEmbedding.CLAP;
using Microsoft.Extensions.AI;
using MLNet.Audio.Core;
using System.Numerics.Tensors;

// Build a small library of audio clips
// In a real app, load from files:
// var clips = Directory.GetFiles("audio_library", "*.wav")
//     .Select(f => AudioIO.LoadWav(f)).ToList();
var clips = new List<AudioData>
{
    new AudioData(guitarSamples, 16000),
    new AudioData(drumSamples, 16000),
    new AudioData(pianoSamples, 16000),
    new AudioData(birdSamples, 16000),
    new AudioData(trafficSamples, 16000),
};
var labels = new[] { "Guitar", "Drums", "Piano", "Birds", "Traffic" };

IEmbeddingGenerator<AudioData, Embedding<float>> generator =
    await CLAPModel.CreateEmbeddingGeneratorAsync();

// Index: generate embeddings for all clips
var libraryEmbeddings = await generator.GenerateAsync(clips);

// Query: embed a new audio clip and find the closest match
var query = new AudioData(unknownSamples, 16000);
var queryEmbedding = (await generator.GenerateAsync([query]))[0];

Console.WriteLine("Similarity to library:");
for (int i = 0; i < clips.Count; i++)
{
    float sim = TensorPrimitives.CosineSimilarity(
        queryEmbedding.Vector.Span,
        libraryEmbeddings[i].Vector.Span);
    Console.WriteLine($"  {labels[i],-10} → {sim:F4}");
}

// Find best match
int bestIdx = Enumerable.Range(0, clips.Count)
    .MaxBy(i => TensorPrimitives.CosineSimilarity(
        queryEmbedding.Vector.Span,
        libraryEmbeddings[i].Vector.Span));
Console.WriteLine($"\nBest match: {labels[bestIdx]}");
```

## References

- [CLAP: Learning Audio Concepts from Natural Language Supervision](https://arxiv.org/abs/2206.04769) — Wu et al., 2023
- [LAION-AI/CLAP — GitHub](https://github.com/LAION-AI/CLAP)
- [CLAP HTSAT Unfused — HuggingFace](https://huggingface.co/lquint/clap-htsat-unfused-onnx)
- [Microsoft.Extensions.AI — IEmbeddingGenerator](https://learn.microsoft.com/dotnet/api/microsoft.extensions.ai.iembeddinggenerator-2)
- [.NET Model Garden — root README](../../../README.md)
