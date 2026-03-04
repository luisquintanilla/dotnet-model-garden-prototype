# Audio Embedding Sample

> Generate audio embeddings and compute acoustic similarity using the [CLAP](https://huggingface.co/laion/clap-htsat-unfused) (Contrastive Language-Audio Pretraining) model via the `Microsoft.Extensions.AI` `IEmbeddingGenerator` interface.

## Prerequisites

- [.NET 10 SDK (preview)](https://dotnet.microsoft.com/download/dotnet/10.0)
- Internet connection (the model downloads ~120 MB on first use)

## What This Sample Demonstrates

Audio embeddings map audio signals into a vector space where acoustically similar sounds are close together. This sample:

1. Generates three synthetic audio signals with different characteristics (two sine waves and white noise).
2. Creates an `IEmbeddingGenerator<AudioData, Embedding<float>>` backed by the CLAP ONNX model.
3. Generates 512-dimensional embeddings for each audio clip.
4. Computes **cosine similarity** between pairs to show that harmonically related tones are more similar to each other than to noise.

## Running the Sample

```sh
cd dotnet-model-garden-prototype
dotnet run --project samples/AudioEmbeddingSample
```

## Expected Output

```
=== Model Garden: Audio Embedding Sample ===

Generating synthetic audio signals...
  Audio 1: 440Hz sine wave (A4)
  Audio 2: 880Hz sine wave (A5)
  Audio 3: White noise

Creating audio embedding generator (downloads on first use)...
Generator ready in 15511ms

Generating embeddings...
Generated 3 embeddings, dimension: 512
Embedding generation took 642ms

Cosine Similarity (similar sounds should score higher):
  440Hz vs 880Hz (similar tones): 0.9694
  440Hz vs noise (different):     0.8956
  880Hz vs noise (different):     0.8791

Model info:
  Model ID: laion/clap-htsat-unfused

Done!
```

## Code Walkthrough

1. **Generate synthetic audio** — Three 1-second, 16 kHz signals are created: a 440 Hz sine wave (A4 note), an 880 Hz sine wave (A5 note — one octave higher), and white noise. Each is wrapped in an `AudioData` object. In a real application you would load WAV files with `AudioIO.LoadWav("sound.wav")`.
2. **Create the embedding generator** — `CLAPModel.CreateEmbeddingGeneratorAsync()` downloads the ONNX model on first run and returns an `IEmbeddingGenerator<AudioData, Embedding<float>>`.
3. **Generate embeddings** — `generator.GenerateAsync([audio1, audio2, audio3])` processes all three clips and produces one 512-dimensional float vector per input.
4. **Compute cosine similarity** — `TensorPrimitives.CosineSimilarity` measures how close two embedding vectors are. The two tones (440 Hz and 880 Hz) score higher similarity to each other than either does to noise.
5. **Show model info** — Prints the model ID and source.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Audio Embedding** | A fixed-length numeric vector that captures the acoustic characteristics of an audio clip. |
| **CLAP** | Contrastive Language-Audio Pretraining — a model trained to align audio and text in a shared embedding space, enabling cross-modal retrieval. |
| **Cosine Similarity** | A metric ranging from −1 to 1 measuring the angle between two vectors. Values closer to 1 indicate higher acoustic similarity. |
| **IEmbeddingGenerator** | The `Microsoft.Extensions.AI` abstraction for embedding generators, enabling model-agnostic code for both text and audio. |

## Next Steps

- **Use real audio** — Replace the synthetic signals with WAV files: `var audio = AudioIO.LoadWav("sound.wav");`
- **Audio search** — Build an audio retrieval system by storing embeddings and finding the closest match to a query clip.
- **Cross-modal search** — CLAP aligns audio and text embeddings, so you could search for audio using natural language descriptions.
- Explore the [EmbeddingSample](../EmbeddingSample/) for text embeddings, or the [AudioClassificationSample](../AudioClassificationSample/) for labeling audio content.

## Model Package References

- [`DotnetAILab.ModelGarden.AudioEmbedding.CLAP`](../../models/audioembeddings/DotnetAILab.ModelGarden.AudioEmbedding.CLAP/) — CLAP ONNX model package
- [Hugging Face: laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused)
