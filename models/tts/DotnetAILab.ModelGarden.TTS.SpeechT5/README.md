# SpeechT5 — Text-to-Speech

> Microsoft SpeechT5 model for high-quality text-to-speech synthesis, producing 16 kHz WAV audio.

## Overview

SpeechT5 is Microsoft's unified-modal encoder-decoder model designed to handle a variety of spoken-language tasks including text-to-speech (TTS), speech-to-text, and speech-to-speech conversion. Originally described by Ao et al. (2022), the architecture shares a single Transformer backbone across tasks while swapping lightweight pre-nets and post-nets to adapt to each modality. This package wraps the ONNX-exported TTS variant so it runs entirely on-device via ONNX Runtime — no cloud service or API key required.

During text-to-speech synthesis the pipeline flows through three stages. First, an **encoder** converts input text (tokenized by a SentencePiece character-level model) into a sequence of hidden representations. Next, an autoregressive **decoder** generates a mel-spectrogram frame by frame, conditioned on the encoder output and a fixed speaker embedding. Finally, a **HiFi-GAN vocoder** (fused with the decoder post-net) converts the mel-spectrogram into a 16 kHz mono PCM waveform.

The model is distributed as five files totalling roughly **643 MB**, all downloaded automatically on first use from the [NeuML/txtai-speecht5-onnx](https://huggingface.co/NeuML/txtai-speecht5-onnx) Hugging Face repository and cached locally:

| File | Purpose | Size |
|---|---|---|
| `encoder_model.onnx` | Text encoder | ~343 MB |
| `decoder_model_merged.onnx` | Autoregressive mel decoder | ~244 MB |
| `decoder_postnet_and_vocoder.onnx` | Post-net + HiFi-GAN vocoder | ~55 MB |
| `spm_char.model` | SentencePiece character tokenizer | ~238 KB |
| `speaker.npy` | Speaker embedding vector | ~2 KB |

## Model Details

| Property | Value |
|---|---|
| **Model ID** | `microsoft/speecht5_tts` |
| **Architecture** | Encoder-decoder Transformer + HiFi-GAN vocoder |
| **Output format** | 16 kHz mono PCM (float32) |
| **License** | MIT |
| **Source repository** | [NeuML/txtai-speecht5-onnx](https://huggingface.co/NeuML/txtai-speecht5-onnx) |
| **Total download size** | ~643 MB |
| **Runtime** | ONNX Runtime (CPU) |
| **Package ID** | `DotnetAILab.ModelGarden.TTS.SpeechT5` |
| **Target framework** | .NET 10 |

## Installation / Quick Start

Add the NuGet package to your project:

```bash
dotnet add package DotnetAILab.ModelGarden.TTS.SpeechT5
```

Synthesize speech and save it to a WAV file:

```csharp
using DotnetAILab.ModelGarden.TTS.SpeechT5;
using MLNet.Audio.Core;

// Create a TTS client — downloads ~643 MB on first run, cached thereafter
var tts = await SpeechT5Model.CreateTextToSpeechClientAsync();

// Synthesize
var response = await tts.GetAudioAsync(
    "Hello! This is a test of the SpeechT5 text to speech model.");

Console.WriteLine($"Duration: {response.Duration.TotalSeconds:F2}s");
Console.WriteLine($"Sample rate: {response.Audio.SampleRate} Hz");

// Save to WAV
AudioIO.SaveWav("output.wav", response.Audio);

tts.Dispose();
```

## API Reference

### `SpeechT5Model.CreateTtsTransformerAsync`

```csharp
public static Task<OnnxSpeechT5TtsTransformer> CreateTtsTransformerAsync(
    MLContext? mlContext = null,
    ModelOptions? options = null,
    CancellationToken ct = default);
```

Creates a low-level `OnnxSpeechT5TtsTransformer` that gives direct control over the synthesis pipeline (e.g. custom `MLContext`, advanced inference options). Use this when you need fine-grained access to the underlying ONNX models. Model files are downloaded and cached on first call.

### `SpeechT5Model.CreateTextToSpeechClientAsync`

```csharp
public static Task<ITextToSpeechClient> CreateTextToSpeechClientAsync(
    ModelOptions? options = null,
    CancellationToken ct = default);
```

Creates a high-level `ITextToSpeechClient` for straightforward text-to-speech synthesis. This is the recommended entry point for most applications. Call `GetAudioAsync(text)` on the returned client to synthesize speech.

> **Note:** `ITextToSpeechClient` is a **prototype** interface defined in `MLNet.AudioInference.Onnx`. It is _not_ part of `Microsoft.Extensions.AI` and its API may change.

### `SpeechT5Model.EnsureFilesAsync`

```csharp
public static Task<ModelFiles> EnsureFilesAsync(
    ModelOptions? options = null,
    CancellationToken ct = default);
```

Downloads all five model files (if not already cached) and returns a `ModelFiles` object. Call `files.GetPath("encoder_model.onnx")` to resolve individual file paths. Useful when you want to manage model files without creating a transformer or client.

### `SpeechT5Model.GetModelInfoAsync`

```csharp
public static Task<ModelInfo> GetModelInfoAsync(
    ModelOptions? options = null,
    CancellationToken ct = default);
```

Returns metadata about the model package — model ID, resolved source, revision, and file information — without downloading the model.

### `SpeechT5Model.VerifyModelAsync`

```csharp
public static Task VerifyModelAsync(
    ModelOptions? options = null,
    CancellationToken ct = default);
```

Verifies the integrity of previously downloaded model files by checking SHA-256 hashes against the manifest.

## Inputs & Outputs

| Direction | Type | Description |
|---|---|---|
| **Input** | `string` | Plain text to synthesize (English) |
| **Output** | `TextToSpeechResponse` | Contains `Audio` (`AudioData` — 16 kHz mono float32 PCM) and `Duration` |

### Saving to WAV

Use `AudioIO.SaveWav` from the `MLNet.Audio.Core` namespace:

```csharp
AudioIO.SaveWav("speech.wav", response.Audio);
```

The output duration scales with the length and complexity of the input text. Short sentences typically produce 1–4 seconds of audio.

## Use Cases

- **Accessibility** — screen readers, text-to-speech for visually impaired users
- **Voice assistants** — offline spoken responses in kiosks or embedded devices
- **Audio content generation** — narrating articles, blog posts, or documentation
- **Language learning tools** — pronunciation examples and listening exercises
- **Notification systems** — spoken alerts and announcements

## Limitations

| Limitation | Detail |
|---|---|
| **English only** | The model was trained on English speech data; other languages are not supported. |
| **Single speaker** | `speaker.npy` provides a fixed female voice; there is no multi-speaker or voice-cloning support. |
| **No SSML** | Speech Synthesis Markup Language is not supported; input is plain text only. |
| **Large download** | ~643 MB must be downloaded on first use (cached for subsequent runs). |
| **CPU speed** | Synthesis is slower than real-time on most CPUs; not suitable for low-latency streaming. |
| **No streaming** | The entire waveform is generated before any audio is returned. |
| **Max length** | `MaxMelFrames` is set to 500, which limits the maximum output duration per call. |

## Example: Generating and Saving Speech

```csharp
using DotnetAILab.ModelGarden.TTS.SpeechT5;
using MLNet.Audio.Core;
using System.Diagnostics;

var sentences = new[]
{
    "Welcome to the dotnet model garden.",
    "This library lets you run AI models entirely on device.",
    "No cloud service or API key is required."
};

// Create the TTS client once and reuse it
var tts = await SpeechT5Model.CreateTextToSpeechClientAsync();

for (int i = 0; i < sentences.Length; i++)
{
    var sw = Stopwatch.StartNew();
    var response = await tts.GetAudioAsync(sentences[i]);
    sw.Stop();

    var path = $"sentence_{i + 1}.wav";
    AudioIO.SaveWav(path, response.Audio);

    Console.WriteLine($"[{i + 1}] \"{sentences[i]}\"");
    Console.WriteLine($"     Duration: {response.Duration.TotalSeconds:F2}s | " +
                      $"Samples: {response.Audio.Samples.Length:N0} | " +
                      $"Generated in: {sw.ElapsedMilliseconds}ms");
}

tts.Dispose();
Console.WriteLine("All files saved.");
```

## References

- **SpeechT5 paper** — Junyi Ao et al., _"SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing"_, ACL 2022. [arXiv:2110.07205](https://arxiv.org/abs/2110.07205)
- **Hugging Face model card** — [microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts)
- **ONNX export source** — [NeuML/txtai-speecht5-onnx](https://huggingface.co/NeuML/txtai-speecht5-onnx)
- **ONNX Runtime** — [https://onnxruntime.ai](https://onnxruntime.ai)
