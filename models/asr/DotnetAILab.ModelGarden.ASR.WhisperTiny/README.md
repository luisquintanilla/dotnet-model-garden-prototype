# DotnetAILab.ModelGarden.ASR.WhisperTiny

> OpenAI Whisper Tiny — the smallest and fastest Whisper speech-to-text model for .NET, powered by ONNX Runtime.

## Overview

**Whisper Tiny** is the most compact variant of OpenAI's Whisper automatic speech recognition (ASR) family, packaged for seamless use in .NET applications. With only 39 million parameters and a ~144 MB download, it delivers the fastest inference of any Whisper model while still supporting 99 languages — all running locally via ONNX Runtime with no cloud dependencies.

Like all Whisper models, Tiny uses an **encoder-decoder Transformer** architecture trained on 680,000 hours of multilingual audio data. The encoder converts a log-Mel spectrogram (80 mel bins, 16 kHz) into hidden representations, and the decoder auto-regressively generates text tokens. The Tiny variant uses fewer layers and a smaller hidden dimension than its larger siblings, which is why it runs roughly **2× faster** than [Whisper Base](../DotnetAILab.ModelGarden.ASR.WhisperBase/README.md) at the cost of some accuracy.

Choose Whisper Tiny when **latency and resource footprint matter more than transcription accuracy** — for example, in real-time voice command processing, edge devices, CI/CD pipelines, or rapid prototyping. If you need higher accuracy for production transcription (especially with accented speech, background noise, or non-English languages), consider the [Whisper Base](../DotnetAILab.ModelGarden.ASR.WhisperBase/README.md) variant instead.

## Model Details

| Property | Value |
|---|---|
| **Model ID** | `openai/whisper-tiny` |
| **Architecture** | Whisper encoder-decoder Transformer |
| **Parameters** | 39 million |
| **License** | Apache 2.0 |
| **Source** | [onnx-community/whisper-tiny](https://huggingface.co/onnx-community/whisper-tiny) (HuggingFace) |
| **Runtime** | ONNX Runtime (via ML.NET) |
| **Download Size** | ~144 MB (encoder 31.4 MB + decoder 113 MB) |
| **Target Framework** | .NET 10+ |
| **Languages** | 99 languages (Afrikaans, Arabic, Chinese, English, French, German, Hindi, Japanese, Korean, Spanish, and [many more](https://github.com/openai/whisper#available-models-and-languages)) |
| **Mel Bins** | 80 |
| **Max Tokens** | 256 |
| **Sample Rate** | 16,000 Hz |

## Installation

```bash
dotnet add package DotnetAILab.ModelGarden.ASR.WhisperTiny
```

The model weights (~144 MB) are **automatically downloaded** from HuggingFace on first use and cached locally. No manual model management is required.

### Dependencies

This package transitively brings in:

- `ModelPackages` (v0.1.0-preview.14) — model download, caching, and verification
- `MLNet.AudioInference.Onnx` (v0.1.0-preview.2) — ONNX-based Whisper inference via ML.NET

## Quick Start

```csharp
using DotnetAILab.ModelGarden.ASR.WhisperTiny;
using MLNet.Audio.Core;

// 1. Load audio (16 kHz, mono, float32 PCM)
var audio = AudioIO.LoadWav("voice-command.wav");

// 2. Create the speech-to-text pipeline (downloads model on first call)
var stt = await WhisperTinyModel.CreateSpeechToTextAsync(language: "en");

// 3. Transcribe
var results = stt.Transcribe([audio]);

foreach (var text in results)
    Console.WriteLine(text);

// 4. Clean up
stt.Dispose();
```

**Output:**
```
Turn on the living room lights.
```

## API Reference

### `WhisperTinyModel.CreateSpeechToTextAsync(...)`

Creates an `OnnxWhisperTransformer` pipeline ready for speech-to-text inference. Downloads model files on first invocation; subsequent calls use the local cache.

```csharp
public static async Task<OnnxWhisperTransformer> CreateSpeechToTextAsync(
    string language = "en",
    MLContext? mlContext = null,
    ModelOptions? options = null,
    CancellationToken ct = default)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `language` | `string` | `"en"` | [ISO 639-1 language code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) for the target transcription language (e.g. `"en"`, `"fr"`, `"zh"`, `"ja"`). |
| `mlContext` | `MLContext?` | `null` | Optional ML.NET context. A new one is created if not provided. Share one across pipelines to reduce memory overhead. |
| `options` | `ModelOptions?` | `null` | Optional download/cache configuration (custom cache directory, proxy settings, etc.). |
| `ct` | `CancellationToken` | `default` | Cancellation token for the async download/load operation. |

**Returns:** `Task<OnnxWhisperTransformer>` — a disposable transformer that exposes the `Transcribe()` method.

**Exceptions:**
- `HttpRequestException` — if model download fails (network issues, HuggingFace unavailable).
- `InvalidOperationException` — if model files are corrupted or SHA-256 verification fails.
- `OperationCanceledException` — if `ct` is cancelled during download.

---

### `OnnxWhisperTransformer.Transcribe(...)`

Runs inference on one or more audio inputs. This method is on the returned transformer, not on `WhisperTinyModel` directly.

```csharp
public IEnumerable<string> Transcribe(IEnumerable<AudioData> audioInputs)
```

| Parameter | Type | Description |
|---|---|---|
| `audioInputs` | `IEnumerable<AudioData>` | One or more audio samples to transcribe. Each must be 16 kHz mono float32 PCM. |

**Returns:** `IEnumerable<string>` — one transcription string per input audio. Empty strings indicate no speech was detected.

---

### `WhisperTinyModel.EnsureFilesAsync(...)`

Downloads model files if they are not already cached locally. Useful for pre-downloading during app startup so that `CreateSpeechToTextAsync` returns instantly later.

```csharp
public static Task<ModelFiles> EnsureFilesAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `options` | `ModelOptions?` | `null` | Optional download/cache configuration. |
| `ct` | `CancellationToken` | `default` | Cancellation token. |

**Returns:** `Task<ModelFiles>` — handle to the cached model files. Use `files.GetPath("onnx/encoder_model.onnx")` to access individual files.

---

### `WhisperTinyModel.GetModelInfoAsync(...)`

Retrieves metadata about the model package without downloading the full model.

```csharp
public static Task<ModelInfo> GetModelInfoAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

**Returns:** `Task<ModelInfo>` — contains `ModelId` (`"openai/whisper-tiny"`), `ResolvedSource`, and other metadata.

---

### `WhisperTinyModel.VerifyModelAsync(...)`

Verifies the integrity of cached model files by checking SHA-256 hashes against the manifest.

```csharp
public static Task VerifyModelAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

**Exceptions:**
- `InvalidOperationException` — if files are missing or hash verification fails.

## Inputs & Outputs

### Input: `AudioData`

The `AudioData` class (from `MLNet.Audio.Core`) wraps raw audio samples:

```csharp
using MLNet.Audio.Core;

// Construct from raw samples
var audio = new AudioData(samples, sampleRate);
```

| Property | Type | Description |
|---|---|---|
| `Samples` | `float[]` | Raw PCM audio samples normalized to the range `[-1.0, 1.0]`. |
| `SampleRate` | `int` | Must be **16,000 Hz**. Other sample rates must be resampled before use. |
| `Duration` | `TimeSpan` | Computed duration of the audio clip. |

**Requirements:**
- **Sample rate:** 16,000 Hz (16 kHz). If your source audio is 44.1 kHz or 48 kHz, resample first.
- **Channels:** Mono. If stereo, mix down to a single channel.
- **Format:** 32-bit floating-point PCM, values in `[-1.0, 1.0]`.

### Loading from WAV Files

```csharp
// The simplest way to load audio:
var audio = AudioIO.LoadWav("path/to/file.wav");
```

`AudioIO.LoadWav` handles WAV header parsing and returns a properly constructed `AudioData` instance. Ensure your WAV file is 16 kHz mono — or resample/convert beforehand with tools like `ffmpeg`:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

### Output: `IEnumerable<string>`

`Transcribe()` returns one string per input `AudioData`:

| Scenario | Output |
|---|---|
| Clear speech detected | `"Turn on the living room lights."` |
| No speech / silence / noise | `""` (empty string) |
| Multiple audio inputs | One string per input, in order |

## Use Cases

- **Voice command processing** — Whisper Tiny's speed makes it ideal for near-real-time voice-to-text in command-and-control interfaces.
- **Edge & IoT deployment** — Smallest Whisper variant; fits on resource-constrained devices with minimal RAM and disk.
- **Rapid prototyping** — Fast download (~144 MB) and inference lets you prototype speech features quickly before committing to a larger model.
- **CI/CD testing** — Use Tiny in automated tests to validate ASR pipelines without the download overhead of larger models.
- **Meeting transcription** — Acceptable accuracy for clear, single-speaker English recordings.
- **Accessibility** — Provide captions for hearing-impaired users where speed is prioritized over perfect accuracy.
- **Multi-language transcription** — Supports 99 languages, though accuracy is best on high-resource languages like English, Spanish, and French.

## Limitations & Considerations

| Limitation | Details |
|---|---|
| **Sample rate** | Input audio **must** be 16 kHz mono. Other rates/channels will produce incorrect results. |
| **No streaming** | Batch-only inference. The entire audio clip must be available before transcription starts. |
| **Synthetic audio** | Non-speech audio (sine waves, music, noise) produces empty transcriptions — this is expected behavior. |
| **No timestamps** | The API returns plain text only. Word-level or segment-level timestamps are not exposed. |
| **Max tokens** | Output is capped at 256 tokens per audio input. Very long audio clips may be truncated. |
| **Accuracy** | Tiny is the least accurate Whisper variant. It struggles more with accented speech, noisy environments, and low-resource languages compared to Base or larger models. |
| **Download size** | ~144 MB downloaded on first use. Smaller than Base (~278 MB) but still requires planning for offline/air-gapped deployments. |
| **Memory** | Expect ~300–500 MB RAM usage at inference time (lower than Base). |

### Tiny vs. Base

| | Whisper Tiny (this) | Whisper Base |
|---|---|---|
| **Parameters** | 39M | 74M |
| **Download** | ~144 MB | ~278 MB |
| **Speed** | ~2× faster | Baseline |
| **Accuracy** | Lower — best on clear, single-speaker audio | Higher — better on accented speech, noise, and non-English |
| **Use when** | Latency/size matters more than accuracy | Accuracy matters more than latency |

> **Rule of thumb:** Start with Tiny for prototyping and switch to Base (or larger) if transcription quality isn't meeting your needs.

## Example: Transcribing a WAV File

```csharp
using DotnetAILab.ModelGarden.ASR.WhisperTiny;
using MLNet.Audio.Core;
using System.Diagnostics;

// Pre-download the model at startup (optional, avoids cold-start latency later)
await WhisperTinyModel.EnsureFilesAsync();

// Load a real WAV file (must be 16 kHz, mono)
var audio = AudioIO.LoadWav("voice-note.wav");
Console.WriteLine($"Loaded audio: {audio.Duration.TotalSeconds:F1}s, {audio.SampleRate} Hz, {audio.Samples.Length} samples");

// Create the transcription pipeline
var stt = await WhisperTinyModel.CreateSpeechToTextAsync(language: "en");

// Transcribe and measure performance
var sw = Stopwatch.StartNew();
var results = stt.Transcribe([audio]);
sw.Stop();

Console.WriteLine($"Transcription completed in {sw.ElapsedMilliseconds} ms:\n");
foreach (var text in results)
{
    if (!string.IsNullOrWhiteSpace(text))
        Console.WriteLine($"  \"{text}\"");
    else
        Console.WriteLine("  (no speech detected)");
}

// Print model info
var info = await WhisperTinyModel.GetModelInfoAsync();
Console.WriteLine($"\nModel: {info.ModelId}");
Console.WriteLine($"Source: {info.ResolvedSource}");

stt.Dispose();
```

### Transcribing Multiple Files

```csharp
var files = Directory.GetFiles("recordings", "*.wav");
var audioClips = files.Select(f => AudioIO.LoadWav(f));

var stt = await WhisperTinyModel.CreateSpeechToTextAsync(language: "en");
var transcriptions = stt.Transcribe(audioClips);

foreach (var (file, text) in files.Zip(transcriptions))
    Console.WriteLine($"{Path.GetFileName(file)}: {text}");

stt.Dispose();
```

### Switching from Tiny to Base

If you find Tiny's accuracy insufficient, upgrading to Base requires changing just one line:

```diff
- using DotnetAILab.ModelGarden.ASR.WhisperTiny;
+ using DotnetAILab.ModelGarden.ASR.WhisperBase;

- var stt = await WhisperTinyModel.CreateSpeechToTextAsync(language: "en");
+ var stt = await WhisperBaseModel.CreateSpeechToTextAsync(language: "en");
```

The rest of your code (audio loading, `Transcribe()` call, output handling) stays exactly the same — both models share an identical API surface.

## Related Models

| Model | Package | Parameters | Best For |
|---|---|---|---|
| [**Whisper Base**](../DotnetAILab.ModelGarden.ASR.WhisperBase/README.md) | `DotnetAILab.ModelGarden.ASR.WhisperBase` | 74M | Balanced accuracy and performance |
| **Whisper Tiny** (this) | `DotnetAILab.ModelGarden.ASR.WhisperTiny` | 39M | Fastest inference, smallest footprint |

## References

- **Whisper paper:** Radford, A., Kim, J.W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). [*Robust Speech Recognition via Large-Scale Weak Supervision*](https://arxiv.org/abs/2212.04356). arXiv:2212.04356.
- **HuggingFace model card:** [onnx-community/whisper-tiny](https://huggingface.co/onnx-community/whisper-tiny)
- **OpenAI Whisper GitHub:** [github.com/openai/whisper](https://github.com/openai/whisper)
- **ONNX Runtime:** [onnxruntime.ai](https://onnxruntime.ai/)
- **ML.NET:** [dot.net/ml](https://dot.net/ml)
