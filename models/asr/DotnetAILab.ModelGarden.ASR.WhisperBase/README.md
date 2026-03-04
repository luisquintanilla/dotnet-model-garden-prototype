# DotnetAILab.ModelGarden.ASR.WhisperBase

> OpenAI Whisper Base — a 74M-parameter speech-to-text model for .NET, powered by ONNX Runtime.

## Overview

**Whisper Base** is a general-purpose automatic speech recognition (ASR) model originally developed by OpenAI and packaged here for seamless use in .NET applications. It converts spoken audio into text across 99 languages with no external service calls — all inference runs locally on your machine using ONNX Runtime.

Under the hood, Whisper is an **encoder-decoder Transformer** trained on 680,000 hours of multilingual and multitask supervised audio from the web. The encoder processes a log-Mel spectrogram of the input audio (80 mel bins, 16 kHz sample rate) and produces a sequence of hidden states. The decoder then auto-regressively generates text tokens conditioned on those hidden states and a language/task prompt. This architecture allows the model to perform multilingual transcription and translation in a single forward pass.

The Base variant strikes a practical balance between accuracy and resource usage. With 74 million parameters and a ~278 MB download, it is well-suited for production workloads where you need reliable multi-language transcription without the memory footprint of the larger Whisper models (Small, Medium, Large). If you need the absolute fastest inference and can tolerate lower accuracy, see the [Whisper Tiny](#related-models) variant instead.

## Model Details

| Property | Value |
|---|---|
| **Model ID** | `openai/whisper-base` |
| **Architecture** | Whisper encoder-decoder Transformer |
| **Parameters** | 74 million |
| **License** | Apache 2.0 |
| **Source** | [onnx-community/whisper-base](https://huggingface.co/onnx-community/whisper-base) (HuggingFace) |
| **Runtime** | ONNX Runtime (via ML.NET) |
| **Download Size** | ~278 MB (encoder 78.6 MB + decoder 199 MB) |
| **Target Framework** | .NET 10+ |
| **Languages** | 99 languages (Afrikaans, Arabic, Chinese, English, French, German, Hindi, Japanese, Korean, Spanish, and [many more](https://github.com/openai/whisper#available-models-and-languages)) |
| **Mel Bins** | 80 |
| **Max Tokens** | 256 |
| **Sample Rate** | 16,000 Hz |

## Installation

```bash
dotnet add package DotnetAILab.ModelGarden.ASR.WhisperBase
```

The model weights (~278 MB) are **automatically downloaded** from HuggingFace on first use and cached locally. No manual model management is required.

### Dependencies

This package transitively brings in:

- `ModelPackages` (v0.1.0-preview.14) — model download, caching, and verification
- `MLNet.AudioInference.Onnx` (v0.1.0-preview.2) — ONNX-based Whisper inference via ML.NET

## Quick Start

```csharp
using DotnetAILab.ModelGarden.ASR.WhisperBase;
using MLNet.Audio.Core;

// 1. Load audio (16 kHz, mono, float32 PCM)
var audio = AudioIO.LoadWav("meeting-recording.wav");

// 2. Create the speech-to-text pipeline (downloads model on first call)
var stt = await WhisperBaseModel.CreateSpeechToTextAsync(language: "en");

// 3. Transcribe
var results = stt.Transcribe([audio]);

foreach (var text in results)
    Console.WriteLine(text);

// 4. Clean up
stt.Dispose();
```

**Output:**
```
Good morning everyone, let's start the standup with updates from the backend team.
```

## API Reference

### `WhisperBaseModel.CreateSpeechToTextAsync(...)`

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

Runs inference on one or more audio inputs. This method is on the returned transformer, not on `WhisperBaseModel` directly.

```csharp
public IEnumerable<string> Transcribe(IEnumerable<AudioData> audioInputs)
```

| Parameter | Type | Description |
|---|---|---|
| `audioInputs` | `IEnumerable<AudioData>` | One or more audio samples to transcribe. Each must be 16 kHz mono float32 PCM. |

**Returns:** `IEnumerable<string>` — one transcription string per input audio. Empty strings indicate no speech was detected.

---

### `WhisperBaseModel.EnsureFilesAsync(...)`

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

### `WhisperBaseModel.GetModelInfoAsync(...)`

Retrieves metadata about the model package without downloading the full model.

```csharp
public static Task<ModelInfo> GetModelInfoAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

**Returns:** `Task<ModelInfo>` — contains `ModelId` (`"openai/whisper-base"`), `ResolvedSource`, and other metadata.

---

### `WhisperBaseModel.VerifyModelAsync(...)`

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
| Clear speech detected | `"Hello, how are you today?"` |
| No speech / silence / noise | `""` (empty string) |
| Multiple audio inputs | One string per input, in order |

## Use Cases

- **Meeting transcription** — Record meetings and generate searchable text transcripts for archival and action-item extraction.
- **Podcast & video subtitling** — Automatically generate subtitles or closed captions for media content.
- **Voice command processing** — Convert spoken commands to text for further NLU/intent classification.
- **Accessibility** — Provide real-time or batch captions for hearing-impaired users in applications.
- **Multi-language transcription** — Transcribe audio in any of 99 supported languages by setting the `language` parameter.
- **Call center analytics** — Transcribe customer service calls for sentiment analysis, compliance, and quality assurance.
- **Note-taking apps** — Convert voice memos and dictation into structured text notes.

## Limitations & Considerations

| Limitation | Details |
|---|---|
| **Sample rate** | Input audio **must** be 16 kHz mono. Other rates/channels will produce incorrect results. |
| **No streaming** | Batch-only inference. The entire audio clip must be available before transcription starts. |
| **Synthetic audio** | Non-speech audio (sine waves, music, noise) produces empty transcriptions — this is expected behavior. |
| **No timestamps** | The API returns plain text only. Word-level or segment-level timestamps are not exposed. |
| **Max tokens** | Output is capped at 256 tokens per audio input. Very long audio clips may be truncated. |
| **Download size** | ~278 MB downloaded on first use. Plan for this in CI/CD pipelines and air-gapped environments. |
| **Memory** | The model loads two ONNX graphs (encoder + decoder) into memory. Expect ~500–800 MB RAM usage at inference time. |

### Base vs. Tiny

| | Whisper Base | Whisper Tiny |
|---|---|---|
| **Parameters** | 74M | 39M |
| **Download** | ~278 MB | ~144 MB |
| **Accuracy** | Higher — better on accented speech, noisy audio, and low-resource languages | Lower — best on clear, English-dominant audio |
| **Speed** | Baseline | ~2× faster |
| **Use when** | Accuracy matters more than latency | Latency/size matters more than accuracy |

## Example: Transcribing a WAV File

```csharp
using DotnetAILab.ModelGarden.ASR.WhisperBase;
using MLNet.Audio.Core;
using System.Diagnostics;

// Pre-download the model at startup (optional, avoids cold-start latency later)
await WhisperBaseModel.EnsureFilesAsync();

// Load a real WAV file (must be 16 kHz, mono)
var audio = AudioIO.LoadWav("interview.wav");
Console.WriteLine($"Loaded audio: {audio.Duration.TotalSeconds:F1}s, {audio.SampleRate} Hz, {audio.Samples.Length} samples");

// Create the transcription pipeline
var stt = await WhisperBaseModel.CreateSpeechToTextAsync(language: "en");

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
var info = await WhisperBaseModel.GetModelInfoAsync();
Console.WriteLine($"\nModel: {info.ModelId}");
Console.WriteLine($"Source: {info.ResolvedSource}");

stt.Dispose();
```

### Transcribing Multiple Files

```csharp
var files = Directory.GetFiles("recordings", "*.wav");
var audioClips = files.Select(f => AudioIO.LoadWav(f));

var stt = await WhisperBaseModel.CreateSpeechToTextAsync(language: "en");
var transcriptions = stt.Transcribe(audioClips);

foreach (var (file, text) in files.Zip(transcriptions))
    Console.WriteLine($"{Path.GetFileName(file)}: {text}");

stt.Dispose();
```

### Multi-Language Transcription

```csharp
// Transcribe French audio
var sttFr = await WhisperBaseModel.CreateSpeechToTextAsync(language: "fr");
var frenchResults = sttFr.Transcribe([frenchAudio]);

// Transcribe Japanese audio
var sttJa = await WhisperBaseModel.CreateSpeechToTextAsync(language: "ja");
var japaneseResults = sttJa.Transcribe([japaneseAudio]);
```

## Related Models

| Model | Package | Parameters | Best For |
|---|---|---|---|
| **Whisper Base** (this) | `DotnetAILab.ModelGarden.ASR.WhisperBase` | 74M | Balanced accuracy and performance |
| [**Whisper Tiny**](../DotnetAILab.ModelGarden.ASR.WhisperTiny/README.md) | `DotnetAILab.ModelGarden.ASR.WhisperTiny` | 39M | Fastest inference, smallest footprint |

## References

- **Whisper paper:** Radford, A., Kim, J.W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). [*Robust Speech Recognition via Large-Scale Weak Supervision*](https://arxiv.org/abs/2212.04356). arXiv:2212.04356.
- **HuggingFace model card:** [onnx-community/whisper-base](https://huggingface.co/onnx-community/whisper-base)
- **OpenAI Whisper GitHub:** [github.com/openai/whisper](https://github.com/openai/whisper)
- **ONNX Runtime:** [onnxruntime.ai](https://onnxruntime.ai/)
- **ML.NET:** [dot.net/ml](https://dot.net/ml)
