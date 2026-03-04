# Silero VAD v4 — Voice Activity Detection

> Lightweight voice activity detection model for identifying speech segments in audio.

## Overview

Voice Activity Detection (VAD) identifies **which parts of an audio signal contain speech** and which are silence or background noise. [Silero VAD](https://github.com/snakers4/silero-vad) is a compact, high-quality ONNX model that achieves near-state-of-the-art accuracy at a fraction of the size of larger models — only **~1.8 MB**.

This package wraps Silero VAD v4 as a NuGet model package. The ONNX binary is **not** included in the NuGet — it downloads transparently on first use and is cached locally for subsequent calls. This makes it an ideal preprocessing step for ASR, TTS, or any pipeline that benefits from skipping silence.

## Model Details

| Property | Value |
|---|---|
| **Model ID** | `snakers4/silero-vad` |
| **Size** | ~1.8 MB (very lightweight) |
| **Format** | ONNX |
| **Sample Rate** | 16 kHz mono |
| **Window Size** | 512 samples |
| **License** | MIT |
| **Source** | [HuggingFace](https://huggingface.co/lquint/silero-vad-v4-onnx) / [GitHub](https://github.com/snakers4/silero-vad) |

## Installation

```bash
dotnet add package DotnetAILab.ModelGarden.VAD.SileroVad
```

> **NuGet source** — this package is published to GitHub Packages. See the [root README](../../../README.md#nuget-source-setup) for source configuration.

## Quick Start

```csharp
using DotnetAILab.ModelGarden.VAD.SileroVad;
using MLNet.Audio.Core;

// Load your audio (16 kHz mono)
// var audio = AudioIO.LoadWav("recording.wav");
var audio = new AudioData(samples, sampleRate: 16000);

// Create the VAD — model downloads (~1.8 MB) on first call, cached thereafter
var vad = await SileroVadModel.CreateVadAsync(threshold: 0.5f);

// Detect speech segments
var segments = vad.DetectSpeech(audio);

foreach (var seg in segments)
    Console.WriteLine($"[{seg.Start:mm\\:ss\\.fff} → {seg.End:mm\\:ss\\.fff}] confidence: {seg.Confidence:F3}");

vad.Dispose();
```

## API Reference

### `SileroVadModel.CreateVadAsync`

```csharp
public static async Task<OnnxVadTransformer> CreateVadAsync(
    float threshold = 0.5f,
    ModelOptions? options = null,
    CancellationToken ct = default)
```

Creates an `OnnxVadTransformer` for voice activity detection. The returned object **also implements `IVoiceActivityDetector`**. Downloads the model on first call; cached thereafter.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `threshold` | `float` | `0.5f` | Speech probability threshold (0–1). Lower values detect more speech; higher values are more selective. |
| `options` | `ModelOptions?` | `null` | Override download source, cache path, etc. |
| `ct` | `CancellationToken` | `default` | Cancellation token. |

**Returns:** `OnnxVadTransformer` — call `DetectSpeech(AudioData)` to get speech segments.

**Internal defaults:**

| Setting | Value |
|---|---|
| MinSpeechDuration | 250 ms |
| MinSilenceDuration | 100 ms |
| SpeechPad | 30 ms |
| WindowSize | 512 samples |
| SampleRate | 16 000 Hz |

---

### `SileroVadModel.CreateVoiceActivityDetectorAsync`

```csharp
public static async Task<IVoiceActivityDetector> CreateVoiceActivityDetectorAsync(
    float threshold = 0.5f,
    ModelOptions? options = null,
    CancellationToken ct = default)
```

Convenience method that returns the VAD typed as `IVoiceActivityDetector`. Identical to `CreateVadAsync` but returns the interface type directly — useful for dependency injection or abstraction.

---

### Utility Methods

```csharp
// Pre-download / cache the model files without creating a detector
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

### Output

`SpeechSegment[]` — each element contains:

| Property | Type | Description |
|---|---|---|
| `Start` | `TimeSpan` | Start time of the speech segment |
| `End` | `TimeSpan` | End time of the speech segment |
| `Confidence` | `float` | Average speech probability for the segment (0–1) |

### `IVoiceActivityDetector` Interface

`OnnxVadTransformer` implements `IVoiceActivityDetector`, which exposes:

```csharp
SpeechSegment[] DetectSpeech(AudioData audio);
```

Use the interface when you want to swap in a different VAD implementation without changing calling code.

## Use Cases

- **Pre-filtering audio before ASR** — skip silence and reduce transcription cost/time.
- **Call center analytics** — segment agent vs. customer speech, measure talk-time ratios.
- **Meeting segmentation** — split recordings into speaker turns or active discussion blocks.
- **Real-time voice detection** — detect when a user starts/stops speaking in an interactive application.
- **Audio trimming / editing** — automatically remove leading/trailing silence from recordings.

## Limitations

- **16 kHz mono only** — audio must be resampled before processing if recorded at a different rate.
- **Batch processing** — the current API processes a complete `AudioData` buffer; it is not a streaming/real-time API.
- **Threshold tuning** — the default `0.5` works well for clean speech, but noisy environments may require adjustment.
- **Edge cases** — at ~1.8 MB the model trades size for occasional misses on very short utterances or unusual background noise.

## Example: Processing a Long Recording

```csharp
using DotnetAILab.ModelGarden.VAD.SileroVad;
using MLNet.Audio.Core;

// Load a long recording
// var audio = AudioIO.LoadWav("meeting_recording.wav");
var audio = new AudioData(samples, sampleRate: 16000);

var vad = await SileroVadModel.CreateVadAsync(threshold: 0.4f);
var segments = vad.DetectSpeech(audio);

Console.WriteLine($"Total duration: {audio.Duration}");
Console.WriteLine($"Speech segments: {segments.Length}");

var totalSpeech = TimeSpan.Zero;
foreach (var seg in segments)
{
    var duration = seg.End - seg.Start;
    totalSpeech += duration;
    Console.WriteLine($"  [{seg.Start:mm\\:ss\\.fff} → {seg.End:mm\\:ss\\.fff}] " +
                      $"duration: {duration.TotalSeconds:F2}s, confidence: {seg.Confidence:F3}");
}

var speechRatio = totalSpeech.TotalSeconds / audio.Duration.TotalSeconds;
Console.WriteLine($"\nSpeech: {totalSpeech.TotalSeconds:F1}s / {audio.Duration.TotalSeconds:F1}s ({speechRatio:P0})");

vad.Dispose();
```

## References

- [Silero VAD — GitHub](https://github.com/snakers4/silero-vad)
- [Silero VAD ONNX — HuggingFace](https://huggingface.co/lquint/silero-vad-v4-onnx)
- [Silero Models paper](https://arxiv.org/abs/2106.04624) — *Silero VAD: pre-trained enterprise-grade Voice Activity Detector*
- [.NET Model Garden — root README](../../../README.md)
