# VAD (Voice Activity Detection) Sample

> Detect speech segments in audio using [Silero VAD v4](https://github.com/snakers4/silero-vad), a lightweight and fast voice activity detection model.

## Prerequisites

- [.NET 10 SDK (preview)](https://dotnet.microsoft.com/download/dotnet/10.0)
- Internet connection (the model downloads ~2 MB on first use)

## What This Sample Demonstrates

Voice Activity Detection (VAD) identifies **when** someone is speaking in an audio stream. This sample:

1. Generates synthetic audio with tone bursts separated by silence to simulate speech and pauses.
2. Creates a Silero VAD model with a configurable confidence threshold.
3. Detects speech segments and reports their start/end timestamps and confidence scores.

## Running the Sample

```sh
cd dotnet-model-garden-prototype
dotnet run --project samples/VADSample
```

## Expected Output

```
=== Model Garden: VAD (Voice Activity Detection) Sample ===

Generating synthetic audio (tone bursts with silence gaps)...
Audio: 3.0s, 16000Hz

Creating voice activity detector (downloads on first use)...
VAD ready in 1329ms

Detecting speech segments...
Detection completed in 58ms

Found 1 speech segment(s):
  [00:00.514 → 00:01.182] confidence: 0.555

Model info:
  Model ID: snakers4/silero-vad

Done!
```

## Code Walkthrough

1. **Generate synthetic audio** — Creates a 3-second, 16 kHz signal with two tone bursts (440 Hz at 0.5–1.5 s and 880 Hz at 2.0–2.5 s) separated by silence. In a real application you would load a WAV file with `AudioIO.LoadWav("recording.wav")`.
2. **Create the VAD model** — `SileroVadModel.CreateVadAsync(threshold: 0.5f)` downloads the tiny ONNX model (~2 MB) on first run. The `threshold` parameter controls sensitivity: lower values detect more speech at the cost of more false positives.
3. **Detect speech segments** — `vad.DetectSpeech(audio)` processes the audio and returns an array of segments, each with a start time, end time, and confidence score.
4. **Display results** — Prints each detected segment with formatted timestamps and confidence.
5. **Show model info** — Prints the model ID and source.
6. **Dispose** — Releases ONNX Runtime resources.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Voice Activity Detection (VAD)** | A binary classification task that determines whether a given audio frame contains speech or silence. |
| **Silero VAD** | A compact, production-ready VAD model that runs efficiently on CPU with very low latency. |
| **Threshold** | The minimum confidence score required to consider a frame as speech. Adjusting this trades off between recall (catching all speech) and precision (avoiding false alarms). |
| **Speech Segment** | A contiguous time interval where the model detects speech, defined by a start time, end time, and average confidence. |

## Next Steps

- **Use real audio** — Replace the synthetic signal with a WAV file: `var audio = AudioIO.LoadWav("recording.wav");`
- **Tune the threshold** — Experiment with values between 0.3 and 0.8 to find the right sensitivity for your use case.
- **Pre-process for ASR** — Use detected speech segments to extract only the spoken portions before sending them to a speech-to-text model (see [ASRSample](../ASRSample/)).
- **Streaming VAD** — Process audio in chunks for real-time applications like live captioning or push-to-talk.

## Model Package References

- [`DotnetAILab.ModelGarden.VAD.SileroVad`](../../models/vad/DotnetAILab.ModelGarden.VAD.SileroVad/) — Silero VAD v4 ONNX model package
- [GitHub: snakers4/silero-vad](https://github.com/snakers4/silero-vad)
