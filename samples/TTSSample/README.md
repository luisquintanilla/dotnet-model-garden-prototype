# TTS (Text-to-Speech) Sample

> Synthesize natural-sounding speech from text using the [Microsoft SpeechT5](https://huggingface.co/microsoft/speecht5_tts) model and save the result as a WAV file.

## Prerequisites

- [.NET 10 SDK (preview)](https://dotnet.microsoft.com/download/dotnet/10.0)
- Internet connection (the model downloads ~643 MB on first use)

## What This Sample Demonstrates

Text-to-Speech (TTS) converts written text into spoken audio. This sample:

1. Creates a SpeechT5 TTS client.
2. Synthesizes speech from a text string.
3. Reports audio duration, sample rate, and sample count.
4. Saves the output as a standard WAV file.

## Running the Sample

```sh
cd dotnet-model-garden-prototype
dotnet run --project samples/TTSSample
```

## Expected Output

```
=== Model Garden: TTS (Text-to-Speech) Sample ===

Creating text-to-speech client (downloads on first use)...
TTS ready in 21236ms

Synthesizing: "Hello! This is a test of the SpeechT5 text to speech model from the dotnet model garden."
Synthesis completed in 8475ms

  Duration:    8.00s
  Sample rate: 16000Hz
  Samples:     128000

Saving to tts_output.wav...
Saved! File size: 256,044 bytes

Model info:
  Model ID: microsoft/speecht5_tts

Done!
```

## Code Walkthrough

1. **Create the TTS client** — `SpeechT5Model.CreateTextToSpeechClientAsync()` downloads the ONNX model (~643 MB) on first run and returns a ready-to-use TTS client.
2. **Synthesize speech** — `tts.GetAudioAsync(text)` converts the input string into audio, returning a response containing the generated `AudioData` and duration.
3. **Inspect the result** — The response provides the audio duration, sample rate (16 kHz), and total sample count.
4. **Save to WAV** — `AudioIO.SaveWav(outputPath, response.Audio)` writes the audio to a standard WAV file that can be played in any audio player.
5. **Show model info** — Prints the model ID and source.
6. **Dispose** — Releases ONNX Runtime resources.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Text-to-Speech (TTS)** | The task of generating natural-sounding spoken audio from written text. |
| **SpeechT5** | A Microsoft encoder-decoder model that unifies speech and text tasks including TTS, ASR, and speech translation. |
| **Sample Rate** | The number of audio samples per second (16,000 Hz in this model). Higher rates capture more detail but produce larger files. |
| **WAV Format** | An uncompressed audio file format that stores raw PCM samples. It is widely supported and easy to work with. |

## Next Steps

- **Try different texts** — Experiment with longer or shorter sentences, different punctuation, and various speaking styles.
- **Play the output** — Open `tts_output.wav` in any audio player or integrate playback into your application.
- **Pipeline with VAD/ASR** — Combine TTS with the [ASRSample](../ASRSample/) to create a round-trip text → speech → text pipeline for testing.
- **Streaming synthesis** — For long texts, investigate chunking input text and synthesizing segments incrementally.
- Explore the [ASRSample](../ASRSample/) for the reverse task — converting speech back to text.

## Model Package References

- [`DotnetAILab.ModelGarden.TTS.SpeechT5`](../../models/tts/DotnetAILab.ModelGarden.TTS.SpeechT5/) — SpeechT5 ONNX model package
- [Hugging Face: microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts)
