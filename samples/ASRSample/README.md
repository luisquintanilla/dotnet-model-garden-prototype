# ASR (Speech-to-Text) Sample

> Transcribe audio to text using [OpenAI Whisper Base](https://huggingface.co/openai/whisper-base), a general-purpose automatic speech recognition model.

## Prerequisites

- [.NET 10 SDK (preview)](https://dotnet.microsoft.com/download/dotnet/10.0)
- Internet connection (the model downloads ~290 MB on first use)

## What This Sample Demonstrates

Automatic Speech Recognition (ASR) converts spoken audio into written text. This sample:

1. Generates a synthetic audio signal (a 440 Hz sine wave) as a placeholder for real speech.
2. Creates a Whisper Base speech-to-text model with language set to English.
3. Runs transcription and displays the result along with timing information.

> **Note:** Because the sample uses a synthetic tone rather than real speech, the transcription result will be empty. Replace the synthetic audio with a real `.wav` file to see meaningful output.

## Running the Sample

```sh
cd dotnet-model-garden-prototype
dotnet run --project samples/ASRSample
```

## Expected Output

```
=== Model Garden: ASR (Speech-to-Text) Sample ===

Generating synthetic audio (440Hz sine wave, 1 second)...
Audio: 1.0s, 16000Hz, 16000 samples

Creating speech-to-text model (downloads on first use)...
Model ready in 9342ms

Transcribing...
Transcription completed in 341ms

  Result: ""

Model info:
  Model ID: openai/whisper-base
  Source: huggingface

Done!
```

## Code Walkthrough

1. **Generate synthetic audio** — Creates a 1-second, 16 kHz, 440 Hz sine wave wrapped in an `AudioData` object. In a real application you would load a WAV file with `AudioIO.LoadWav("speech.wav")`.
2. **Create the speech-to-text model** — `WhisperBaseModel.CreateSpeechToTextAsync(language: "en")` downloads the ONNX model on first run and configures it for English transcription.
3. **Transcribe** — `stt.Transcribe([audio])` runs inference on the audio and returns a list of transcribed strings.
4. **Display results** — Prints the transcription text and elapsed time.
5. **Show model info** — `WhisperBaseModel.GetModelInfoAsync()` returns metadata including the model ID and source.
6. **Dispose** — The model is disposed to release ONNX Runtime resources.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **ASR / Speech-to-Text** | The task of converting audio containing speech into written text. |
| **Whisper** | A transformer-based ASR model by OpenAI trained on 680,000 hours of multilingual audio. |
| **AudioData** | A lightweight struct from `MLNet.Audio.Core` holding raw float samples and a sample rate. |
| **Language parameter** | Specifying the language (e.g., `"en"`) skips Whisper's language detection step and can improve accuracy. |

## Next Steps

- **Use real audio** — Replace the synthetic signal with a WAV file: `var audio = AudioIO.LoadWav("speech.wav");`
- **Try other languages** — Pass a different language code to `CreateSpeechToTextAsync` (e.g., `"fr"`, `"de"`, `"ja"`).
- **Batch transcription** — Pass multiple `AudioData` objects to `Transcribe` for batch processing.
- Explore the [VADSample](../VADSample/) to detect speech segments before transcription.

## Model Package References

- [`DotnetAILab.ModelGarden.ASR.WhisperBase`](../../models/asr/DotnetAILab.ModelGarden.ASR.WhisperBase/) — Whisper Base ONNX model package
- [Hugging Face: openai/whisper-base](https://huggingface.co/openai/whisper-base)
