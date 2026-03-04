# Audio Classification Sample

> Classify audio clips into hundreds of sound categories using the [Audio Spectrogram Transformer (AST)](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) model fine-tuned on AudioSet.

## Prerequisites

- [.NET 10 SDK (preview)](https://dotnet.microsoft.com/download/dotnet/10.0)
- Internet connection (the model downloads ~347 MB on first use)

## What This Sample Demonstrates

Audio classification assigns one or more labels to an audio clip describing what sounds it contains (e.g., "Speech", "Music", "Dog bark"). This sample:

1. Generates a synthetic 440 Hz tone as a stand-in for real audio.
2. Creates an AST-based audio classifier.
3. Classifies the audio and displays the top-5 predicted categories with their scores.

## Running the Sample

```sh
cd dotnet-model-garden-prototype
dotnet run --project samples/AudioClassificationSample
```

## Expected Output

```
=== Model Garden: Audio Classification Sample ===

Generating synthetic audio (440Hz sine wave, 1 second)...
Audio: 1.0s, 16000Hz

Creating audio classifier (downloads on first use)...
Classifier ready in 11022ms

Classifying audio...
Classification completed in 1403ms

Top predictions:
  → Speech (score: 0.1702)
     Speech: 0.1702
     Music: 0.1177
     Inside, small room: 0.0313
     Silence: 0.0307
     Musical instrument: 0.0272

Model info:
  Model ID: MIT/ast-finetuned-audioset-10-10-0.4593

Done!
```

## Code Walkthrough

1. **Generate synthetic audio** — Creates a 1-second, 16 kHz, 440 Hz sine wave in an `AudioData` object. In a real application you would load a WAV file with `AudioIO.LoadWav("sound.wav")`.
2. **Create the classifier** — `ASTAudioSetModel.CreateClassifierAsync()` downloads the ONNX model on first run and returns a classifier.
3. **Classify** — `classifier.Classify([audio])` runs inference and returns results containing the predicted label, score, and full probability distribution over all 527 AudioSet categories.
4. **Display top predictions** — The result's `Probabilities` and `Labels` arrays are sorted to show the five most likely categories.
5. **Show model info** — Prints the model ID and source.
6. **Dispose** — Releases ONNX Runtime resources.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Audio Classification** | Assigning category labels to an audio clip based on its content (speech, music, environmental sounds, etc.). |
| **Audio Spectrogram Transformer (AST)** | A vision-transformer-based model that converts audio into a spectrogram image and classifies it. |
| **AudioSet** | A large-scale dataset of 527 sound event categories used to train and evaluate audio classifiers. |
| **Score / Probability** | The model's confidence for each label. Higher scores indicate the model is more certain that category is present. |

## Next Steps

- **Use real audio** — Replace the synthetic tone with a WAV file: `var audio = AudioIO.LoadWav("sound.wav");`
- **Sound event detection** — Use the probabilities to detect multiple overlapping sound events in a single clip.
- **Filter by threshold** — Set a confidence threshold to only surface high-confidence predictions.
- Explore the [VADSample](../VADSample/) for detecting speech vs. silence, or the [ASRSample](../ASRSample/) to transcribe detected speech.

## Model Package References

- [`DotnetAILab.ModelGarden.AudioClassification.ASTAudioSet`](../../models/audioclassification/DotnetAILab.ModelGarden.AudioClassification.ASTAudioSet/) — AST AudioSet ONNX model package
- [Hugging Face: MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
