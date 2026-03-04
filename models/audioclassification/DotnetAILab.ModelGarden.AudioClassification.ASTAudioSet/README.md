# AST AudioSet — Audio Classification

> Audio Spectrogram Transformer fine-tuned on AudioSet for 527-class sound classification.

[![NuGet](https://img.shields.io/badge/nuget-DotnetAILab.ModelGarden.AudioClassification.ASTAudioSet-blue)](https://nuget.pkg.github.com/luisquintanilla/index.json)

## Overview

The **Audio Spectrogram Transformer (AST)** is a convolution-free, purely attention-based architecture for audio classification. Originally introduced by Gong et al. (2021), AST applies the Vision Transformer (ViT) paradigm directly to audio by treating mel spectrogram patches as visual tokens. Instead of relying on hand-crafted audio features or convolutional front-ends, the model splits a mel spectrogram into a sequence of 16×16 patches, linearly projects each patch into an embedding, and feeds the resulting sequence through a standard Transformer encoder with self-attention. This allows the model to capture both local acoustic patterns and long-range temporal dependencies in a single, unified framework.

This package wraps the **AST model fine-tuned on AudioSet**, Google's large-scale audio event dataset containing over 2 million human-annotated 10-second sound clips spanning **527 classes**. The AudioSet ontology covers an extraordinarily broad range of everyday sounds — from human speech, laughter, and crying to musical instruments (guitar, piano, drums), animal vocalizations (dog bark, bird song, cat meow), environmental sounds (rain, thunder, wind), vehicle noises (car horn, train, helicopter), domestic sounds (doorbell, vacuum cleaner, keyboard), and many more. The fine-tuned checkpoint achieves a mean average precision (mAP) of 0.459 on the AudioSet evaluation set.

This NuGet package makes it trivial to add audio classification to any .NET application. The ~347 MB ONNX model binary downloads automatically on first use and is cached locally for subsequent calls. All mel spectrogram preprocessing is handled internally — you supply raw 16 kHz mono audio samples and receive classified labels with confidence scores. Typical use cases include environmental sound monitoring, content tagging for audio/video platforms, smart home audio triggers, wildlife monitoring, and accessibility applications.

## Model Details

| Property | Value |
|---|---|
| **Model ID** | `MIT/ast-finetuned-audioset-10-10-0.4593` |
| **Architecture** | Audio Spectrogram Transformer (AST / ViT) |
| **Parameters** | ~87M |
| **License** | BSD-3-Clause |
| **Source** | [HuggingFace — onnx-community/ast-finetuned-audioset-10-10-0.4593-ONNX](https://huggingface.co/onnx-community/ast-finetuned-audioset-10-10-0.4593-ONNX) |
| **Runtime** | ONNX Runtime via ML.NET |
| **Download Size** | ~347 MB (auto-downloaded on first use) |
| **Classes** | 527 AudioSet labels |
| **Sample Rate** | 16 kHz mono |
| **Mel Bins** | 128 |
| **FFT Size** | 400 |
| **Hop Length** | 160 |

## Installation

```bash
dotnet add package DotnetAILab.ModelGarden.AudioClassification.ASTAudioSet
```

> **Note:** This package is published to GitHub Packages. Ensure you have the GitHub NuGet source configured in your `nuget.config`:
>
> ```xml
> <packageSources>
>   <add key="github" value="https://nuget.pkg.github.com/luisquintanilla/index.json" />
> </packageSources>
> ```

## Quick Start

```csharp
using DotnetAILab.ModelGarden.AudioClassification.ASTAudioSet;
using Microsoft.ML;

// Create the classifier — model downloads automatically on first call
var classifier = await ASTAudioSetModel.CreateClassifierAsync();

// Load or generate 16 kHz mono audio samples
float[] audioSamples = LoadYourAudio(); // your audio loading logic

// Run classification
var mlContext = new MLContext();
var inputData = mlContext.Data.LoadFromEnumerable(new[]
{
    new { Audio = audioSamples }
});

var predictions = classifier.Transform(inputData);

// Read top-5 results
var results = mlContext.Data.CreateEnumerable<ClassificationResult>(predictions, reuseRowObject: false);
foreach (var result in results)
{
    Console.WriteLine("Top-5 predictions:");
    var topLabels = result.Labels
        .Zip(result.Scores)
        .OrderByDescending(x => x.Second)
        .Take(5);

    foreach (var (label, score) in topLabels)
    {
        Console.WriteLine($"  {label}: {score:P1}");
    }
}
```

## API Reference

### `ASTAudioSetModel.CreateClassifierAsync(...)`

Creates an `OnnxAudioClassificationTransformer` configured for AudioSet 527-class classification.

```csharp
public static async Task<OnnxAudioClassificationTransformer> CreateClassifierAsync(
    MLContext? mlContext = null,
    ModelOptions? options = null,
    CancellationToken ct = default)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mlContext` | `MLContext?` | `null` | ML.NET context. A new instance is created if `null`. |
| `options` | `ModelOptions?` | `null` | Options for model download behavior (cache location, source override, etc.). |
| `ct` | `CancellationToken` | `default` | Cancellation token for the async download. |

**Returns:** `OnnxAudioClassificationTransformer` — an ML.NET transformer that accepts audio input and produces classification results with 527 AudioSet labels and confidence scores.

**Behavior:**
- Downloads the ONNX model (~347 MB) on first call from HuggingFace. Subsequent calls use the cached copy.
- Internally configures a `MelSpectrogramExtractor` with: 128 mel bins, FFT size 400, hop length 160, and 16 kHz sample rate.
- All 527 AudioSet labels are embedded in the package — no external label files are needed.

**Exceptions:**
- `HttpRequestException` — if the model download fails (network error, 404, etc.).
- `InvalidOperationException` — if model integrity verification fails after download.

---

### `ASTAudioSetModel.EnsureFilesAsync(...)`

Downloads the model files if not already cached and returns the local paths.

```csharp
public static Task<ModelFiles> EnsureFilesAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `options` | `ModelOptions?` | `null` | Options for model download behavior. |
| `ct` | `CancellationToken` | `default` | Cancellation token. |

**Returns:** `ModelFiles` — contains the local path to the downloaded ONNX model file via `PrimaryModelPath`.

---

### `ASTAudioSetModel.GetModelInfoAsync(...)`

Retrieves metadata about the model package without downloading the model binary.

```csharp
public static Task<ModelInfo> GetModelInfoAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `options` | `ModelOptions?` | `null` | Options for model source resolution. |
| `ct` | `CancellationToken` | `default` | Cancellation token. |

**Returns:** `ModelInfo` — model ID, expected file sizes, SHA256 hashes, and source information.

---

### `ASTAudioSetModel.VerifyModelAsync(...)`

Verifies that the cached model files are intact by checking SHA256 hashes and file sizes.

```csharp
public static Task VerifyModelAsync(
    ModelOptions? options = null,
    CancellationToken ct = default)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `options` | `ModelOptions?` | `null` | Options for model location. |
| `ct` | `CancellationToken` | `default` | Cancellation token. |

**Exceptions:**
- `InvalidOperationException` — if the model file is missing, corrupt, or has a mismatched hash.

## Inputs & Outputs

### Input

The classifier expects **16 kHz mono audio** as a `float[]` array of normalized samples (values typically in the range `[-1.0, 1.0]`).

Internally, the pipeline converts raw audio into a **mel spectrogram** before feeding it to the AST model:

| Preprocessing Step | Configuration |
|---|---|
| **Sample Rate** | 16,000 Hz |
| **Mel Bins** | 128 |
| **FFT Size** | 400 (25 ms window at 16 kHz) |
| **Hop Length** | 160 (10 ms hop at 16 kHz) |
| **Feature Extractor** | `MelSpectrogramExtractor` |

The mel spectrogram is computed using a Short-Time Fourier Transform (STFT) with a 400-sample (25 ms) window and a 160-sample (10 ms) hop. The frequency axis is mapped to 128 mel-scale filter banks, producing a 128 × T time-frequency representation that the AST model consumes as a sequence of 16×16 patches.

### Output

The transformer produces classification results containing:

- **Labels** — all 527 AudioSet class names
- **Scores** — corresponding confidence scores (higher = more likely)

Results are not pre-sorted; use LINQ to sort by descending score to get top-K predictions.

## AudioSet Label Categories

The 527 labels span a remarkably broad taxonomy of sounds. Here are the major categories with examples:

### 🗣️ Human Sounds
Speech, Male speech, Female speech, Child speech, Conversation, Narration, Shout, Screaming, Whispering, Laughter, Baby laughter, Giggle, Crying/sobbing, Baby cry, Singing, Choir, Rapping, Humming, Whistling, Breathing, Snoring, Cough, Sneeze, Clapping, Cheering, Applause, Chewing, Footsteps

### 🎵 Music & Instruments
Musical instrument, Guitar, Electric guitar, Acoustic guitar, Bass guitar, Piano, Organ, Synthesizer, Drum kit, Drum, Snare drum, Bass drum, Cymbal, Hi-hat, Tambourine, Violin, Cello, Flute, Saxophone, Clarinet, Trumpet, Trombone, French horn, Harmonica, Accordion, Bagpipes, Harp, Banjo, Sitar, Ukulele, Mandolin

### 🎶 Music Genres
Pop music, Hip hop, Rock music, Heavy metal, Punk rock, Jazz, Classical music, Opera, Electronic music, House music, Techno, Dubstep, Country, Blues, Reggae, Soul music, Funk, Folk music, Disco, Ambient music, Gospel music, Flamenco, Ska

### 🐾 Animals
Dog, Bark, Howl, Growling, Cat, Purr, Meow, Horse, Neigh, Cattle, Moo, Pig, Oink, Goat, Bleat, Sheep, Chicken, Rooster, Duck, Quack, Goose, Bird, Chirp/tweet, Owl, Hoot, Crow, Frog, Croak, Snake, Cricket, Mosquito, Bee, Whale vocalization

### 🌧️ Environmental & Nature
Wind, Rustling leaves, Thunderstorm, Thunder, Rain, Raindrop, Stream, Waterfall, Ocean, Waves, Fire, Crackle

### 🚗 Vehicles & Transport
Car, Vehicle horn, Car alarm, Truck, Bus, Emergency vehicle, Police car (siren), Ambulance (siren), Fire engine (siren), Motorcycle, Train, Train whistle, Subway, Aircraft, Jet engine, Helicopter, Airplane, Bicycle, Skateboard, Boat, Motorboat, Ship

### 🏠 Domestic & Indoor Sounds
Door, Doorbell, Sliding door, Slam, Knock, Cupboard, Dishes/pots/pans, Microwave oven, Blender, Water tap, Hair dryer, Toilet flush, Vacuum cleaner, Keys jangling, Scissors, Typing, Computer keyboard, Alarm, Telephone, Ringtone, Alarm clock, Smoke detector

### 🔧 Mechanical & Industrial
Engine, Lawn mower, Chainsaw, Drill, Jackhammer, Hammer, Sawing, Power tool, Sewing machine, Mechanical fan, Air conditioning, Cash register, Printer, Camera

### 💥 Impact & Material Sounds
Explosion, Gunshot, Fireworks, Glass (shatter, clink), Wood (chop, crack), Liquid (splash, pour, drip), Crushing, Tearing, Bouncing, Scratch, Scrape

### 🔊 Acoustic Environments
Silence, Inside (small room), Inside (large room/hall), Inside (public space), Outside (urban), Outside (rural/natural), Reverberation, Echo, White noise, Pink noise, Static

## Use Cases

| Use Case | Description |
|---|---|
| **Environmental sound monitoring** | Detect rain, wind, thunder, or other weather events from outdoor audio feeds. |
| **Content tagging for audio/video** | Automatically tag video clips with sound labels (music genre, speech, applause) for search and discovery. |
| **Smart home audio triggers** | Trigger automations based on detected sounds — doorbell, smoke alarm, glass breaking, baby crying. |
| **Wildlife monitoring** | Identify bird species, insect activity, or animal vocalizations from field recordings. |
| **Industrial sound anomaly detection** | Monitor machinery for unusual sounds (engine knocking, abnormal vibrations) that may indicate faults. |
| **Accessibility** | Provide sound awareness notifications for hearing-impaired users — alert when a phone rings, someone knocks, or an alarm sounds. |
| **Music information retrieval** | Classify audio by genre, detect musical instruments, or identify vocal vs. instrumental segments. |
| **Security & surveillance** | Detect gunshots, glass breaking, screaming, or siren sounds in security audio feeds. |

## Limitations & Considerations

- **Fixed mel spectrogram configuration** — The model requires exactly 128 mel bins, FFT size 400, and hop length 160. These are configured internally by `CreateClassifierAsync()` and should not be changed.
- **16 kHz mono input required** — Audio must be resampled to 16,000 Hz mono before classification. Higher sample rates or stereo audio will produce incorrect results.
- **527-label taxonomy** — While AudioSet is comprehensive, it may not cover every conceivable sound. Niche or domain-specific sounds not in the ontology will be mapped to the nearest matching label.
- **Model download size (~347 MB)** — The ONNX model is downloaded on first use. Ensure adequate disk space and network connectivity. Subsequent calls use the cached local copy.
- **Classification confidence varies** — Audio quality, background noise, overlapping sounds, and recording distance all affect classification accuracy. The model works best with clear, well-recorded audio.
- **Multi-label nature** — Real-world audio often contains multiple simultaneous sounds. The model produces scores for all 527 labels; multiple labels may have high confidence simultaneously (e.g., "Speech" + "Background music").
- **Not a real-time streaming classifier** — The model processes complete audio segments. For real-time applications, you must buffer and segment the audio stream before classification.

## Example: Classifying Environmental Sounds

```csharp
using DotnetAILab.ModelGarden.AudioClassification.ASTAudioSet;
using Microsoft.ML;

// Create the classifier (downloads model on first run)
var classifier = await ASTAudioSetModel.CreateClassifierAsync();
var mlContext = new MLContext();

// Simulate different environmental audio sources
var audioClips = new Dictionary<string, float[]>
{
    ["rainstorm.wav"]  = LoadAudioFile("rainstorm.wav"),
    ["birdsong.wav"]   = LoadAudioFile("birdsong.wav"),
    ["traffic.wav"]    = LoadAudioFile("traffic.wav"),
    ["piano.wav"]      = LoadAudioFile("piano.wav"),
};

foreach (var (fileName, samples) in audioClips)
{
    var inputData = mlContext.Data.LoadFromEnumerable(new[]
    {
        new { Audio = samples }
    });

    var predictions = classifier.Transform(inputData);
    var results = mlContext.Data
        .CreateEnumerable<ClassificationResult>(predictions, reuseRowObject: false)
        .First();

    Console.WriteLine($"\n🎵 {fileName}:");
    var topLabels = results.Labels
        .Zip(results.Scores)
        .OrderByDescending(x => x.Second)
        .Take(5);

    foreach (var (label, score) in topLabels)
    {
        Console.WriteLine($"   {label,-40} {score:P1}");
    }
}

// Helper: Load a WAV file as 16kHz mono float samples
static float[] LoadAudioFile(string path)
{
    // Use your preferred audio library to load and resample to 16kHz mono
    // e.g., NAudio, CSCore, or System.Speech
    throw new NotImplementedException("Replace with your audio loading logic");
}
```

**Example output:**

```
🎵 rainstorm.wav:
   Rain                                     87.3%
   Rain on surface                           72.1%
   Thunderstorm                              45.6%
   Water                                     38.2%
   Thunder                                   22.4%

🎵 birdsong.wav:
   Bird vocalization, bird call, bird song   92.5%
   Bird                                      89.1%
   Chirp, tweet                              76.3%
   Outside, rural or natural                 41.0%
   Animal                                    35.8%

🎵 traffic.wav:
   Traffic noise, roadway noise              85.4%
   Motor vehicle (road)                      71.2%
   Car                                       52.8%
   Vehicle horn, car horn, honking           23.1%
   Outside, urban or manmade                 19.7%

🎵 piano.wav:
   Piano                                     94.2%
   Musical instrument                        88.7%
   Keyboard (musical)                        79.5%
   Music                                     72.4%
   Classical music                           31.6%
```

## References

- **AST Paper:** Gong, Y., Chung, Y.-A., & Glass, J. (2021). *AST: Audio Spectrogram Transformer.* Proceedings of Interspeech 2021. [arXiv:2104.01778](https://arxiv.org/abs/2104.01778)
- **AudioSet:** Gemmeke, J. F., et al. (2017). *Audio Set: An ontology and human-labeled dataset for audio events.* IEEE ICASSP 2017. [AudioSet](https://research.google.com/audioset/)
- **HuggingFace Model Card:** [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
- **ONNX Export:** [onnx-community/ast-finetuned-audioset-10-10-0.4593-ONNX](https://huggingface.co/onnx-community/ast-finetuned-audioset-10-10-0.4593-ONNX)
- **Vision Transformer (ViT):** Dosovitskiy, A., et al. (2020). *An Image is Worth 16x16 Words.* [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

## License

This model package is licensed under **MIT**. The underlying AST model weights are released under **BSD-3-Clause** by MIT.
