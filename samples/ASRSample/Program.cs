using DotnetAILab.ModelGarden.ASR.WhisperBase;
using MLNet.Audio.Core;
using System.Diagnostics;

Console.WriteLine("=== Model Garden: ASR (Speech-to-Text) Sample ===\n");

// Generate a simple synthetic audio signal (1 second of 440Hz sine wave at 16kHz)
// In a real app, you'd load a .wav file: var audio = AudioIO.LoadWav("speech.wav");
Console.WriteLine("Generating synthetic audio (440Hz sine wave, 1 second)...");
var sampleRate = 16000;
var duration = 1.0f;
var samples = new float[(int)(sampleRate * duration)];
for (int i = 0; i < samples.Length; i++)
    samples[i] = MathF.Sin(2 * MathF.PI * 440 * i / sampleRate) * 0.5f;
var audio = new AudioData(samples, sampleRate);
Console.WriteLine($"Audio: {audio.Duration.TotalSeconds:F1}s, {audio.SampleRate}Hz, {audio.Samples.Length} samples\n");

// Create speech-to-text model (downloads ~290MB on first run)
Console.WriteLine("Creating speech-to-text model (downloads on first use)...");
var sw = Stopwatch.StartNew();
var stt = await WhisperBaseModel.CreateSpeechToTextAsync(language: "en");
Console.WriteLine($"Model ready in {sw.ElapsedMilliseconds}ms\n");

// Transcribe
Console.WriteLine("Transcribing...");
sw.Restart();
var results = stt.Transcribe([audio]);
Console.WriteLine($"Transcription completed in {sw.ElapsedMilliseconds}ms\n");

foreach (var text in results)
    Console.WriteLine($"  Result: \"{text}\"");

// Model info
Console.WriteLine("\nModel info:");
var info = await WhisperBaseModel.GetModelInfoAsync();
Console.WriteLine($"  Model ID: {info.ModelId}");
Console.WriteLine($"  Source:   {info.ResolvedSource}");

stt.Dispose();
Console.WriteLine("\nDone!");
