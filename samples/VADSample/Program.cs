using DotnetAILab.ModelGarden.VAD.SileroVad;
using MLNet.Audio.Core;
using System.Diagnostics;

Console.WriteLine("=== Model Garden: VAD (Voice Activity Detection) Sample ===\n");

// Generate synthetic audio: 0.5s silence + 1s tone (simulated speech) + 0.5s silence + 0.5s tone + 0.5s silence
// In a real app, you'd load: var audio = AudioIO.LoadWav("recording.wav");
Console.WriteLine("Generating synthetic audio (tone bursts with silence gaps)...");
var sampleRate = 16000;
var totalSeconds = 3.0f;
var samples = new float[(int)(sampleRate * totalSeconds)];

// Fill with silence, then add tone bursts to simulate speech
// Burst 1: 0.5s-1.5s (440Hz)
for (int i = (int)(0.5f * sampleRate); i < (int)(1.5f * sampleRate); i++)
    samples[i] = MathF.Sin(2 * MathF.PI * 440 * i / sampleRate) * 0.8f;
// Burst 2: 2.0s-2.5s (880Hz)
for (int i = (int)(2.0f * sampleRate); i < (int)(2.5f * sampleRate); i++)
    samples[i] = MathF.Sin(2 * MathF.PI * 880 * i / sampleRate) * 0.8f;

var audio = new AudioData(samples, sampleRate);
Console.WriteLine($"Audio: {audio.Duration.TotalSeconds:F1}s, {audio.SampleRate}Hz\n");

// Create VAD (downloads ~2MB on first run)
Console.WriteLine("Creating voice activity detector (downloads on first use)...");
var sw = Stopwatch.StartNew();
var vad = await SileroVadModel.CreateVadAsync(threshold: 0.5f);
Console.WriteLine($"VAD ready in {sw.ElapsedMilliseconds}ms\n");

// Detect speech segments
Console.WriteLine("Detecting speech segments...");
sw.Restart();
var segments = vad.DetectSpeech(audio);
Console.WriteLine($"Detection completed in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine($"Found {segments.Length} speech segment(s):");
foreach (var seg in segments)
    Console.WriteLine($"  [{seg.Start:mm\\:ss\\.fff} → {seg.End:mm\\:ss\\.fff}] confidence: {seg.Confidence:F3}");

// Model info
Console.WriteLine("\nModel info:");
var info = await SileroVadModel.GetModelInfoAsync();
Console.WriteLine($"  Model ID: {info.ModelId}");
Console.WriteLine($"  Source:   {info.ResolvedSource}");

vad.Dispose();
Console.WriteLine("\nDone!");
