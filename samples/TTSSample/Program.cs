using DotnetAILab.ModelGarden.TTS.SpeechT5;
using MLNet.Audio.Core;
using System.Diagnostics;

Console.WriteLine("=== Model Garden: TTS (Text-to-Speech) Sample ===\n");

// Create TTS client (downloads ~643MB on first run)
Console.WriteLine("Creating text-to-speech client (downloads on first use)...");
var sw = Stopwatch.StartNew();
var tts = await SpeechT5Model.CreateTextToSpeechClientAsync();
Console.WriteLine($"TTS ready in {sw.ElapsedMilliseconds}ms\n");

// Synthesize speech
var text = "Hello! This is a test of the SpeechT5 text to speech model from the dotnet model garden.";
Console.WriteLine($"Synthesizing: \"{text}\"");
sw.Restart();
var response = await tts.GetAudioAsync(text);
Console.WriteLine($"Synthesis completed in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine($"  Duration:    {response.Duration.TotalSeconds:F2}s");
Console.WriteLine($"  Sample rate: {response.Audio.SampleRate}Hz");
Console.WriteLine($"  Samples:     {response.Audio.Samples.Length}");

// Save to WAV
var outputPath = "tts_output.wav";
Console.WriteLine($"\nSaving to {outputPath}...");
AudioIO.SaveWav(outputPath, response.Audio);
Console.WriteLine($"Saved! File size: {new FileInfo(outputPath).Length:N0} bytes");

// Model info
Console.WriteLine("\nModel info:");
var info = await SpeechT5Model.GetModelInfoAsync();
Console.WriteLine($"  Model ID: {info.ModelId}");
Console.WriteLine($"  Source:   {info.ResolvedSource}");

tts.Dispose();
Console.WriteLine("\nDone!");
