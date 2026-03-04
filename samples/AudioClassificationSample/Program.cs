using DotnetAILab.ModelGarden.AudioClassification.ASTAudioSet;
using MLNet.Audio.Core;
using System.Diagnostics;

Console.WriteLine("=== Model Garden: Audio Classification Sample ===\n");

// Generate a synthetic audio signal (1 second of 440Hz sine wave at 16kHz)
// In a real app, you'd load: var audio = AudioIO.LoadWav("sound.wav");
Console.WriteLine("Generating synthetic audio (440Hz sine wave, 1 second)...");
var sampleRate = 16000;
var duration = 1.0f;
var samples = new float[(int)(sampleRate * duration)];
for (int i = 0; i < samples.Length; i++)
    samples[i] = MathF.Sin(2 * MathF.PI * 440 * i / sampleRate) * 0.5f;
var audio = new AudioData(samples, sampleRate);
Console.WriteLine($"Audio: {audio.Duration.TotalSeconds:F1}s, {audio.SampleRate}Hz\n");

// Create classifier (downloads ~347MB on first run)
Console.WriteLine("Creating audio classifier (downloads on first use)...");
var sw = Stopwatch.StartNew();
var classifier = await ASTAudioSetModel.CreateClassifierAsync();
Console.WriteLine($"Classifier ready in {sw.ElapsedMilliseconds}ms\n");

// Classify
Console.WriteLine("Classifying audio...");
sw.Restart();
var results = classifier.Classify([audio]);
Console.WriteLine($"Classification completed in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("Top predictions:");
foreach (var result in results)
{
    Console.WriteLine($"  → {result.PredictedLabel} (score: {result.Score:F4})");
    if (result.Probabilities != null && result.Labels != null)
    {
        // Show top 5 by probability
        var topIndices = result.Probabilities
            .Select((p, i) => (prob: p, idx: i))
            .OrderByDescending(x => x.prob)
            .Take(5);
        foreach (var (prob, idx) in topIndices)
            Console.WriteLine($"     {result.Labels[idx]}: {prob:F4}");
    }
}

// Model info
Console.WriteLine("\nModel info:");
var info = await ASTAudioSetModel.GetModelInfoAsync();
Console.WriteLine($"  Model ID: {info.ModelId}");
Console.WriteLine($"  Source:   {info.ResolvedSource}");

classifier.Dispose();
Console.WriteLine("\nDone!");
