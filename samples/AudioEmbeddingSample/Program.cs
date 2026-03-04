using DotnetAILab.ModelGarden.AudioEmbedding.CLAP;
using Microsoft.Extensions.AI;
using MLNet.Audio.Core;
using System.Diagnostics;
using System.Numerics.Tensors;

Console.WriteLine("=== Model Garden: Audio Embedding Sample ===\n");

// Generate 3 synthetic audio signals with different characteristics
// In a real app, you'd load: var audio = AudioIO.LoadWav("sound.wav");
Console.WriteLine("Generating synthetic audio signals...");
var sampleRate = 16000;
var duration = 1.0f;
var numSamples = (int)(sampleRate * duration);

// Audio 1: 440Hz sine wave (A4 note)
var samples1 = new float[numSamples];
for (int i = 0; i < numSamples; i++)
    samples1[i] = MathF.Sin(2 * MathF.PI * 440 * i / sampleRate) * 0.5f;
var audio1 = new AudioData(samples1, sampleRate);

// Audio 2: 880Hz sine wave (A5 note — similar harmonic)
var samples2 = new float[numSamples];
for (int i = 0; i < numSamples; i++)
    samples2[i] = MathF.Sin(2 * MathF.PI * 880 * i / sampleRate) * 0.5f;
var audio2 = new AudioData(samples2, sampleRate);

// Audio 3: White noise (very different)
var rng = new Random(42);
var samples3 = new float[numSamples];
for (int i = 0; i < numSamples; i++)
    samples3[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
var audio3 = new AudioData(samples3, sampleRate);

Console.WriteLine("  Audio 1: 440Hz sine wave (A4)");
Console.WriteLine("  Audio 2: 880Hz sine wave (A5)");
Console.WriteLine("  Audio 3: White noise\n");

// Create embedding generator (downloads ~120MB on first run)
Console.WriteLine("Creating audio embedding generator (downloads on first use)...");
var sw = Stopwatch.StartNew();
IEmbeddingGenerator<AudioData, Embedding<float>> generator =
    await CLAPModel.CreateEmbeddingGeneratorAsync();
Console.WriteLine($"Generator ready in {sw.ElapsedMilliseconds}ms\n");

// Generate embeddings
Console.WriteLine("Generating embeddings...");
sw.Restart();
var embeddings = await generator.GenerateAsync([audio1, audio2, audio3]);
Console.WriteLine($"Generated {embeddings.Count} embeddings, dimension: {embeddings[0].Vector.Length}");
Console.WriteLine($"Embedding generation took {sw.ElapsedMilliseconds}ms\n");

// Compute cosine similarity
Console.WriteLine("Cosine Similarity (similar sounds should score higher):");
var e1 = embeddings[0].Vector.Span;
var e2 = embeddings[1].Vector.Span;
var e3 = embeddings[2].Vector.Span;

Console.WriteLine($"  440Hz vs 880Hz (similar tones): {TensorPrimitives.CosineSimilarity(e1, e2):F4}");
Console.WriteLine($"  440Hz vs noise (different):     {TensorPrimitives.CosineSimilarity(e1, e3):F4}");
Console.WriteLine($"  880Hz vs noise (different):     {TensorPrimitives.CosineSimilarity(e2, e3):F4}");

// Model info
Console.WriteLine("\nModel info:");
var info = await CLAPModel.GetModelInfoAsync();
Console.WriteLine($"  Model ID: {info.ModelId}");
Console.WriteLine($"  Source:   {info.ResolvedSource}");

Console.WriteLine("\nDone!");
