using DotnetAILab.ModelGarden.ImageClassification.ViTBase;
using Microsoft.ML.Data;
using System.Diagnostics;

Console.WriteLine("=== Model Garden: Image Classification Sample ===\n");

Console.WriteLine("Creating image classifier (model downloads on first use)...");
var sw = Stopwatch.StartNew();
var classifier = await ViTBaseModel.CreateClassifierAsync();
Console.WriteLine($"Classifier ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("Classifying image...");
using var image = MLImage.CreateFromFile("test-image.jpg");
var results = classifier.Classify(image);

Console.WriteLine("Top predictions:");
foreach (var (label, probability) in results)
    Console.WriteLine($"  → {label}: {probability:P1}");

Console.WriteLine("\nModel info:");
var info = await ViTBaseModel.GetModelInfoAsync();
Console.WriteLine($"  Model ID: {info.ModelId}");
Console.WriteLine($"  Source:   {info.ResolvedSource}");

classifier.Dispose();
Console.WriteLine("\nDone!");
