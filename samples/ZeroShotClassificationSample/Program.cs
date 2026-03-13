using DotnetAILab.ModelGarden.ZeroShotClassification.CLIPViT;
using Microsoft.ML.Data;
using System.Diagnostics;

Console.WriteLine("=== Model Garden: Zero-Shot Image Classification Sample ===\n");

var candidateLabels = new[]
{
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a bird",
    "a photo of a car",
    "a photo of a person"
};

Console.WriteLine("Creating zero-shot classifier (model downloads on first use)...");
var sw = Stopwatch.StartNew();
var classifier = await ZeroShotCLIPViTModel.CreateClassifierAsync(candidateLabels);
Console.WriteLine($"Classifier ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("Classifying image against custom labels...");
using var image = MLImage.CreateFromFile("test-image.jpg");
var results = classifier.Classify(image);

Console.WriteLine("Predictions:");
foreach (var (label, probability) in results)
    Console.WriteLine($"  → {label}: {probability:P1}");

Console.WriteLine("\nModel info:");
var info = await ZeroShotCLIPViTModel.GetModelInfoAsync();
Console.WriteLine($"  Model ID: {info.ModelId}");
Console.WriteLine($"  Source:   {info.ResolvedSource}");

classifier.Dispose();
Console.WriteLine("\nDone!");
