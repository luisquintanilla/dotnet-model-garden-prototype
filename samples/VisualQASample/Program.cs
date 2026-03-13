using DotnetAILab.ModelGarden.VisualQA.GITBaseTextVQA;
using Microsoft.ML.Data;
using System.Diagnostics;

Console.WriteLine("=== Model Garden: Visual Question Answering Sample ===\n");

Console.WriteLine("Creating VQA transformer (model downloads on first use)...");
var sw = Stopwatch.StartNew();
var transformer = await GITBaseTextVQAModel.CreateTransformerAsync();
Console.WriteLine($"Transformer ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("Asking questions about image...");
using var image = MLImage.CreateFromFile("test-image.jpg");

var questions = new[]
{
    "What is shown in this image?",
    "How many objects are there?",
    "What color is the main object?"
};

foreach (var question in questions)
{
    var answer = transformer.AnswerQuestion(image, question);
    Console.WriteLine($"  Q: {question}");
    Console.WriteLine($"  A: {answer}\n");
}

Console.WriteLine("Generating caption...");
var caption = transformer.GenerateCaption(image);
Console.WriteLine($"  Caption: {caption}");

Console.WriteLine("\nModel info:");
var info = await GITBaseTextVQAModel.GetModelInfoAsync();
Console.WriteLine($"  Model ID: {info.ModelId}");
Console.WriteLine($"  Source:   {info.ResolvedSource}");

transformer.Dispose();
Console.WriteLine("\nDone!");
