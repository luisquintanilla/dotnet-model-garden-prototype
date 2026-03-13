using DotnetAILab.ModelGarden.ImageCaptioning.GITBaseCoco;
using Microsoft.ML.Data;
using System.Diagnostics;

Console.WriteLine("=== Model Garden: Image Captioning Sample ===\n");

Console.WriteLine("Creating image captioner (model downloads on first use)...");
var sw = Stopwatch.StartNew();
var captioner = await GITBaseCocoModel.CreateCaptionerAsync();
Console.WriteLine($"Captioner ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("Generating caption...");
using var image = MLImage.CreateFromFile("test-image.jpg");
var caption = captioner.GenerateCaption(image);
Console.WriteLine($"  Caption: {caption}");

Console.WriteLine("\nVisual question answering...");
var answer = captioner.AnswerQuestion(image, "What color is the sky?");
Console.WriteLine($"  Q: What color is the sky?");
Console.WriteLine($"  A: {answer}");

Console.WriteLine("\nModel info:");
var info = await GITBaseCocoModel.GetModelInfoAsync();
Console.WriteLine($"  Model ID: {info.ModelId}");
Console.WriteLine($"  Source:   {info.ResolvedSource}");

captioner.Dispose();
Console.WriteLine("\nDone!");
