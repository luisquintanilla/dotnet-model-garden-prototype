using DotnetAILab.ModelGarden.TextToImage.StableDiffusionV14;
using System.Diagnostics;

Console.WriteLine("=== Model Garden: Text-to-Image Generation Sample ===\n");

Console.WriteLine("Creating image generator (model downloads on first use — ~4 GB)...");
var sw = Stopwatch.StartNew();
var generator = await StableDiffusionV14Model.CreateGeneratorAsync();
Console.WriteLine($"Generator ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("Generating image from prompt...");
var prompt = "a cat sitting on a beach at sunset";
Console.WriteLine($"  Prompt: \"{prompt}\"");

using var image = generator.Generate(prompt, seed: 42);
Console.WriteLine($"  Generated image: {image.Width}x{image.Height}");

Console.WriteLine("\nModel info:");
var info = await StableDiffusionV14Model.GetModelInfoAsync();
Console.WriteLine($"  Model ID: {info.ModelId}");
Console.WriteLine($"  Source:   {info.ResolvedSource}");

generator.Dispose();
Console.WriteLine("\nDone!");
