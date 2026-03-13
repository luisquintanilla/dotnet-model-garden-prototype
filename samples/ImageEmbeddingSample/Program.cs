using DotnetAILab.ModelGarden.ImageEmbedding.CLIPViT;
using Microsoft.Extensions.AI;
using Microsoft.ML.Data;
using System.Diagnostics;
using System.Numerics.Tensors;

Console.WriteLine("=== Model Garden: Image Embedding Sample ===\n");

Console.WriteLine("Creating embedding generator (model downloads on first use)...");
var sw = Stopwatch.StartNew();
var generator = await CLIPViTModel.CreateEmbeddingGeneratorAsync();
Console.WriteLine($"Generator ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("Generating embeddings for images...");
using var image1 = MLImage.CreateFromFile("cat.jpg");
using var image2 = MLImage.CreateFromFile("dog.jpg");
using var image3 = MLImage.CreateFromFile("cat2.jpg");

var images = new[] { image1, image2, image3 };
var embeddings = await generator.GenerateAsync(images);

Console.WriteLine($"Generated {embeddings.Count} embeddings of dim {embeddings[0].Vector.Length}\n");

Console.WriteLine("Cosine similarity (similar images should score higher):");
var e1 = embeddings[0].Vector.ToArray();
var e2 = embeddings[1].Vector.ToArray();
var e3 = embeddings[2].Vector.ToArray();

Console.WriteLine($"  cat vs dog:   {TensorPrimitives.CosineSimilarity(e1.AsSpan(), e2.AsSpan()):F4}");
Console.WriteLine($"  cat vs cat2:  {TensorPrimitives.CosineSimilarity(e1.AsSpan(), e3.AsSpan()):F4}");
Console.WriteLine($"  dog vs cat2:  {TensorPrimitives.CosineSimilarity(e2.AsSpan(), e3.AsSpan()):F4}");

Console.WriteLine("\nModel info:");
var info = await CLIPViTModel.GetModelInfoAsync();
Console.WriteLine($"  Model ID: {info.ModelId}");
Console.WriteLine($"  Source:   {info.ResolvedSource}");

Console.WriteLine("\nDone!");
