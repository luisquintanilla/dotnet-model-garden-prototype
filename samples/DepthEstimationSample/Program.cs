using DotnetAILab.ModelGarden.DepthEstimation.DPTHybrid;
using Microsoft.ML.Data;
using System.Diagnostics;

Console.WriteLine("=== Model Garden: Depth Estimation Sample ===\n");

Console.WriteLine("Creating depth estimator (model downloads on first use)...");
var sw = Stopwatch.StartNew();
var estimator = await DPTHybridModel.CreateEstimatorAsync();
Console.WriteLine($"Estimator ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("Estimating depth...");
using var image = MLImage.CreateFromFile("test-image.jpg");
var depthMap = estimator.Estimate(image);

Console.WriteLine($"Depth map: {depthMap.Width}x{depthMap.Height}");
Console.WriteLine($"Raw depth range: [{depthMap.MinDepth:F2}, {depthMap.MaxDepth:F2}]");

var values = depthMap.Values;
Console.WriteLine($"\nNormalized depth stats:");
Console.WriteLine($"  Mean: {values.Average():F4}");
Console.WriteLine($"  Min:  {values.Min():F4}");
Console.WriteLine($"  Max:  {values.Max():F4}");

Console.WriteLine("\nModel info:");
var info = await DPTHybridModel.GetModelInfoAsync();
Console.WriteLine($"  Model ID: {info.ModelId}");
Console.WriteLine($"  Source:   {info.ResolvedSource}");

estimator.Dispose();
Console.WriteLine("\nDone!");
