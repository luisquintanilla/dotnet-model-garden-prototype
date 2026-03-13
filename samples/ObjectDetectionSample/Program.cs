using DotnetAILab.ModelGarden.ObjectDetection.YoloV8s;
using Microsoft.ML.Data;
using System.Diagnostics;

Console.WriteLine("=== Model Garden: Object Detection Sample ===\n");

Console.WriteLine("Creating object detector (model downloads on first use)...");
var sw = Stopwatch.StartNew();
var detector = await YoloV8sModel.CreateDetectorAsync();
Console.WriteLine($"Detector ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("Detecting objects...");
using var image = MLImage.CreateFromFile("test-image.jpg");
var detections = detector.Detect(image);

Console.WriteLine($"Found {detections.Length} objects:");
foreach (var box in detections)
    Console.WriteLine($"  → {box}");

Console.WriteLine("\nModel info:");
var info = await YoloV8sModel.GetModelInfoAsync();
Console.WriteLine($"  Model ID: {info.ModelId}");
Console.WriteLine($"  Source:   {info.ResolvedSource}");

detector.Dispose();
Console.WriteLine("\nDone!");
