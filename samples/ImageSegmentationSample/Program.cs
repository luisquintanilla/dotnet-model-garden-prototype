using DotnetAILab.ModelGarden.ImageSegmentation.SegFormerB0;
using Microsoft.ML.Data;
using System.Diagnostics;

Console.WriteLine("=== Model Garden: Image Segmentation Sample ===\n");

Console.WriteLine("Creating image segmenter (model downloads on first use)...");
var sw = Stopwatch.StartNew();
var segmenter = await SegFormerB0Model.CreateSegmenterAsync();
Console.WriteLine($"Segmenter ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("Segmenting image...");
using var image = MLImage.CreateFromFile("test-image.jpg");
var mask = segmenter.Segment(image);

Console.WriteLine($"Mask size: {mask.Width}x{mask.Height}");

var uniqueClasses = mask.ClassIds.Distinct().OrderBy(id => id).ToArray();
Console.WriteLine($"Unique classes found: {uniqueClasses.Length}\n");

foreach (var classId in uniqueClasses)
{
    var label = mask.Labels is not null && classId < mask.Labels.Length
        ? mask.Labels[classId]
        : $"class_{classId}";
    var pixelCount = mask.ClassIds.Count(id => id == classId);
    var percentage = (float)pixelCount / mask.ClassIds.Length * 100;
    Console.WriteLine($"  [{classId}] {label}: {pixelCount} pixels ({percentage:F1}%)");
}

Console.WriteLine("\nModel info:");
var info = await SegFormerB0Model.GetModelInfoAsync();
Console.WriteLine($"  Model ID: {info.ModelId}");
Console.WriteLine($"  Source:   {info.ResolvedSource}");

segmenter.Dispose();
Console.WriteLine("\nDone!");
