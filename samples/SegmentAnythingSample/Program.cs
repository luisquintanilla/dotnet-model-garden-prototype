using DotnetAILab.ModelGarden.SegmentAnything.SAM2HieraTiny;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.SegmentAnything;
using System.Diagnostics;

Console.WriteLine("=== Model Garden: Segment Anything (SAM2) Sample ===\n");

Console.WriteLine("Creating SAM2 transformer (model downloads on first use)...");
var sw = Stopwatch.StartNew();
var transformer = await SAM2HieraTinyModel.CreateTransformerAsync();
Console.WriteLine($"Transformer ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("Segmenting with point prompt...");
using var image = MLImage.CreateFromFile("test-image.jpg");

var pointPrompt = SegmentAnythingPrompt.FromPoint(256f, 256f);
var pointResult = transformer.Segment(image, pointPrompt);

Console.WriteLine($"  Masks: {pointResult.NumMasks}");
Console.WriteLine($"  Best IoU: {pointResult.GetBestIoU():F4}");
Console.WriteLine($"  Mask pixels: {pointResult.GetBestMask().Count(v => v > 0)} foreground");

Console.WriteLine("\nSegmenting with bounding box prompt...");
var boxPrompt = SegmentAnythingPrompt.FromBoundingBox(128f, 128f, 384f, 384f);
var boxResult = transformer.Segment(image, boxPrompt);

Console.WriteLine($"  Masks: {boxResult.NumMasks}");
Console.WriteLine($"  Best IoU: {boxResult.GetBestIoU():F4}");
Console.WriteLine($"  Mask pixels: {boxResult.GetBestMask().Count(v => v > 0)} foreground");

Console.WriteLine("\nCached embedding (multiple prompts)...");
var embedding = transformer.EncodeImage(image);
for (int i = 0; i < 3; i++)
{
    float x = 128f + i * 128f;
    var prompt = SegmentAnythingPrompt.FromPoint(x, 256f);
    var result = transformer.Segment(embedding, prompt);
    Console.WriteLine($"  Point ({x:F0}, 256) → IoU={result.GetBestIoU():F4}");
}

Console.WriteLine("\nModel info:");
var info = await SAM2HieraTinyModel.GetModelInfoAsync();
Console.WriteLine($"  Model ID: {info.ModelId}");
Console.WriteLine($"  Source:   {info.ResolvedSource}");

transformer.Dispose();
Console.WriteLine("\nDone!");
