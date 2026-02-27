using Microsoft.ML;
using MLNet.TextGeneration.OnnxGenAI;
using ModelPackages;

namespace DotnetAILab.ModelGarden.TextGeneration.Phi3Mini;

/// <summary>
/// Model package for Phi-3 Mini text generation.
/// Downloads ONNX model directory on first use, caches locally.
/// Uses ONNX Runtime GenAI for local autoregressive text generation.
/// </summary>
public static class Phi3MiniModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(Phi3MiniModel).Assembly));

    /// <summary>Returns local path to the cached model directory.</summary>
    public static Task<string> EnsureModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureModelAsync(options, ct);

    /// <summary>
    /// Creates a text generation transformer backed by the local ONNX model.
    /// Downloads the model directory on first call, cached thereafter.
    /// </summary>
    public static async Task<OnnxTextGenerationTransformer> CreateTextGeneratorAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var modelDir = await EnsureModelAsync(options, ct);

        var mlContext = new MLContext();
        var estimator = mlContext.Transforms.OnnxTextGeneration(new OnnxTextGenerationOptions
        {
            ModelPath = modelDir,
            MaxLength = 256,
            Temperature = 0.7f,
            TopP = 0.9f
        });

        var dummyData = mlContext.Data.LoadFromEnumerable(new[] { new TextData { Text = "" } });
        return estimator.Fit(dummyData);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    public static Task VerifyModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.VerifyModelAsync(options, ct);

    private sealed class TextData
    {
        public string Text { get; set; } = "";
    }
}
