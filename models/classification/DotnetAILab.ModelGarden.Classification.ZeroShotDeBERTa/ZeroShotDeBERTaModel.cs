using Microsoft.ML;
using MLNet.TextInference.Onnx;
using ModelPackages;

namespace DotnetAILab.ModelGarden.Classification.ZeroShotDeBERTa;

/// <summary>
/// Model package for DeBERTa-v3 zero-shot NLI classification.
/// Downloads ONNX model on first use, caches locally.
/// </summary>
public static class ZeroShotDeBERTaModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(ZeroShotDeBERTaModel).Assembly));

    /// <summary>Classification labels for this model.</summary>
    public static readonly string[] Labels = ["contradiction", "neutral", "entailment"];

    /// <summary>Returns local path to the cached ONNX model file.</summary>
    public static Task<string> EnsureModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureModelAsync(options, ct);

    /// <summary>
    /// Creates a text classification transformer backed by the local ONNX model.
    /// Downloads the model on first call, cached thereafter.
    /// </summary>
    public static async Task<OnnxTextClassificationTransformer> CreateClassifierAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var modelPath = await EnsureModelAsync(options, ct);
        var tokenizerDir = ModelPackage.ExtractResources(typeof(ZeroShotDeBERTaModel).Assembly, "ZeroShotDeBERTa");

        var mlContext = new MLContext();
        var estimator = mlContext.Transforms.OnnxTextClassification(new OnnxTextClassificationOptions
        {
            ModelPath = modelPath,
            TokenizerPath = tokenizerDir,
            Labels = Labels,
            MaxTokenLength = 256,
            BatchSize = 8
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
