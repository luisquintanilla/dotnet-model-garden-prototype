using Microsoft.ML;
using MLNet.TextInference.Onnx;
using ModelPackages;

namespace DotnetAILab.ModelGarden.Classification.EmotionRoBERTa;

/// <summary>
/// Model package for RoBERTa GoEmotions emotion classification.
/// Downloads ONNX model on first use, caches locally.
/// </summary>
public static class EmotionRoBERTaModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(EmotionRoBERTaModel).Assembly));

    /// <summary>Classification labels for this model.</summary>
    public static readonly string[] Labels =
    [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
        "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
        "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
        "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
    ];

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
        var tokenizerDir = ModelPackage.ExtractResources(typeof(EmotionRoBERTaModel).Assembly, "EmotionRoBERTa");

        var mlContext = new MLContext();
        var estimator = mlContext.Transforms.OnnxTextClassification(new OnnxTextClassificationOptions
        {
            ModelPath = modelPath,
            TokenizerPath = tokenizerDir,
            Labels = Labels,
            MaxTokenLength = 128,
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
