using Microsoft.Extensions.AI;
using Microsoft.ML;
using MLNet.TextInference.Onnx;
using ModelPackages;

namespace DotnetAILab.ModelGarden.Embeddings.BgeSmallEn;

/// <summary>
/// Model package for bge-small-en-v1.5.
/// Downloads ONNX model on first use, caches locally.
/// </summary>
public static class BgeSmallEnModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(BgeSmallEnModel).Assembly));

    /// <summary>Returns local path to the cached ONNX model file.</summary>
    public static Task<string> EnsureModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureModelAsync(options, ct);

    /// <summary>
    /// Creates an IEmbeddingGenerator backed by the local ONNX model.
    /// Downloads the model on first call, cached thereafter.
    /// </summary>
    public static async Task<IEmbeddingGenerator<string, Embedding<float>>> CreateEmbeddingGeneratorAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var modelPath = await EnsureModelAsync(options, ct);
        var tokenizerDir = ModelPackage.ExtractResources(typeof(BgeSmallEnModel).Assembly, "BgeSmallEn");

        var mlContext = new MLContext();
        var estimator = new OnnxTextEmbeddingEstimator(mlContext, new OnnxTextEmbeddingOptions
        {
            ModelPath = modelPath,
            TokenizerPath = tokenizerDir,
            Pooling = PoolingStrategy.ClsToken,
            Normalize = true,
            BatchSize = 32
        });

        var dummyData = mlContext.Data.LoadFromEnumerable(new[] { new TextData { Text = "" } });
        var transformer = estimator.Fit(dummyData);

        return new OnnxEmbeddingGenerator(mlContext, transformer, ownsTransformer: true);
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
