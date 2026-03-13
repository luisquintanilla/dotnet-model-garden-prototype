using MLNet.Image.Core;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.Classification;
using ModelPackages;

namespace DotnetAILab.ModelGarden.ImageClassification.ViTBase;

/// <summary>
/// ViT-Base image classification model package.
/// Classifies images into 1000 ImageNet categories (top-5).
/// Downloads ONNX model on first use, caches locally.
/// </summary>
public static class ViTBaseModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(ViTBaseModel).Assembly));

    /// <summary>Returns the model files, downloading if needed.</summary>
    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    /// <summary>
    /// Creates an image classifier backed by the local ONNX model.
    /// Downloads the model on first call, cached thereafter.
    /// </summary>
    public static async Task<OnnxImageClassificationTransformer> CreateClassifierAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct).ConfigureAwait(false);

        var classificationOptions = new OnnxImageClassificationOptions
        {
            ModelPath = files.PrimaryModelPath,
            PreprocessorConfig = PreprocessorConfig.ImageNet,
            TopK = 5
        };

        var estimator = new OnnxImageClassificationEstimator(classificationOptions);
        return estimator.Fit(null!);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    public static Task VerifyModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.VerifyModelAsync(options, ct);
}
