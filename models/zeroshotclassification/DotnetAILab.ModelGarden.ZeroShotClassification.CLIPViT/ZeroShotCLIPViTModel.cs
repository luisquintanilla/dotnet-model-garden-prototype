using MLNet.Image.Core;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.ZeroShot;
using ModelPackages;

namespace DotnetAILab.ModelGarden.ZeroShotClassification.CLIPViT;

/// <summary>
/// CLIP ViT-Base zero-shot image classification model package.
/// Classifies images against arbitrary text labels without task-specific training.
/// Requires separate vision/text ONNX models and tokenizer files.
/// Downloads all files on first use, caches locally.
/// </summary>
public static class ZeroShotCLIPViTModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(ZeroShotCLIPViTModel).Assembly));

    /// <summary>Returns the model files, downloading if needed.</summary>
    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    /// <summary>
    /// Creates a zero-shot image classifier for the given candidate labels.
    /// Downloads the model on first call, cached thereafter.
    /// </summary>
    public static async Task<OnnxZeroShotImageClassificationTransformer> CreateClassifierAsync(
        string[] candidateLabels,
        ModelOptions? options = null,
        CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct).ConfigureAwait(false);

        var classificationOptions = new OnnxZeroShotImageClassificationOptions
        {
            ImageModelPath = files.GetPath("onnx/vision_model.onnx"),
            TextModelPath = files.GetPath("onnx/text_model.onnx"),
            VocabPath = files.GetPath("vocab.json"),
            MergesPath = files.GetPath("merges.txt"),
            CandidateLabels = candidateLabels,
            PreprocessorConfig = PreprocessorConfig.CLIP
        };

        var estimator = new OnnxZeroShotImageClassificationEstimator(classificationOptions);
        return estimator.Fit(null!);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    public static Task VerifyModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.VerifyModelAsync(options, ct);
}
