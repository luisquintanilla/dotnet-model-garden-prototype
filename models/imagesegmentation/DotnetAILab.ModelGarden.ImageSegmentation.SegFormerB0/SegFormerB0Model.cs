using MLNet.Image.Core;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.Segmentation;
using ModelPackages;

namespace DotnetAILab.ModelGarden.ImageSegmentation.SegFormerB0;

/// <summary>
/// SegFormer-B0 image segmentation model package.
/// Segments images into semantic regions (ADE20K 150 classes).
/// Downloads ONNX model on first use, caches locally.
/// </summary>
public static class SegFormerB0Model
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(SegFormerB0Model).Assembly));

    /// <summary>Returns the model files, downloading if needed.</summary>
    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    /// <summary>
    /// Creates an image segmenter backed by the local ONNX model.
    /// Downloads the model on first call, cached thereafter.
    /// </summary>
    public static async Task<OnnxImageSegmentationTransformer> CreateSegmenterAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct).ConfigureAwait(false);

        var segmentationOptions = new OnnxImageSegmentationOptions
        {
            ModelPath = files.PrimaryModelPath,
            PreprocessorConfig = PreprocessorConfig.SegFormer,
            ResizeToOriginal = true
        };

        var estimator = new OnnxImageSegmentationEstimator(segmentationOptions);
        return estimator.Fit(null!);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    public static Task VerifyModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.VerifyModelAsync(options, ct);
}
