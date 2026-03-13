using MLNet.Image.Core;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.Detection;
using ModelPackages;

namespace DotnetAILab.ModelGarden.ObjectDetection.YoloV8s;

/// <summary>
/// YOLOv8s object detection model package.
/// Detects objects in images with bounding boxes and labels.
/// Downloads ONNX model on first use, caches locally.
/// </summary>
public static class YoloV8sModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(YoloV8sModel).Assembly));

    /// <summary>Returns the model files, downloading if needed.</summary>
    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    /// <summary>
    /// Creates an object detector backed by the local ONNX model.
    /// Downloads the model on first call, cached thereafter.
    /// </summary>
    public static async Task<OnnxObjectDetectionTransformer> CreateDetectorAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct).ConfigureAwait(false);

        var detectionOptions = new OnnxObjectDetectionOptions
        {
            ModelPath = files.PrimaryModelPath,
            PreprocessorConfig = PreprocessorConfig.YOLOv8,
            ConfidenceThreshold = 0.5f,
            IouThreshold = 0.45f
        };

        var estimator = new OnnxObjectDetectionEstimator(detectionOptions);
        return estimator.Fit(null!);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    public static Task VerifyModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.VerifyModelAsync(options, ct);
}
