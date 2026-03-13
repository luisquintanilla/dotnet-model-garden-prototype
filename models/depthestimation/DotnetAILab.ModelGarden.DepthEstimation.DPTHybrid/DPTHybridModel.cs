using MLNet.Image.Core;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.DepthEstimation;
using ModelPackages;

namespace DotnetAILab.ModelGarden.DepthEstimation.DPTHybrid;

/// <summary>
/// DPT-Hybrid (MiDaS) depth estimation model package.
/// Estimates relative depth from monocular images.
/// Downloads ONNX model on first use, caches locally.
/// </summary>
public static class DPTHybridModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(DPTHybridModel).Assembly));

    /// <summary>Returns the model files, downloading if needed.</summary>
    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    /// <summary>
    /// Creates a depth estimator backed by the local ONNX model.
    /// Downloads the model on first call, cached thereafter.
    /// </summary>
    public static async Task<OnnxImageDepthEstimationTransformer> CreateEstimatorAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct).ConfigureAwait(false);

        var depthOptions = new OnnxImageDepthEstimationOptions
        {
            ModelPath = files.PrimaryModelPath,
            PreprocessorConfig = PreprocessorConfig.DPT,
            ResizeToOriginal = true
        };

        var estimator = new OnnxImageDepthEstimationEstimator(depthOptions);
        return estimator.Fit(null!);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    public static Task VerifyModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.VerifyModelAsync(options, ct);
}
