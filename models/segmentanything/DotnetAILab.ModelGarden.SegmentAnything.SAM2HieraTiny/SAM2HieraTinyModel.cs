using MLNet.Image.Core;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.SegmentAnything;
using ModelPackages;

namespace DotnetAILab.ModelGarden.SegmentAnything.SAM2HieraTiny;

/// <summary>
/// SAM2 Hiera-Tiny segment anything model package.
/// Segments any object in an image given point or box prompts.
/// Supports cached image embeddings for multi-prompt segmentation.
/// Downloads encoder+decoder on first use, caches locally.
/// </summary>
public static class SAM2HieraTinyModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(SAM2HieraTinyModel).Assembly));

    /// <summary>Returns the model files, downloading if needed.</summary>
    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    /// <summary>
    /// Creates a SAM2 transformer backed by the local ONNX models.
    /// Downloads the models on first call, cached thereafter.
    /// </summary>
    public static async Task<OnnxSegmentAnythingTransformer> CreateTransformerAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct).ConfigureAwait(false);

        var samOptions = new OnnxSegmentAnythingOptions
        {
            EncoderModelPath = files.GetPath("sam2_hiera_tiny_encoder.onnx"),
            DecoderModelPath = files.GetPath("sam2_hiera_tiny_decoder.onnx"),
            PreprocessorConfig = PreprocessorConfig.SAM2
        };

        return new OnnxSegmentAnythingTransformer(samOptions);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    public static Task VerifyModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.VerifyModelAsync(options, ct);
}
