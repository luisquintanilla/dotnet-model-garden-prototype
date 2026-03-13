using Microsoft.Extensions.AI;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.ImageCaptioning;
using MLNet.ImageInference.Onnx.MEAI;
using ModelPackages;

namespace DotnetAILab.ModelGarden.ImageCaptioning.GITBaseCoco;

/// <summary>
/// GIT-Base (COCO) image captioning model package.
/// Generates natural language captions for images.
/// Also supports IChatClient via MEAI for conversational image understanding.
/// Downloads encoder+decoder+vocab on first use, caches locally.
/// </summary>
public static class GITBaseCocoModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(GITBaseCocoModel).Assembly));

    /// <summary>Returns the model files, downloading if needed.</summary>
    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    /// <summary>
    /// Creates an image captioner backed by the local ONNX models.
    /// Downloads the models on first call, cached thereafter.
    /// </summary>
    public static async Task<OnnxImageCaptioningTransformer> CreateCaptionerAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct).ConfigureAwait(false);

        var captioningOptions = new OnnxImageCaptioningOptions
        {
            EncoderModelPath = files.GetPath("encoder.onnx"),
            DecoderModelPath = files.GetPath("decoder.onnx"),
            VocabPath = files.GetPath("vocab.txt"),
            PreprocessorConfig = PreprocessorConfig.GIT,
            MaxLength = 50
        };

        var estimator = new OnnxImageCaptioningEstimator(captioningOptions);
        return estimator.Fit(null!);
    }

    /// <summary>
    /// Creates an IChatClient for conversational image understanding.
    /// Downloads the models on first call, cached thereafter.
    /// </summary>
    public static async Task<IChatClient> CreateChatClientAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct).ConfigureAwait(false);

        var captioningOptions = new OnnxImageCaptioningOptions
        {
            EncoderModelPath = files.GetPath("encoder.onnx"),
            DecoderModelPath = files.GetPath("decoder.onnx"),
            VocabPath = files.GetPath("vocab.txt"),
            PreprocessorConfig = PreprocessorConfig.GIT,
            MaxLength = 50
        };

        return new OnnxImageCaptioningChatClient(captioningOptions);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    public static Task VerifyModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.VerifyModelAsync(options, ct);
}
