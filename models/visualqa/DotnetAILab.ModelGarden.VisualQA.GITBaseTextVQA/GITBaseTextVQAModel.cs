using Microsoft.Extensions.AI;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.ImageCaptioning;
using MLNet.ImageInference.Onnx.MEAI;
using ModelPackages;

namespace DotnetAILab.ModelGarden.VisualQA.GITBaseTextVQA;

/// <summary>
/// GIT-Base (TextVQA) visual question answering model package.
/// Answers questions about image content.
/// Uses the same captioning architecture as ImageCaptioning but fine-tuned for VQA.
/// Also supports IChatClient via MEAI.
/// Downloads encoder+decoder+vocab on first use, caches locally.
/// </summary>
public static class GITBaseTextVQAModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(GITBaseTextVQAModel).Assembly));

    /// <summary>Returns the model files, downloading if needed.</summary>
    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    /// <summary>
    /// Creates a VQA transformer backed by the local ONNX models.
    /// Downloads the models on first call, cached thereafter.
    /// </summary>
    public static async Task<OnnxImageCaptioningTransformer> CreateTransformerAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct).ConfigureAwait(false);

        var vqaOptions = new OnnxImageCaptioningOptions
        {
            EncoderModelPath = files.GetPath("encoder.onnx"),
            DecoderModelPath = files.GetPath("decoder.onnx"),
            VocabPath = files.GetPath("vocab.txt"),
            PreprocessorConfig = PreprocessorConfig.GITVQA,
            MaxLength = 30
        };

        return new OnnxImageCaptioningTransformer(vqaOptions);
    }

    /// <summary>
    /// Creates an IChatClient for conversational visual question answering.
    /// Downloads the models on first call, cached thereafter.
    /// </summary>
    public static async Task<IChatClient> CreateChatClientAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct).ConfigureAwait(false);

        var vqaOptions = new OnnxImageCaptioningOptions
        {
            EncoderModelPath = files.GetPath("encoder.onnx"),
            DecoderModelPath = files.GetPath("decoder.onnx"),
            VocabPath = files.GetPath("vocab.txt"),
            PreprocessorConfig = PreprocessorConfig.GITVQA,
            MaxLength = 30
        };

        return new OnnxImageCaptioningChatClient(vqaOptions);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    public static Task VerifyModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.VerifyModelAsync(options, ct);
}
