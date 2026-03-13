using MLNet.ImageGeneration.OnnxGenAI;
using ModelPackages;

namespace DotnetAILab.ModelGarden.TextToImage.StableDiffusionV14;

/// <summary>
/// Stable Diffusion v1.4 text-to-image generation model package.
/// Generates images from text prompts using ONNX GenAI runtime.
/// Downloads all model files (~4 GB) on first use, caches locally.
/// </summary>
public static class StableDiffusionV14Model
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(StableDiffusionV14Model).Assembly));

    /// <summary>Returns the model files, downloading if needed.</summary>
    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    /// <summary>
    /// Creates an image generator backed by the local ONNX models.
    /// Downloads the models on first call (~4 GB), cached thereafter.
    /// </summary>
    public static async Task<OnnxImageGenerationTransformer> CreateGeneratorAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct).ConfigureAwait(false);

        var textEncoderDir = Path.GetDirectoryName(files.GetPath("text_encoder/model.onnx"))!;
        var modelDirectory = Path.GetDirectoryName(textEncoderDir)!;

        var genOptions = new OnnxImageGenerationOptions
        {
            ModelDirectory = modelDirectory,
            VocabPath = files.GetPath("tokenizer/vocab.json"),
            MergesPath = files.GetPath("tokenizer/merges.txt"),
            NumInferenceSteps = 20,
            GuidanceScale = 7.5f,
            Width = 512,
            Height = 512
        };

        return new OnnxImageGenerationTransformer(genOptions);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    public static Task VerifyModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.VerifyModelAsync(options, ct);
}
