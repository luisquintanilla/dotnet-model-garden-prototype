using Microsoft.ML;
using MLNet.AudioInference.Onnx;
using ModelPackages;

namespace DotnetAILab.ModelGarden.TTS.SpeechT5;

/// <summary>
/// SpeechT5 text-to-speech model package (~643MB total).
/// Multi-model architecture: encoder + decoder + vocoder + tokenizer + speaker embedding.
/// ITextToSpeechClient is a prototype interface (not yet in MEAI).
/// Downloads all 5 files on first use, caches locally.
/// </summary>
public static class SpeechT5Model
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(SpeechT5Model).Assembly));

    /// <summary>Returns the model files, downloading if needed.</summary>
    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    /// <summary>
    /// Creates an OnnxSpeechT5TtsTransformer for low-level TTS access.
    /// Downloads all model files on first call, cached thereafter.
    /// </summary>
    public static async Task<OnnxSpeechT5TtsTransformer> CreateTtsTransformerAsync(
        MLContext? mlContext = null,
        ModelOptions? options = null,
        CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct).ConfigureAwait(false);
        mlContext ??= new MLContext();

        var ttsOptions = new OnnxSpeechT5Options
        {
            EncoderModelPath = files.GetPath("encoder_model.onnx"),
            DecoderModelPath = files.GetPath("decoder_model_merged.onnx"),
            VocoderModelPath = files.GetPath("decoder_postnet_and_vocoder.onnx"),
            MaxMelFrames = 500,
            StopThreshold = 0.5f
        };

        return new OnnxSpeechT5TtsTransformer(mlContext, ttsOptions);
    }

    /// <summary>
    /// Creates an ITextToSpeechClient for high-level TTS access.
    /// Note: ITextToSpeechClient is a prototype interface (not yet in MEAI).
    /// Downloads all model files on first call, cached thereafter.
    /// </summary>
    public static async Task<ITextToSpeechClient> CreateTextToSpeechClientAsync(
        ModelOptions? options = null,
        CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct).ConfigureAwait(false);

        var ttsOptions = new OnnxSpeechT5Options
        {
            EncoderModelPath = files.GetPath("encoder_model.onnx"),
            DecoderModelPath = files.GetPath("decoder_model_merged.onnx"),
            VocoderModelPath = files.GetPath("decoder_postnet_and_vocoder.onnx"),
            MaxMelFrames = 500,
            StopThreshold = 0.5f
        };

        return new OnnxTextToSpeechClient(ttsOptions);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    public static Task VerifyModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.VerifyModelAsync(options, ct);
}
