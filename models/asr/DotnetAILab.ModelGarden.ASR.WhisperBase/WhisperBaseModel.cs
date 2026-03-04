using Microsoft.ML;
using MLNet.AudioInference.Onnx;
using ModelPackages;

namespace DotnetAILab.ModelGarden.ASR.WhisperBase;

/// <summary>
/// Whisper Base speech-to-text model package (74M params).
/// Downloads ONNX encoder+decoder on first use, caches locally.
/// </summary>
public static class WhisperBaseModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(WhisperBaseModel).Assembly));

    /// <summary>Returns the model files, downloading if needed.</summary>
    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    /// <summary>
    /// Creates an OnnxWhisperTransformer for speech-to-text transcription.
    /// Downloads the model on first call, cached thereafter.
    /// </summary>
    public static async Task<OnnxWhisperTransformer> CreateSpeechToTextAsync(
        string language = "en",
        MLContext? mlContext = null,
        ModelOptions? options = null,
        CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct).ConfigureAwait(false);
        mlContext ??= new MLContext();

        var whisperOptions = new OnnxWhisperOptions
        {
            EncoderModelPath = files.GetPath("onnx/encoder_model.onnx"),
            DecoderModelPath = files.GetPath("onnx/decoder_model_merged.onnx"),
            Language = language,
            NumMelBins = 80,
            MaxTokens = 256,
            SampleRate = 16000
        };

        var estimator = new OnnxWhisperEstimator(mlContext, whisperOptions);
        var dummyData = mlContext.Data.LoadFromEnumerable(Array.Empty<AudioInput>());
        return (OnnxWhisperTransformer)estimator.Fit(dummyData);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    public static Task VerifyModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.VerifyModelAsync(options, ct);

    private sealed class AudioInput { public float[] Audio { get; set; } = []; }
}
