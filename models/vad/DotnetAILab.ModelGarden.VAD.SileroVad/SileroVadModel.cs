using Microsoft.ML;
using MLNet.AudioInference.Onnx;
using ModelPackages;

namespace DotnetAILab.ModelGarden.VAD.SileroVad;

/// <summary>
/// Silero VAD v4 voice activity detection model package (~2MB).
/// Downloads ONNX model on first use, caches locally.
/// NuGet package version tracks the upstream Silero VAD model version.
/// </summary>
public static class SileroVadModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(SileroVadModel).Assembly));

    /// <summary>Returns the model files, downloading if needed.</summary>
    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    /// <summary>
    /// Creates an OnnxVadTransformer for voice activity detection.
    /// The returned object also implements IVoiceActivityDetector.
    /// Downloads the model on first call, cached thereafter.
    /// </summary>
    public static async Task<OnnxVadTransformer> CreateVadAsync(
        float threshold = 0.5f,
        ModelOptions? options = null,
        CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct).ConfigureAwait(false);
        var mlContext = new MLContext();

        var vadOptions = new OnnxVadOptions
        {
            ModelPath = files.PrimaryModelPath,
            Threshold = threshold,
            MinSpeechDuration = TimeSpan.FromMilliseconds(250),
            MinSilenceDuration = TimeSpan.FromMilliseconds(100),
            SpeechPad = TimeSpan.FromMilliseconds(30),
            WindowSize = 512,
            SampleRate = 16000
        };

        var estimator = mlContext.Transforms.OnnxVad(vadOptions);
        var emptyData = mlContext.Data.LoadFromEnumerable(Array.Empty<AudioInput>());
        return (OnnxVadTransformer)estimator.Fit(emptyData);
    }

    /// <summary>
    /// Convenience method that returns the VAD as IVoiceActivityDetector.
    /// OnnxVadTransformer implements IVoiceActivityDetector directly.
    /// </summary>
    public static async Task<IVoiceActivityDetector> CreateVoiceActivityDetectorAsync(
        float threshold = 0.5f,
        ModelOptions? options = null,
        CancellationToken ct = default)
    {
        return await CreateVadAsync(threshold, options, ct).ConfigureAwait(false);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    public static Task VerifyModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.VerifyModelAsync(options, ct);

    private sealed class AudioInput { public float[] Audio { get; set; } = []; }
}
