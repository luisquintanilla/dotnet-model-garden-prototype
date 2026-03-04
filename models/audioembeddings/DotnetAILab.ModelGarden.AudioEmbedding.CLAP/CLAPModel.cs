using Microsoft.Extensions.AI;
using Microsoft.ML;
using MLNet.Audio.Core;
using MLNet.AudioInference.Onnx;
using ModelPackages;

namespace DotnetAILab.ModelGarden.AudioEmbedding.CLAP;

/// <summary>
/// CLAP audio embedding model package.
/// Generates 512-dim L2-normalized embeddings from audio.
/// Uses 64 mel bins at 16kHz.
/// </summary>
public static class CLAPModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(CLAPModel).Assembly));

    /// <summary>Returns the model files, downloading if needed.</summary>
    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    /// <summary>
    /// Creates an IEmbeddingGenerator for audio data.
    /// Downloads the model on first call, cached thereafter.
    /// </summary>
    public static async Task<IEmbeddingGenerator<AudioData, Embedding<float>>> CreateEmbeddingGeneratorAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct).ConfigureAwait(false);
        var mlContext = new MLContext();

        var embeddingOptions = new OnnxAudioEmbeddingOptions
        {
            ModelPath = files.PrimaryModelPath,
            FeatureExtractor = new MelSpectrogramExtractor(sampleRate: 16000)
            {
                NumMelBins = 64,
                FftSize = 512,
                HopLength = 160
            },
            Pooling = AudioPoolingStrategy.MeanPooling,
            Normalize = true,
            SampleRate = 16000
        };

        var estimator = new OnnxAudioEmbeddingEstimator(mlContext, embeddingOptions);
        var dummyData = mlContext.Data.LoadFromEnumerable(Array.Empty<AudioInput>());
        var transformer = estimator.Fit(dummyData);
        return new OnnxAudioEmbeddingGenerator(transformer);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    public static Task VerifyModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.VerifyModelAsync(options, ct);

    private sealed class AudioInput { public float[] Audio { get; set; } = []; }
}
