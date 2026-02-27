using Microsoft.ML;
using MLNet.TextInference.Onnx;
using ModelPackages;

namespace DotnetAILab.ModelGarden.Reranking.MsMarcoMiniLM;

/// <summary>
/// Model package for MS MARCO MiniLM L-6 v2.
/// Downloads ONNX model on first use, caches locally.
/// </summary>
public static class MsMarcoMiniLMModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(MsMarcoMiniLMModel).Assembly));

    /// <summary>Returns local path to the cached ONNX model file.</summary>
    public static Task<string> EnsureModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureModelAsync(options, ct);

    /// <summary>
    /// Creates a reranker transformer backed by the local ONNX model.
    /// Downloads the model on first call, cached thereafter.
    /// </summary>
    public static async Task<OnnxRerankerTransformer> CreateRerankerAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var modelPath = await EnsureModelAsync(options, ct);
        var vocabPath = ExtractEmbeddedVocab();

        var mlContext = new MLContext();
        var estimator = mlContext.Transforms.OnnxRerank(new OnnxRerankerOptions
        {
            ModelPath = modelPath,
            TokenizerPath = vocabPath,
            MaxTokenLength = 512,
            BatchSize = 8
        });

        var dummyData = mlContext.Data.LoadFromEnumerable(new[] { new TextPairData { Query = "", Document = "" } });
        return estimator.Fit(dummyData);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    public static Task VerifyModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.VerifyModelAsync(options, ct);

    private static string ExtractEmbeddedVocab()
    {
        var assembly = typeof(MsMarcoMiniLMModel).Assembly;
        var resourceName = assembly.GetManifestResourceNames()
            .FirstOrDefault(n => n.EndsWith("vocab.txt", StringComparison.OrdinalIgnoreCase));

        if (resourceName == null)
            throw new FileNotFoundException("Embedded resource 'vocab.txt' not found in assembly.");

        var tempDir = Path.Combine(Path.GetTempPath(), "modelpackages-vocab");
        Directory.CreateDirectory(tempDir);
        var vocabPath = Path.Combine(tempDir, "vocab.txt");

        if (!File.Exists(vocabPath))
        {
            using var stream = assembly.GetManifestResourceStream(resourceName)!;
            using var file = File.Create(vocabPath);
            stream.CopyTo(file);
        }

        return vocabPath;
    }

    private sealed class TextPairData
    {
        public string Query { get; set; } = "";
        public string Document { get; set; } = "";
    }
}
