using Microsoft.Extensions.AI;
using Microsoft.ML;
using MLNet.TextInference.Onnx;
using ModelPackages;

namespace DotnetAILab.ModelGarden.Embeddings.AllMiniLM;

/// <summary>
/// Model package for all-MiniLM-L6-v2.
/// Downloads ONNX model on first use, caches locally.
/// </summary>
public static class AllMiniLMModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(AllMiniLMModel).Assembly));

    /// <summary>Returns local path to the cached ONNX model file.</summary>
    public static Task<string> EnsureModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureModelAsync(options, ct);

    /// <summary>
    /// Creates an IEmbeddingGenerator backed by the local ONNX model.
    /// Downloads the model on first call, cached thereafter.
    /// </summary>
    public static async Task<IEmbeddingGenerator<string, Embedding<float>>> CreateEmbeddingGeneratorAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var modelPath = await EnsureModelAsync(options, ct);
        var tokenizerDir = ExtractEmbeddedTokenizer();

        var mlContext = new MLContext();
        var estimator = new OnnxTextEmbeddingEstimator(mlContext, new OnnxTextEmbeddingOptions
        {
            ModelPath = modelPath,
            TokenizerPath = tokenizerDir,
            Pooling = PoolingStrategy.MeanPooling,
            Normalize = true,
            BatchSize = 32
        });

        var dummyData = mlContext.Data.LoadFromEnumerable(new[] { new TextData { Text = "" } });
        var transformer = estimator.Fit(dummyData);

        return new OnnxEmbeddingGenerator(mlContext, transformer, ownsTransformer: true);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    public static Task VerifyModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.VerifyModelAsync(options, ct);

    private static readonly string[] TokenizerFilePatterns =
        ["vocab.txt", "vocab.json", "merges.txt", "spm.model", "tokenizer.json",
         "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"];

    private static string ExtractEmbeddedTokenizer()
    {
        var assembly = typeof(AllMiniLMModel).Assembly;
        var tokenizerDir = Path.Combine(
            Path.GetTempPath(), "modelpackages-tokenizer", "AllMiniLM");
        Directory.CreateDirectory(tokenizerDir);

        foreach (var resourceName in assembly.GetManifestResourceNames())
        {
            var matchedFile = TokenizerFilePatterns
                .FirstOrDefault(p => resourceName.EndsWith(p, StringComparison.OrdinalIgnoreCase));
            if (matchedFile == null) continue;

            var targetPath = Path.Combine(tokenizerDir, matchedFile);
            if (!File.Exists(targetPath))
            {
                using var stream = assembly.GetManifestResourceStream(resourceName)!;
                using var file = File.Create(targetPath);
                stream.CopyTo(file);
            }
        }

        return tokenizerDir;
    }

    private sealed class TextData
    {
        public string Text { get; set; } = "";
    }
}
