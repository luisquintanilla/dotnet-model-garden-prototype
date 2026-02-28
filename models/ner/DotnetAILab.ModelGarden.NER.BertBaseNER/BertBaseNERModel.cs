using Microsoft.ML;
using MLNet.TextInference.Onnx;
using ModelPackages;

namespace DotnetAILab.ModelGarden.NER.BertBaseNER;

/// <summary>
/// Model package for BERT Base NER.
/// Downloads ONNX model on first use, caches locally.
/// </summary>
public static class BertBaseNERModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(BertBaseNERModel).Assembly));

    /// <summary>BIO entity labels for this NER model.</summary>
    public static readonly string[] Labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"];

    /// <summary>Returns local path to the cached ONNX model file.</summary>
    public static Task<string> EnsureModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureModelAsync(options, ct);

    /// <summary>
    /// Creates a NER transformer backed by the local ONNX model.
    /// Downloads the model on first call, cached thereafter.
    /// </summary>
    public static async Task<OnnxNerTransformer> CreateNerAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var modelPath = await EnsureModelAsync(options, ct);
        var tokenizerDir = ExtractEmbeddedTokenizer();

        var mlContext = new MLContext();
        var estimator = mlContext.Transforms.OnnxNer(new OnnxNerOptions
        {
            ModelPath = modelPath,
            TokenizerPath = tokenizerDir,
            Labels = Labels,
            MaxTokenLength = 128,
            BatchSize = 8
        });

        var dummyData = mlContext.Data.LoadFromEnumerable(new[] { new TextData { Text = "" } });
        return estimator.Fit(dummyData);
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
        var assembly = typeof(BertBaseNERModel).Assembly;
        var tokenizerDir = Path.Combine(
            Path.GetTempPath(), "modelpackages-tokenizer", "BertBaseNER");
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
