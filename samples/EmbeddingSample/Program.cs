using DotnetAILab.ModelGarden.Embeddings.AllMiniLM;
using Microsoft.Extensions.AI;
using System.Numerics.Tensors;

Console.WriteLine("=== Model Garden: Embedding Sample ===\n");

// One-liner: create embedding generator (model auto-downloads on first run)
Console.WriteLine("Creating embedding generator (model downloads on first use)...");
IEmbeddingGenerator<string, Embedding<float>> generator =
    await AllMiniLMModel.CreateEmbeddingGeneratorAsync();
Console.WriteLine("Generator ready!\n");

// Generate embeddings
var texts = new[]
{
    "What is machine learning?",
    "ML.NET is a machine learning framework for .NET",
    "How to cook pasta",
    "Deep learning and neural networks"
};

Console.WriteLine("Generating embeddings...");
var embeddings = await generator.GenerateAsync(texts);
Console.WriteLine($"Generated {embeddings.Count} embeddings, dimension: {embeddings[0].Vector.Length}\n");

// Cosine similarity
Console.WriteLine("Cosine Similarity:");
for (int i = 0; i < texts.Length; i++)
{
    for (int j = i + 1; j < texts.Length; j++)
    {
        float sim = TensorPrimitives.CosineSimilarity(
            embeddings[i].Vector.Span, embeddings[j].Vector.Span);
        Console.WriteLine($"  \"{texts[i]}\" vs \"{texts[j]}\": {sim:F4}");
    }
}

Console.WriteLine("\nDone!");
