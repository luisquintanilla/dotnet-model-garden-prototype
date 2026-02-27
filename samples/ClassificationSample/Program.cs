using DotnetAILab.ModelGarden.Classification.SentimentDistilBERT;

Console.WriteLine("=== Model Garden: Classification Sample ===\n");

// One-liner: create classifier (model auto-downloads on first run)
Console.WriteLine("Creating sentiment classifier (model downloads on first use)...");
var classifier = await SentimentDistilBERTModel.CreateClassifierAsync();
Console.WriteLine("Classifier ready!\n");

Console.WriteLine("Available labels: " + string.Join(", ", SentimentDistilBERTModel.Labels));

// Classify some texts
var texts = new[]
{
    "This movie was absolutely wonderful!",
    "The food was terrible and the service was slow.",
    "I love programming in C# with ML.NET",
    "The weather is okay today."
};

Console.WriteLine("\nClassifying texts:");
var results = classifier.Classify(texts);
for (int i = 0; i < texts.Length; i++)
{
    Console.WriteLine($"  \"{texts[i]}\"");
    Console.WriteLine($"    → {results[i].PredictedLabel} (confidence: {results[i].Confidence:P1})");
}

Console.WriteLine("\nDone!");
