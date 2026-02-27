using DotnetAILab.ModelGarden.Classification.SentimentDistilBERT;

Console.WriteLine("=== Model Garden: Classification Sample ===\n");

// One-liner: create classifier (model auto-downloads on first run)
Console.WriteLine("Creating sentiment classifier (model downloads on first use)...");
var classifier = await SentimentDistilBERTModel.CreateClassifierAsync();
Console.WriteLine("Classifier ready!\n");

Console.WriteLine("Available labels: " + string.Join(", ", SentimentDistilBERTModel.Labels));

Console.WriteLine("\nDone! Use classifier.Transform(dataView) with ML.NET pipelines.");
