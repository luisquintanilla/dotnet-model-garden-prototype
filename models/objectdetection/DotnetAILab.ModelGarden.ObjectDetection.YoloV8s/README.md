# DotnetAILab.ModelGarden.ObjectDetection.YoloV8s

YOLOv8s object detection model package. Detects objects in images with bounding boxes and labels.

## Model Details

- **Model**: [ultralytics/yolov8s](https://huggingface.co/ultralytics/yolov8s)
- **Task**: Object Detection
- **Architecture**: YOLOv8 Small
- **Size**: ~45 MB (ONNX)
- **Output**: Bounding boxes with class labels and confidence scores

## Quick Start

```csharp
using DotnetAILab.ModelGarden.ObjectDetection.YoloV8s;
using Microsoft.ML.Data;

var detector = await YoloV8sModel.CreateDetectorAsync();
using var image = MLImage.CreateFromFile("photo.jpg");
var detections = detector.Detect(image);

foreach (var box in detections)
    Console.WriteLine(box);
```

## API

| Method | Description |
|--------|-------------|
| `CreateDetectorAsync()` | Creates an object detector (downloads model on first use) |
| `EnsureFilesAsync()` | Downloads model files and returns local paths |
| `GetModelInfoAsync()` | Returns model metadata |
| `VerifyModelAsync()` | Verifies cached model integrity |
```
