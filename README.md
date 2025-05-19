# Rust Face Landmarks Demo
The 478 points face landmarks demo written in rust, runing on onnx platform.

![demo](screenshots\screenshot.png)

## Requirements
* ONNX Runtime 1.20 (DirectML)
```
pip install onnxruntime-directml==1.20.0
python copy_runtime.py
```

## Inference Model
* [YoloV5 Face](https://github.com/rede97?tab=repositories) detect face position
* [Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) from [Google Mediapipe](https://ai.google.dev/edge/mediapipe/solutions/guide) Genarate 478 landmarks
    * Convert TFlite to ONNX: [tensorflow-onnx](https://github.com/onnx/tensorflow-onnx)


# Profiling
* [Tracy profiler](https://github.com/wolfpld/tracy)