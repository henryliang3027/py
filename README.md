# py

Edge AI experiments on MediaTek Genio: real-time YOLOv8 detection + ReID over
GStreamer/NNstreamer and ONNX Runtime NPU (Neuron EP), plus a carton/date
recognition app (Semicon2026).

- `MediaMTX/` — RTSP client pipelines, YOLOv8 detection + ReID tracking (CPU and Neuron EP variants)
- `NNstreamer/` — GStreamer/NNstreamer YOLOv8 inference demos and model assets
- `Semicon2026/` — box and date recognition app for the Genio 720 viewer, dataset splitting tools
- `TaiROS/` — YOLOv8 object detection for common items across everyday scenes (camera/RTSP/WebSocket sources, GStreamer/NNstreamer pipeline, CJK label overlay)
