#!/usr/bin/env python3
"""Standalone test for the int8 date-detection DLA model.

Runs yolov8s_bbox_date_20260807_int8.dla on a single image via a GStreamer
neuronsdk pipeline (same wiring as YoloInfer in image_receive.py, minus the
GTK/FastAPI app), prints the decoded detections, and saves an annotated copy.

Usage:
    python3 test_int8.py [image_path] [--out OUT_PATH]
"""

import argparse
import os
import sys
import threading

import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yolov8_postprocess as pp

DLA_MODEL   = "/py/yolov8s_bbox_date_20260807_int8.dla"
MODEL_W     = 640
MODEL_H     = 640
NC          = 1
LABELS      = ["date"]
CONF_THRES  = 0.45
IOU_THRES   = 0.45
DEFAULT_IMAGE = "/py/for_training/date/images/20260813_073407_789181_1.jpg"


def letterbox(img, w, h, fill=(114, 114, 114)):
    """Resize img to fit within w×h preserving aspect ratio, padding the rest.
    Returns (canvas, scale, pad_x, pad_y) so callers can map model-space
    coordinates back: orig = (model_coord - pad) / scale.
    """
    ow, oh = img.size
    scale = min(w / ow, h / oh)
    nw, nh = round(ow * scale), round(oh * scale)
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (w, h), fill)
    pad_x, pad_y = (w - nw) // 2, (h - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


class YoloInfer:
    """Single-image neuronsdk inference via GStreamer appsrc pipeline."""

    def __init__(self, model_path, nc, conf_thres, iou_thres,
                 model_w=MODEL_W, model_h=MODEL_H):
        Gst.init(None)
        self._event  = threading.Event()
        self._arrays = None
        self._lock   = threading.Lock()
        self.nc         = nc
        self.conf_thres = conf_thres
        self.iou_thres  = iou_thres
        self.model_w    = model_w
        self.model_h    = model_h

        ch = 64 + nc  # DFL box channels (64) + per-class logits
        cmd = (
            f'appsrc name=src format=time block=true '
            f'caps=video/x-raw,format=RGB,width={model_w},height={model_h},framerate=0/1 ! '
            'tensor_converter ! '
            'tensor_transform mode=transpose option=1:2:0:3 ! '
            'tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! '
            f'tensor_filter framework=neuronsdk model={model_path} '
            f'inputtype=float32 input={model_w}:{model_h}:3:1 '
            f'outputtype=float32,float32,float32 '
            f'output=80:80:{ch}:1,40:40:{ch}:1,20:20:{ch}:1 ! '
            'tensor_sink name=sink emit-signal=true sync=false'
        )
        self.pipeline = Gst.parse_launch(cmd)
        self.src = self.pipeline.get_by_name('src')
        sink = self.pipeline.get_by_name('sink')
        sink.connect('new-data', self._on_result)
        self.pipeline.set_state(Gst.State.PLAYING)

    def _on_result(self, sink, buffer):
        arrays = []
        for i in range(buffer.n_memory()):
            mem = buffer.peek_memory(i)
            ok, info = mem.map(Gst.MapFlags.READ)
            if ok:
                try:
                    arrays.append(np.frombuffer(info.data, dtype=np.float32).copy())
                finally:
                    mem.unmap(info)
        with self._lock:
            self._arrays = arrays
        self._event.set()

    def run(self, rgb_np):
        """rgb_np: uint8 ndarray (model_h, model_w, 3). Returns pp.detect() list."""
        self._event.clear()
        buf = Gst.Buffer.new_wrapped(rgb_np.tobytes())
        self.src.emit('push-buffer', buf)
        if not self._event.wait(timeout=5.0):
            return []
        with self._lock:
            arrays = self._arrays
        if not arrays:
            return []
        try:
            return pp.detect(arrays, conf_thres=self.conf_thres,
                             iou_thres=self.iou_thres, nc=self.nc)
        except Exception as e:
            print(f"[detect] {e}")
            return []

    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", default=DEFAULT_IMAGE,
                         help="path to the input image")
    parser.add_argument("--out", default=None,
                         help="path to save the annotated output image "
                              "(default: <image>_int8_out.jpg next to the input)")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f"image not found: {args.image}")
    if not os.path.isfile(DLA_MODEL):
        sys.exit(f"model not found: {DLA_MODEL}")

    out_path = args.out
    if out_path is None:
        root, ext = os.path.splitext(args.image)
        out_path = f"{root}_int8_out{ext or '.jpg'}"

    img = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    lb_img, scale, pad_x, pad_y = letterbox(img, MODEL_W, MODEL_H)

    infer = YoloInfer(DLA_MODEL, NC, CONF_THRES, IOU_THRES)
    dets = infer.run(np.array(lb_img, dtype=np.uint8))
    infer.stop()

    print(f"[test_int8] model={DLA_MODEL}")
    print(f"[test_int8] image={args.image} ({img.width}x{img.height})")
    print(f"[test_int8] {len(dets)} detection(s)")

    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2, cls, score) in dets:
        fx1 = (x1 - pad_x) / scale
        fy1 = (y1 - pad_y) / scale
        fx2 = (x2 - pad_x) / scale
        fy2 = (y2 - pad_y) / scale
        label = LABELS[cls] if cls < len(LABELS) else str(cls)
        print(f"  {label}: score={score:.4f} bbox=({fx1:.1f}, {fy1:.1f}, {fx2:.1f}, {fy2:.1f})")
        draw.rectangle([fx1, fy1, fx2, fy2], outline=(255, 0, 0), width=3)
        draw.text((fx1 + 2, max(0, fy1 - 14)), f"{label} {score:.2f}", fill=(255, 0, 0))

    img.save(out_path)
    print(f"[test_int8] saved annotated image to {out_path}")


if __name__ == "__main__":
    main()
