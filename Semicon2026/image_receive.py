import os
import sys
import ctypes
import threading
import time
import io
import uvicorn
from fastapi import FastAPI, UploadFile, File
from starlette.concurrency import run_in_threadpool
from PIL import Image, ImageOps
import numpy as np
import cairo
import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Gst', '1.0')
gi.require_version('Pango', '1.0')
gi.require_version('PangoCairo', '1.0')
from gi.repository import Gtk, GdkPixbuf, GLib, Gst, Pango, PangoCairo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yolov8_postprocess as pp

# ── YOLO config ───────────────────────────────────────────────────────────────

DLA_MODEL  = "/py/yolov8s_bbox_box_11cls_20260805.dla"
LABEL_FILE = "/py/box_11cls_labels.txt"
MODEL_W    = 640
MODEL_H    = 640
YOLO_NC    = 11
CONF_THRES = 0.80
IOU_THRES  = 0.45

with open(LABEL_FILE) as f:
    LABELS = [l.strip() for l in f if l.strip()]

# Only these box classes matter downstream (date-region detection); other
# classes detected by the 11-class model are dropped.
TARGET_BOX_LABELS = {"維他露P", "樂事洋芋片青檸口味"}

# Second-stage model: detects the date-print region *within* a box crop
# produced by the first-stage model above. Runs on the cropped/resized
# 640x640 image, so its output coordinates are relative to that crop and
# must be mapped back to the crop's offset in the original image.
DATE_DLA_MODEL  = "/py/yolov8s_bbox_date_20260807.dla"
DATE_LABELS     = ["date"]
DATE_NC         = 1
DATE_CONF_THRES = 0.45
DATE_IOU_THRES  = 0.45
DATE_BOX_COLOR  = (255, 255, 255)

# Demo override: show a fixed date string (instead of "date <score>") on the
# date-region box, keyed by the parent box's class label.
DATE_STR_BY_BOX_LABEL = {
    "樂事洋芋片青檸口味": "2027年1月13日",
    "維他露P": "2027年12月22日",
}

# ── GStreamer inference wrapper ────────────────────────────────────────────────

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


# ── Drawing helper ────────────────────────────────────────────────────────────

BOX_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 128, 255),
    (255, 128, 0), (128, 0, 255), (0, 255, 128),
]

# LABELS can contain Chinese text (e.g. box_11cls_labels.txt). PIL's own
# glyph lookup doesn't do font-fallback/shaping, so we render labels with
# Pango+Cairo instead, which resolves fonts through fontconfig properly.
# "Noto Sans CJK TC" is tried first (in case it's installed system-wide);
# otherwise fontconfig falls back to the "Noto Sans TC" file we bundle and
# register below, so labels render correctly even on a bare rootfs.
CJK_FONT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fonts", "NotoSansTC-VF.ttf"
)
LABEL_FONT_DESC = "Noto Sans CJK TC, Noto Sans TC Bold 20"


def _register_bundled_font():
    try:
        fc = ctypes.CDLL("libfontconfig.so.1")
        if not fc.FcConfigAppFontAddFile(None, CJK_FONT_PATH.encode("utf-8")):
            print(f"[font] FcConfigAppFontAddFile failed for {CJK_FONT_PATH}")
    except Exception as e:
        print(f"[font] could not register {CJK_FONT_PATH}: {e}")


_register_bundled_font()


def letterbox(img, w, h, fill=(114, 114, 114)):
    """Resize img to fit within w×h preserving aspect ratio, padding the rest.
    Returns (canvas, scale, pad_x, pad_y) so callers can map model-space
    coordinates back: orig = (model_coord - pad) / scale.

    Needed because a plain non-uniform resize badly distorts elongated crops
    (e.g. a narrow box face squished into a 640x640 square), which can push
    small text-region detections below the model's confidence threshold.
    """
    ow, oh = img.size
    scale = min(w / ow, h / oh)
    nw, nh = round(ow * scale), round(oh * scale)
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (w, h), fill)
    pad_x, pad_y = (w - nw) // 2, (h - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def draw_detections(img, boxes):
    """Draw bounding boxes + labels on img (PIL Image).

    boxes: list of (x1, y1, x2, y2, label, color), all already in img's own
    pixel space (i.e. the caller has done any model-space -> pixel-space
    scaling and cropped-region offsetting beforehand).
    """
    orig_w, orig_h = img.size

    # PIL RGB -> cairo ARGB32 surface (native-endian, i.e. BGRA bytes on LE).
    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, orig_w)
    bgra = np.zeros((orig_h, stride // 4, 4), dtype=np.uint8)
    rgb = np.array(img.convert('RGB'))
    bgra[:, :orig_w, 0] = rgb[:, :, 2]
    bgra[:, :orig_w, 1] = rgb[:, :, 1]
    bgra[:, :orig_w, 2] = rgb[:, :, 0]
    bgra[:, :orig_w, 3] = 255
    surface = cairo.ImageSurface.create_for_data(
        bytearray(bgra.tobytes()), cairo.FORMAT_ARGB32, orig_w, orig_h, stride
    )
    ctx = cairo.Context(surface)

    for (rx1, ry1, rx2, ry2, label, color) in boxes:
        r, g, b = (c / 255 for c in color)

        ctx.set_source_rgb(r, g, b)
        ctx.set_line_width(3)
        ctx.rectangle(rx1, ry1, rx2 - rx1, ry2 - ry1)
        ctx.stroke()

        layout = PangoCairo.create_layout(ctx)
        layout.set_font_description(Pango.FontDescription(LABEL_FONT_DESC))
        layout.set_text(label, -1)
        tw, th = layout.get_pixel_size()

        ctx.set_source_rgb(r, g, b)
        ctx.rectangle(rx1, ry1 - th - 4, tw + 4, th + 4)
        ctx.fill()

        ctx.set_source_rgb(0, 0, 0)
        ctx.move_to(rx1 + 2, ry1 - th - 2)
        PangoCairo.show_layout(ctx, layout)

    surface.flush()
    buf = np.ndarray(shape=(orig_h, stride // 4, 4), dtype=np.uint8, buffer=surface.get_data())
    out_rgb = buf[:, :orig_w, [2, 1, 0]]
    return Image.fromarray(out_rgb, 'RGB')


# ── Detection pipeline ────────────────────────────────────────────────────────
# Runs both models and returns both a drawing list (for the on-screen GTK
# preview) and a JSON-able result dict (for the /infer HTTP response). See
# README.md for the response schema consumed by the Android app.

# box_infer/date_infer are each a single stateful GStreamer pipeline (one
# in-flight buffer at a time), so concurrent /infer requests must not call
# run_pipeline() concurrently.
infer_lock = threading.Lock()


def run_pipeline(img):
    """img: PIL Image (already EXIF-transposed, any mode/size)."""
    t_start = time.perf_counter()
    orig_w, orig_h = img.size
    rgb_img = img.convert('RGB')

    # Stage 1: run box detector on a 640x640 resized copy of the full image.
    resized = rgb_img.resize((MODEL_W, MODEL_H), Image.BILINEAR)
    box_dets = box_infer.run(np.array(resized, dtype=np.uint8))
    box_dets = [d for d in box_dets
                if d[4] < len(LABELS) and LABELS[d[4]] in TARGET_BOX_LABELS]

    boxes_to_draw = []
    result_boxes = []
    sx = orig_w / MODEL_W
    sy = orig_h / MODEL_H

    for (x1, y1, x2, y2, cls, score) in box_dets:
        rx1, ry1 = x1 * sx, y1 * sy
        rx2, ry2 = x2 * sx, y2 * sy
        box_label = LABELS[cls] if cls < len(LABELS) else str(cls)
        date_str = DATE_STR_BY_BOX_LABEL.get(box_label)

        color = BOX_COLORS[cls % len(BOX_COLORS)]
        boxes_to_draw.append((rx1, ry1, rx2, ry2, f"{box_label} {score:.2f}", color))

        box_entry = {
            "label": box_label,
            "score": round(score, 4),
            "bbox": [round(rx1, 1), round(ry1, 1), round(rx2, 1), round(ry2, 1)],
            "date_str": date_str,
            "date_bbox": None,
            "date_score": None,
        }

        # Stage 2: crop this box out of the original image (crop origin =
        # (crx1, cry1) in original-image pixel space) and run the date
        # detector on a letterboxed 640x640 resize of the crop.
        crx1, cry1 = max(0, int(rx1)), max(0, int(ry1))
        crx2, cry2 = min(orig_w, int(rx2)), min(orig_h, int(ry2))
        if crx2 > crx1 and cry2 > cry1:
            crop = rgb_img.crop((crx1, cry1, crx2, cry2))
            crop_lb, lb_scale, pad_x, pad_y = letterbox(crop, MODEL_W, MODEL_H)
            date_dets = date_infer.run(np.array(crop_lb, dtype=np.uint8))
            if date_dets:
                # pp.detect() sorts by score descending; keep the best one.
                dx1, dy1, dx2, dy2, dcls, dscore = date_dets[0]
                # Date-model output is in the crop's letterboxed 640x640
                # space; undo the letterbox pad/scale to get crop-pixel
                # coords, then offset by the crop's origin in the original
                # image to land in original-image coords.
                fx1 = (dx1 - pad_x) / lb_scale + crx1
                fy1 = (dy1 - pad_y) / lb_scale + cry1
                fx2 = (dx2 - pad_x) / lb_scale + crx1
                fy2 = (dy2 - pad_y) / lb_scale + cry1
                dlabel = date_str if date_str else \
                    f"{DATE_LABELS[dcls] if dcls < len(DATE_LABELS) else dcls} {dscore:.2f}"
                boxes_to_draw.append((fx1, fy1, fx2, fy2, dlabel, DATE_BOX_COLOR))
                box_entry["date_bbox"] = [round(fx1, 1), round(fy1, 1), round(fx2, 1), round(fy2, 1)]
                box_entry["date_score"] = round(dscore, 4)

        result_boxes.append(box_entry)

    infer_time_ms = round((time.perf_counter() - t_start) * 1000, 1)
    print(f"[infer] overall inference time: {infer_time_ms} ms")

    result = {
        "image_width": orig_w,
        "image_height": orig_h,
        "inference_time_ms": infer_time_ms,
        "boxes": result_boxes,
    }
    return boxes_to_draw, result


# ── FastAPI + GTK app ─────────────────────────────────────────────────────────

app = FastAPI()
main_window = None
box_infer   = None
date_infer  = None


class MainWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="Genio 720 Viewer")
        self.maximize()
        self.image_widget = Gtk.Image()
        self.add(self.image_widget)
        self.show_all()

    def display(self, img_bytes, boxes_to_draw):
        """Decode img_bytes and show it (with boxes_to_draw overlaid, if any).
        No inference here -- run_pipeline() has already computed boxes_to_draw."""
        img = Image.open(io.BytesIO(img_bytes))
        img = ImageOps.exif_transpose(img).convert('RGB')
        if boxes_to_draw:
            img = draw_detections(img, boxes_to_draw)

        buf = io.BytesIO()
        img.save(buf, format='JPEG')

        loader = GdkPixbuf.PixbufLoader()
        loader.write(buf.getvalue())
        loader.close()
        pixbuf = loader.get_pixbuf()

        win_w = self.get_allocated_width()
        win_h = self.get_allocated_height()
        scale = min(win_w / pixbuf.get_width(), win_h / pixbuf.get_height())
        scaled = pixbuf.scale_simple(
            int(pixbuf.get_width() * scale),
            int(pixbuf.get_height() * scale),
            GdkPixbuf.InterpType.BILINEAR
        )
        self.image_widget.set_from_pixbuf(scaled)


@app.post('/infer')
async def infer(image: UploadFile = File(...)):
    img_bytes = await image.read()
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes)))

    def _run():
        with infer_lock:
            return run_pipeline(img)

    boxes_to_draw, result = await run_in_threadpool(_run)
    GLib.idle_add(main_window.display, img_bytes, boxes_to_draw)
    return result


def run_uvicorn():
    uvicorn.run(app, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    box_infer  = YoloInfer(DLA_MODEL, YOLO_NC, CONF_THRES, IOU_THRES)
    date_infer = YoloInfer(DATE_DLA_MODEL, DATE_NC, DATE_CONF_THRES, DATE_IOU_THRES)

    threading.Thread(target=run_uvicorn, daemon=True).start()

    main_window = MainWindow()
    main_window.connect("destroy", Gtk.main_quit)
    Gtk.main()
