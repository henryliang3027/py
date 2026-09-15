import os
import sys
import ctypes
import threading
import queue
import io
from datetime import datetime
import uvicorn
from fastapi import FastAPI, UploadFile, File
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
CONF_THRES = 0.40
IOU_THRES  = 0.45

with open(LABEL_FILE) as f:
    LABELS = [l.strip() for l in f if l.strip()]

# Second-stage model: detects the date-print region *within* a box crop
# produced by the first-stage model above.
DATE_DLA_MODEL  = "/py/yolov8s_bbox_date_20260807.dla"
DATE_NC         = 1
DATE_CONF_THRES = 0.45
DATE_IOU_THRES  = 0.45

# ── Auto-labeled training data output ──────────────────────────────────────────
# box/  : full received images + YOLO-format labels from the box model, for
#         (re)training the box_11cls model.
# date/ : box crops (cropped from the same full images) + YOLO-format labels
#         from the date model, for training the date model.
CROP_DIR       = "/py/for_training"
BOX_IMG_DIR    = os.path.join(CROP_DIR, "box", "images")
BOX_LABEL_DIR  = os.path.join(CROP_DIR, "box", "labels")
DATE_IMG_DIR   = os.path.join(CROP_DIR, "date", "images")
DATE_LABEL_DIR = os.path.join(CROP_DIR, "date", "labels")
for _d in (BOX_IMG_DIR, BOX_LABEL_DIR, DATE_IMG_DIR, DATE_LABEL_DIR):
    os.makedirs(_d, exist_ok=True)

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


def draw_detections(img, dets):
    """Draw bounding boxes + labels on img (PIL Image). dets are in MODEL_W×MODEL_H space."""
    orig_w, orig_h = img.size
    sx = orig_w / MODEL_W
    sy = orig_h / MODEL_H

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

    for (x1, y1, x2, y2, cls, score) in dets:
        rx1 = x1 * sx
        ry1 = y1 * sy
        rx2 = x2 * sx
        ry2 = y2 * sy
        color = BOX_COLORS[cls % len(BOX_COLORS)]
        r, g, b = (c / 255 for c in color)
        label = f"{LABELS[cls] if cls < len(LABELS) else cls} {score:.2f}"

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


def _yolo_line(cls, x1, y1, x2, y2, img_w, img_h):
    """One YOLO-format label line: 'class x_center y_center width height',
    all normalized to [0, 1] by img_w/img_h."""
    xc = (x1 + x2) / 2 / img_w
    yc = (y1 + y2) / 2 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def save_box_labels(img, dets, ts):
    """Save the full received image + a YOLO-format label file (one line per
    detected box, normalized to the full image) under BOX_IMG_DIR/BOX_LABEL_DIR."""
    orig_w, orig_h = img.size
    sx = orig_w / MODEL_W
    sy = orig_h / MODEL_H

    lines = []
    for (x1, y1, x2, y2, cls, score) in dets:
        rx1 = max(0.0, x1 * sx)
        ry1 = max(0.0, y1 * sy)
        rx2 = min(float(orig_w), x2 * sx)
        ry2 = min(float(orig_h), y2 * sy)
        if rx2 <= rx1 or ry2 <= ry1:
            continue
        lines.append(_yolo_line(cls, rx1, ry1, rx2, ry2, orig_w, orig_h))
    if not lines:
        return

    img.convert('RGB').save(os.path.join(BOX_IMG_DIR, f"{ts}.jpg"))
    with open(os.path.join(BOX_LABEL_DIR, f"{ts}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def save_date_labels(img, dets, ts):
    """For each detected box: crop it out of img, run the date model on a
    letterboxed 640x640 resize of the crop, and (if anything was found) save
    the crop image + a YOLO-format label file (normalized to the crop) under
    DATE_IMG_DIR/DATE_LABEL_DIR."""
    orig_w, orig_h = img.size
    sx = orig_w / MODEL_W
    sy = orig_h / MODEL_H
    rgb_img = img.convert('RGB')

    for i, (x1, y1, x2, y2, cls, score) in enumerate(dets):
        crx1 = max(0, int(x1 * sx))
        cry1 = max(0, int(y1 * sy))
        crx2 = min(orig_w, int(x2 * sx))
        cry2 = min(orig_h, int(y2 * sy))
        if crx2 <= crx1 or cry2 <= cry1:
            continue

        crop = rgb_img.crop((crx1, cry1, crx2, cry2))
        crop_w, crop_h = crop.size
        crop_lb, lb_scale, pad_x, pad_y = letterbox(crop, MODEL_W, MODEL_H)
        date_dets = date_infer.run(np.array(crop_lb, dtype=np.uint8))
        if not date_dets:
            continue

        lines = []
        for (dx1, dy1, dx2, dy2, dcls, dscore) in date_dets:
            # Date-model output is in the crop's letterboxed 640x640 space;
            # undo the letterbox pad/scale to land back in crop-pixel space.
            fx1 = max(0.0, (dx1 - pad_x) / lb_scale)
            fy1 = max(0.0, (dy1 - pad_y) / lb_scale)
            fx2 = min(float(crop_w), (dx2 - pad_x) / lb_scale)
            fy2 = min(float(crop_h), (dy2 - pad_y) / lb_scale)
            if fx2 <= fx1 or fy2 <= fy1:
                continue
            lines.append(_yolo_line(dcls, fx1, fy1, fx2, fy2, crop_w, crop_h))
        if not lines:
            continue

        suffix = f"_{i}" if len(dets) > 1 else ""
        crop.save(os.path.join(DATE_IMG_DIR, f"{ts}{suffix}.jpg"))
        with open(os.path.join(DATE_LABEL_DIR, f"{ts}{suffix}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")


# ── FastAPI + GTK app ─────────────────────────────────────────────────────────

app = FastAPI()
main_window = None
img_queue   = queue.Queue()
box_infer   = None
date_infer  = None


class MainWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="Genio 720 Viewer")
        self.maximize()
        self.image_widget = Gtk.Image()
        self.add(self.image_widget)
        self.show_all()

    def update_image(self, img_bytes):
        img = Image.open(io.BytesIO(img_bytes))
        img = ImageOps.exif_transpose(img)

        # run YOLO on 640×640 resized copy, then draw on original
        resized = img.resize((MODEL_W, MODEL_H), Image.BILINEAR).convert('RGB')
        dets = box_infer.run(np.array(resized, dtype=np.uint8))
        if dets:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_box_labels(img, dets, ts)
            save_date_labels(img, dets, ts)
            img = draw_detections(img.convert('RGB'), dets)

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
    img_queue.put(img_bytes)
    return {"status": "queued"}


def process_queue():
    while True:
        img_bytes = img_queue.get()
        GLib.idle_add(main_window.update_image, img_bytes)
        img_queue.task_done()


def run_uvicorn():
    uvicorn.run(app, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    box_infer  = YoloInfer(DLA_MODEL, YOLO_NC, CONF_THRES, IOU_THRES)
    date_infer = YoloInfer(DATE_DLA_MODEL, DATE_NC, DATE_CONF_THRES, DATE_IOU_THRES)

    threading.Thread(target=run_uvicorn, daemon=True).start()
    threading.Thread(target=process_queue, daemon=True).start()

    main_window = MainWindow()
    main_window.connect("destroy", Gtk.main_quit)
    Gtk.main()
