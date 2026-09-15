import os
import sys

import gi
import numpy as np
import cairo
import cv2
import onnxruntime as ort

gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

# yolov8_postprocess.py (raw-head YOLOv8 decode: DFL + anchor-free NMS) and the
# compiled .dla / label file all live in the NNstreamer demo project.
NNSTREAMER_DIR = '/py/NNstreamer'
sys.path.insert(0, NNSTREAMER_DIR)
import yolov8_postprocess as pp

RTSP_URL = "rtsp://192.168.51.77:8554/live/test"


VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

MODEL_INPUT = 640
DLA_MODEL = os.path.join(NNSTREAMER_DIR, 'yolov8s_fp32.dla')
LABELS_FILE = os.path.join(NNSTREAMER_DIR, 'coco_labels.txt')

CONF_THRES = 0.60
IOU_THRES = 0.45
PERSON_CLASS_ID = 0  # COCO 'person'

# ---- Re-ID (NVIDIA TAO ReIdentificationNet, ResNet50 / Market-1501) ----
# Re-ID needs one crop per detected person -- a variable-count input
# nnstreamer's static tensor_filter shapes can't express -- so it runs
# imperatively via onnxruntime, invoked from the YOLOv8 tensor_sink callback
# instead of being wired into the GStreamer graph. Accelerated on the APU via
# onnxruntime's NeuronExecutionProvider (see reidnet_neuron_test.py), with a
# CPU fallback if the Neuron EP isn't available or session creation fails.
MEDIAMTX_DIR = os.path.dirname(os.path.abspath(__file__))
REID_MODEL = os.path.join(MEDIAMTX_DIR, 'resnet50_market1501_aicity156.onnx')
REID_NEURON_OPTIONS = {
    "NEURON_FLAG_USE_FP16": "1",
    "NEURON_FLAG_MIN_GROUP_SIZE": "0",
    "NEURON_FLAG_OPTIMIZATION_STRING": "--opt=3 --num-mdla=1 --reshape-to-4d",
}
REID_INPUT_W = 128
REID_INPUT_H = 256
# Preprocessing per NVIDIA TAO docs (RGB, /255 then channel-wise normalize):
# https://docs.nvidia.com/tao/tao-toolkit/text/cv_finetuning/pytorch/re_identification/re_identification.html
REID_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
REID_STD = np.array([0.226, 0.226, 0.226], dtype=np.float32)
REID_MATCH_THRES = 0.65   # cosine similarity to reuse an existing track id
REID_EMA = 0.9            # weight kept on the existing gallery embedding
REID_MAX_AGE = 90         # frames a track can go unseen before it's dropped


def load_labels(path):
    with open(path, encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip() != '']
    if len(labels) != pp.NC:
        print(f"[warn] label count {len(labels)} != NC {pp.NC} in yolov8_postprocess")
    return labels


class ReidGallery:
    """
    Minimal appearance-only tracker: matches each new embedding against a
    gallery of previously-seen people by cosine similarity (embeddings are
    L2-normalized, so dot product == cosine similarity), greedily preferring
    the strongest match first so two people in the same frame can't both
    claim the same id. No motion model (Kalman/IoU) -- this is appearance
    matching only, good enough to keep an id stable across a static camera
    view, not a full MOT tracker.
    """

    def __init__(self):
        self.embeddings = {}   # track_id -> (256,) float32, L2-normalized
        self.last_seen = {}    # track_id -> frame_no
        self.next_id = 1
        self.frame_no = 0

    def assign(self, embeddings):
        self.frame_no += 1
        n = len(embeddings)
        gal_ids = list(self.embeddings.keys())

        pairs = []
        for i in range(n):
            for j, tid in enumerate(gal_ids):
                sim = float(np.dot(embeddings[i], self.embeddings[tid]))
                if sim >= REID_MATCH_THRES:
                    pairs.append((sim, i, j))
        pairs.sort(reverse=True)

        ids = [None] * n
        claimed_gal = set()
        for sim, i, j in pairs:
            if ids[i] is not None or gal_ids[j] in claimed_gal:
                continue
            ids[i] = gal_ids[j]
            claimed_gal.add(gal_ids[j])

        for i in range(n):
            if ids[i] is None:
                ids[i] = self.next_id
                self.next_id += 1
                self.embeddings[ids[i]] = embeddings[i]
            else:
                merged = REID_EMA * self.embeddings[ids[i]] + (1 - REID_EMA) * embeddings[i]
                self.embeddings[ids[i]] = merged / (np.linalg.norm(merged) + 1e-9)
            self.last_seen[ids[i]] = self.frame_no

        stale = [tid for tid, seen in self.last_seen.items() if self.frame_no - seen > REID_MAX_AGE]
        for tid in stale:
            self.embeddings.pop(tid, None)
            self.last_seen.pop(tid, None)

        return ids


class Detector:
    """
    Bridges the two halves of the pipeline: tensor_sink (streaming thread)
    decodes the raw YOLOv8 heads into boxes, cairooverlay's draw callback
    (also the streaming thread, but serialized with tensor_sink by GStreamer)
    paints them on the next display-branch frame. `detected` is replaced
    wholesale rather than mutated, so a torn read is never a half-written list.

    `latest_frame` (full-res BGRx, from the frame_sink appsink tap) is the
    same story: the frame_sink and tensor_sink callbacks run on different tee
    branches' streaming threads, so it's swapped by reference, never mutated.
    """

    def __init__(self, labels, reid_session=None):
        self.labels = labels
        self.detected = []
        self.overlay_valid = False
        self.frame_width = VIDEO_WIDTH
        self.frame_height = VIDEO_HEIGHT

        self.latest_frame = None
        self.reid_session = reid_session
        self.reid_input_name = reid_session.get_inputs()[0].name if reid_session else None
        self.gallery = ReidGallery()

    def on_new_frame(self, sink):
        sample = sink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.OK
        buf = sample.get_buffer()
        s = sample.get_caps().get_structure(0)
        w = s.get_int('width').value
        h = s.get_int('height').value
        ok, info = buf.map(Gst.MapFlags.READ)
        if ok:
            try:
                self.latest_frame = np.frombuffer(info.data, dtype=np.uint8).reshape(h, w, 4).copy()
            finally:
                buf.unmap(info)
        return Gst.FlowReturn.OK

    def _reid_embed(self, dets):
        """Crop each detected person out of the full-res frame, run them
        through the Re-ID model as one batch, and return a track id per
        detection (None for non-person classes or if a frame isn't ready
        yet)."""
        ids = [None] * len(dets)
        frame = self.latest_frame
        if self.reid_session is None or frame is None or not dets:
            return ids

        fh, fw = frame.shape[0], frame.shape[1]
        sx = fw / float(MODEL_INPUT)
        sy = fh / float(MODEL_INPUT)

        crops, idxs = [], []
        for i, (x1, y1, x2, y2, cls, score) in enumerate(dets):
            px1, py1 = max(int(x1 * sx), 0), max(int(y1 * sy), 0)
            px2, py2 = min(int(x2 * sx), fw), min(int(y2 * sy), fh)
            if px2 - px1 < 8 or py2 - py1 < 8:
                continue
            crop_bgr = frame[py1:py2, px1:px2, :3]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crops.append(cv2.resize(crop_rgb, (REID_INPUT_W, REID_INPUT_H), interpolation=cv2.INTER_LINEAR))
            idxs.append(i)

        if not crops:
            return ids

        try:
            # The model's onnx export has a fixed batch dim of 1 (not a
            # dynamic axis), so multiple people can't be stacked into one
            # inference call -- run each crop separately instead.
            embs = []
            for crop in crops:
                x = crop.astype(np.float32) / 255.0
                x = (x - REID_MEAN) / REID_STD
                x = x.transpose(2, 0, 1)[np.newaxis, ...]  # HWC -> NCHW, N=1
                embs.append(self.reid_session.run(None, {self.reid_input_name: x})[0][0])
            embs = np.stack(embs)
            embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
            track_ids = self.gallery.assign(embs)
            for idx, tid in zip(idxs, track_ids):
                ids[idx] = tid
        except Exception as e:
            print(f"[reid error] {e}")

        return ids

    def on_new_data(self, sink, buffer):
        arrays = []
        for i in range(buffer.n_memory()):
            mem = buffer.peek_memory(i)
            ok, info = mem.map(Gst.MapFlags.READ)
            if ok:
                try:
                    arrays.append(np.frombuffer(info.data, dtype=np.float32).copy())
                finally:
                    mem.unmap(info)
        if not arrays:
            return
        try:
            dets = pp.detect(arrays, CONF_THRES, IOU_THRES)
        except Exception as e:
            print(f"[postprocess error] {e}")
            return

        dets = [d for d in dets if d[4] == PERSON_CLASS_ID]

        track_ids = self._reid_embed(dets)
        self.detected = [d + (tid,) for d, tid in zip(dets, track_ids)]

    def on_caps_changed(self, overlay, caps):
        self.overlay_valid = True
        s = caps.get_structure(0)
        self.frame_width = s.get_int('width').value
        self.frame_height = s.get_int('height').value

    def on_draw(self, overlay, context, timestamp, duration):
        if not self.overlay_valid:
            return
        dets = self.detected
        if not dets:
            return

        sx = self.frame_width / float(MODEL_INPUT)
        sy = self.frame_height / float(MODEL_INPUT)

        context.select_font_face('sans-serif', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        context.set_font_size(58)

        for (x1, y1, x2, y2, cls, score, track_id) in dets:
            rx, ry = x1 * sx, y1 * sy
            rw, rh = (x2 - x1) * sx, (y2 - y1) * sy

            context.set_source_rgb(0.0, 1.0, 0.2)
            context.set_line_width(2)
            context.rectangle(rx, ry, rw, rh)
            context.stroke()

            name = self.labels[cls] if 0 <= cls < len(self.labels) else str(cls)
            # label = f'{name}#{track_id} {score:.2f}' if track_id is not None else f'{name} {score:.2f}'
            label = f'#{track_id}' if track_id is not None else f''
            extents = context.text_extents(label)
            tx, ty = rx + 2, max(ry - 6, extents.height + 4)

            context.set_source_rgba(0.0, 0.0, 0.0, 0.6)
            context.rectangle(tx - 2, ty - extents.height - 2, extents.width + 4, extents.height + 6)
            context.fill()

            context.set_source_rgb(0.0, 1.0, 0.2)
            context.move_to(tx, ty)
            context.show_text(label)


def on_bus_message(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"[gst error] {err}: {debug}")
        loop.quit()
    elif t == Gst.MessageType.EOS:
        print("[gst] end of stream")
        loop.quit()
    return True


def build_pipeline():
    # Decode side: same UDP/avdec_h264 setup as rtsp_client_webcam.py --
    # v4l2h264dec fails to negotiate against this live RTSP source on this
    # board (see rtsp_client_webcam.py for the root cause), so software
    # decode is used here too. Output is converted straight to ARGB since
    # that's what cairooverlay draws onto.
    cmd = (
        f'rtspsrc location={RTSP_URL} protocols=udp do-rtcp=false latency=50 ! '
        f'rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! '
        f'video/x-raw,format=BGRx,width={VIDEO_WIDTH},height={VIDEO_HEIGHT} ! '
        f'tee name=t_raw '

        # ---- display branch: cairooverlay paints boxes onto the full-res frame ----
        f't_raw. ! queue leaky=2 max-size-buffers=10 ! '
        f'cairooverlay name=overlay ! videoconvert ! '
        f'waylandsink name=sink sync=false qos=false '

        # ---- inference branch: resize to model input, run on the APU (neuronsdk / NeuronRT) ----
        f't_raw. ! queue leaky=2 max-size-buffers=2 ! '
        f'videoconvert ! videoscale ! '
        f'video/x-raw,width={MODEL_INPUT},height={MODEL_INPUT},format=RGB ! '
        f'tensor_converter ! '
        f'tensor_transform mode=transpose option=1:2:0:3 ! '
        f'tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! '
        f'tensor_filter framework=neuronsdk name=nn model={DLA_MODEL} '
        f'inputtype=float32 input={MODEL_INPUT}:{MODEL_INPUT}:3:1 '
        f'outputtype=float32,float32,float32 '
        f'output=80:80:{4 * pp.REG_MAX + pp.NC}:1,40:40:{4 * pp.REG_MAX + pp.NC}:1,20:20:{4 * pp.REG_MAX + pp.NC}:1 ! '
        f'tensor_sink name=res_sink '

        # ---- frame tap: full-res BGRx pulled into Python for Re-ID crops ----
        f't_raw. ! queue leaky=2 max-size-buffers=2 ! '
        f'appsink name=frame_sink emit-signals=true sync=false max-buffers=1 drop=true'
    )
    print(f"pipeline: {cmd}")
    return Gst.parse_launch(cmd)


def main():
    Gst.init(None)

    if not os.path.exists(DLA_MODEL):
        print(f"[error] cannot find dla model [{DLA_MODEL}]")
        return
    if not os.path.exists(LABELS_FILE):
        print(f"[error] cannot find label file [{LABELS_FILE}]")
        return

    reid_session = None
    if os.path.exists(REID_MODEL):
        try:
            if "NeuronExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("NeuronExecutionProvider not available")
            reid_session = ort.InferenceSession(
                REID_MODEL,
                providers=[("NeuronExecutionProvider", REID_NEURON_OPTIONS)],
            )
            print("[reid] using NeuronExecutionProvider")
        except Exception as e:
            print(f"[warn] failed to init reid on Neuron EP ({e}), falling back to CPU")
            reid_session = ort.InferenceSession(REID_MODEL, providers=['CPUExecutionProvider'])
    else:
        print(f"[warn] cannot find reid model [{REID_MODEL}], running without re-identification")

    labels = load_labels(LABELS_FILE)
    detector = Detector(labels, reid_session)

    pipeline = build_pipeline()

    overlay = pipeline.get_by_name('overlay')
    overlay.connect('draw', detector.on_draw)
    overlay.connect('caps-changed', detector.on_caps_changed)

    res_sink = pipeline.get_by_name('res_sink')
    res_sink.connect('new-data', detector.on_new_data)

    frame_sink = pipeline.get_by_name('frame_sink')
    frame_sink.connect('new-sample', detector.on_new_frame)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect('message', on_bus_message, loop)

    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    main()
