#!/usr/bin/env python3
"""
GTK3 wrapper for ws_nnstreamer_yolov8s_object_detection.py

Replaces waylandsink/cairooverlay with an appsink + Gtk.DrawingArea so the
YOLOv8 live feed (including WebSocket source) runs inside a regular GTK window
with an Exit button.

Source / inference logic is identical to the original demo; only the display
branch changes:
  cairooverlay ! fpsdisplaysink(waylandsink)
  →  appsink (pull BGRx frames each tick, draw boxes in GTK draw callback)
"""

import os
import sys
import gi
import logging
import threading
import time
import asyncio

import numpy as np
import cairo
import websockets

gi.require_version('Gtk', '3.0')
gi.require_version('Gst', '1.0')
gi.require_version('Pango', '1.0')
gi.require_version('PangoCairo', '1.0')

from gi.repository import Gtk, GLib, Gst, Pango, PangoCairo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nnstreamer_example import (
    argument_parser_init, find_cpu_cores, find_armnn_delegate_library,
    enable_performance,
)
import yolov8_postprocess as pp

DISPLAY_FPS = 30


# ── Pipeline manager ──────────────────────────────────────────────────────────

class PipelineManager:
    def __init__(self, cfg, status_cb):
        self.cfg       = cfg
        self._status   = status_cb
        self.pipeline  = None
        self._bus_id   = None
        self.running   = False

        self._ws_start_time = None

        self.detected      = []
        self.labels        = cfg['labels']
        self.conf_thres    = cfg['conf_thres']
        self.iou_thres     = cfg['iou_thres']
        self.invoke_ms     = 0.0

    # ── pipeline string ────────────────────────────────────────────────────────

    def _build_cmd(self):
        cfg = self.cfg
        cam_type  = cfg['cam_type']
        vid_w     = cfg['video_width']
        vid_h     = cfg['video_height']
        mod_w     = cfg['model_width']
        mod_h     = cfg['model_height']
        cam_id    = cfg['cam_id']
        cam_rot   = cfg['cam_rot']
        framework = cfg['framework']
        engine    = cfg['engine']
        throughput= cfg['throughput']
        tflite    = cfg['tflite_model']
        dla       = cfg['dla']

        # Source
        if cam_type == 'uvc':
            cmd = (f'v4l2src name=src device=/dev/video{cam_id} io-mode=mmap ! '
                   f'video/x-raw,width={vid_w},height={vid_h},format=YUY2 ! tee name=t_raw ')
        elif cam_type == 'yuvsensor':
            cmd = (f'v4l2src name=src device=/dev/video{cam_id} ! '
                   f'video/x-raw,width=1920,height=1080,format=UYVY ! tee name=t_raw ')
        elif cam_type == 'yuvsensor_d9':
            cmd = (f'v4l2src name=src device=/dev/video{cam_id} ! '
                   f'video/x-raw,width=1920,height=1080,format=YUY2 ! tee name=t_raw ')
        elif cam_type == 'rawsensor':
            cmd = (f'v4l2src name=src device=/dev/video{cam_id} ! '
                   f'video/x-raw,width=2048,height=1536,format=YUY2 ! tee name=t_raw ')
        elif cam_type == 'websocket':
            cmd = (f'appsrc name=src is-live=true format=time block=true '
                   f'caps=image/jpeg ! jpegdec ! videoconvert ! '
                   f'video/x-raw,format=BGRx ! tee name=t_raw ')
        else:
            raise ValueError(f'Unknown cam_type: {cam_type}')

        # Display branch → appsink (replaces cairooverlay + waylandsink)
        cmd += f't_raw. ! queue leaky=2 max-size-buffers=2 ! '
        if cam_type in ('uvc', 'websocket'):
            cmd += f'videoconvert ! video/x-raw,format=BGRx,width={vid_w},height={vid_h} ! '
        else:
            cmd += (f'v4l2convert output-io-mode=dmabuf-import '
                    f'extra-controls="cid,rotate={cam_rot}" ! '
                    f'video/x-raw,width={vid_w},height={vid_h},'
                    f'format=BGRx,pixel-aspect-ratio=1/1 ! ')
        cmd += ('appsink name=display_sink emit-signals=false '
                'max-buffers=1 drop=true sync=false ')

        # Inference branch
        cmd += f't_raw. ! queue leaky=2 max-size-buffers=2 ! '
        if cam_type in ('uvc', 'websocket'):
            cmd += (f'videoconvert ! videoscale ! '
                    f'video/x-raw,width={mod_w},height={mod_h},format=RGB ! ')
        else:
            cmd += (f'v4l2convert output-io-mode=dmabuf-import capture-io-mode=mmap '
                    f'extra-controls="cid,rotate={cam_rot}" ! '
                    f'video/x-raw,width={mod_w},height={mod_h},'
                    f'format=RGB,pixel-aspect-ratio=1/1 ! ')
        cmd += 'tensor_converter ! '
        cmd += 'tensor_transform mode=transpose option=1:2:0:3 ! '
        cmd += 'tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! '

        if framework == 'neuronsdk':
            cmd += (f'tensor_filter framework=neuronsdk throughput={throughput} name=nn '
                    f'model={dla} inputtype=float32 input={mod_w}:{mod_h}:3:1 '
                    f'outputtype=float32,float32,float32 '
                    f'output=80:80:96:1,40:40:96:1,20:20:96:1 !')
        elif framework == 'tflite':
            if engine == 'cpu':
                cores = find_cpu_cores()
                cmd += (f'tensor_filter framework=tensorflow-lite throughput={throughput} '
                        f'name=nn model={tflite} custom=NumThreads:{cores} ! ')
            elif engine == 'armnn':
                lib = find_armnn_delegate_library()
                cmd += (f'tensor_filter framework=tensorflow-lite throughput={throughput} '
                        f'name=nn model={tflite} '
                        f'custom=Delegate:External,ExtDelegateLib:{lib},'
                        f'ExtDelegateKeyVal:backends#GpuAcc ! ')
            elif engine == 'stable_delegate':
                cmd += (f'tensor_filter framework=tensorflow-lite throughput={throughput} '
                        f'name=nn model={tflite} '
                        f'custom=Delegate:Stable,'
                        f'StaDelegateSettingFile:/usr/share/label_image/stable_delegate_settings.json,'
                        f'ExtDelegateKeyVal:backends#GpuAcc ! ')

        cmd += 'tensor_sink name=res_sink '
        return cmd

    # ── start / stop ───────────────────────────────────────────────────────────

    def start(self):
        if self.pipeline:
            return
        cmd = self._build_cmd()
        logging.info("Pipeline:\n%s", cmd)
        self.pipeline = Gst.parse_launch(cmd)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        self._bus_id = bus.connect('message', self._on_bus_message)

        self.pipeline.get_by_name('res_sink').connect('new-data', self._on_new_data)

        if self.cfg.get('throughput') == '1':
            nn = self.pipeline.get_by_name('nn')
            if nn:
                nn.get_static_pad('src').add_probe(
                    Gst.PadProbeType.BUFFER, self._on_nn_buffer)

        self.pipeline.set_state(Gst.State.PLAYING)
        self.running = True

        if self.cfg['cam_type'] == 'websocket':
            t = threading.Thread(target=self._ws_thread, daemon=True)
            t.start()

        GLib.idle_add(self._status, 'Pipeline running')

    def stop(self):
        if not self.pipeline:
            return
        self.running = False
        self.pipeline.set_state(Gst.State.NULL)
        bus = self.pipeline.get_bus()
        if self._bus_id:
            bus.disconnect(self._bus_id)
            self._bus_id = None
        bus.remove_signal_watch()
        self.pipeline = None
        self.detected = []
        GLib.idle_add(self._status, 'Pipeline stopped')

    # ── pull display frame (called from GTK draw callback) ────────────────────

    def pull_display_frame(self):
        if not self.pipeline:
            return None
        sink = self.pipeline.get_by_name('display_sink')
        if not sink:
            return None
        sample = sink.emit('pull-sample')
        if not sample:
            return None
        buf  = sample.get_buffer()
        caps = sample.get_caps()
        w = caps.get_structure(0).get_value('width')
        h = caps.get_structure(0).get_value('height')
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None
        try:
            return np.frombuffer(info.data, dtype=np.uint8).reshape(h, w, 4).copy()
        finally:
            buf.unmap(info)

    # ── WebSocket thread ───────────────────────────────────────────────────────

    def _ws_thread(self):
        src = self.pipeline.get_by_name('src')
        self._ws_start_time = time.monotonic()

        async def _recv():
            logging.info('[ws] connecting to %s', self.cfg['ws_url'])
            async with websockets.connect(
                self.cfg['ws_url'],
                max_size=None,
                ping_interval=20,
                ping_timeout=20,
            ) as ws:
                logging.info('[ws] connected')
                async for message in ws:
                    if not self.running:
                        break
                    if not isinstance(message, bytes):
                        continue
                    pts = int((time.monotonic() - self._ws_start_time) * Gst.SECOND)
                    buf = Gst.Buffer.new_wrapped(bytes(message))
                    buf.pts = pts
                    buf.dts = pts
                    ret = src.emit('push-buffer', buf)
                    if ret != Gst.FlowReturn.OK:
                        logging.warning('[ws] push-buffer returned %s', ret)

        async def _run():
            try:
                await _recv()
            except Exception as e:
                logging.warning('[ws] error: %s', e)
            finally:
                if self.running and self.pipeline:
                    src.emit('end-of-stream')

        asyncio.run(_run())

    # ── tensor_sink callback ───────────────────────────────────────────────────

    def _on_new_data(self, sink, buffer):
        arrays = []
        for i in range(buffer.n_memory()):
            mem = buffer.peek_memory(i)
            ok, info = mem.map(Gst.MapFlags.READ)
            if ok:
                try:
                    arrays.append(np.frombuffer(info.data, dtype=np.float32).copy())
                finally:
                    mem.unmap(info)
        if arrays:
            try:
                self.detected = pp.detect(arrays, self.conf_thres, self.iou_thres)
            except Exception as e:
                logging.warning('[decode] %s', e)

    def _on_nn_buffer(self, pad, info):
        nn = self.pipeline.get_by_name('nn') if self.pipeline else None
        if nn:
            throughput = nn.get_property('throughput')
            if throughput > 0:
                self.invoke_ms = (1.0 / (throughput / 1000.0)) * 1000.0
        return Gst.PadProbeReturn.OK

    # ── GStreamer bus ──────────────────────────────────────────────────────────

    def _on_bus_message(self, bus, message):
        if message.type == Gst.MessageType.EOS:
            logging.info('[bus] EOS')
            self.stop()
        elif message.type == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            logging.error('[bus] %s : %s', err.message, dbg)
            GLib.idle_add(self._status, f'Error: {err.message}')
            self.stop()
        elif message.type == Gst.MessageType.WARNING:
            err, dbg = message.parse_warning()
            logging.warning('[bus] %s : %s', err.message, dbg)


# ── GTK Window ────────────────────────────────────────────────────────────────

class App(Gtk.Window):
    def __init__(self, cfg):
        super().__init__(title='  ')
        self.cfg = cfg
        vid_w = cfg['video_width']
        vid_h = cfg['video_height']
        self.set_default_size(vid_w, vid_h + 120)
        self.set_border_width(8)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(root)

        # Video DrawingArea
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.set_size_request(vid_w, vid_h)
        self.drawing_area.connect('draw', self._on_draw_frame)
        root.pack_start(self.drawing_area, True, True, 0)

        # Exit button
        root.pack_start(Gtk.Separator(), False, False, 2)
        btn_exit = Gtk.Button(label='Exit')
        btn_exit.set_size_request(-1, 44)
        btn_exit.get_style_context().remove_class('destructive-action')
        btn_exit.connect('clicked', self._on_exit)
        root.pack_start(btn_exit, False, False, 0)

        self.mgr = PipelineManager(cfg, status_cb=self._set_status)

        self._refresh_id  = None
        self._fps_count   = 0
        self._fps_time    = time.monotonic()
        self._display_fps = 0.0
        self.connect('destroy', self._on_exit)
        self.show_all()

        if cfg.get('fullscreen'):
            self.maximize()

        # Auto-start pipeline
        GLib.idle_add(self._start)

    # ── pipeline start / refresh timer ────────────────────────────────────────

    def _start(self):
        self.mgr.start()
        return False

    def _start_refresh(self):
        if self._refresh_id is None:
            self._refresh_id = GLib.timeout_add(
                int(1000 / DISPLAY_FPS), self._tick)

    def _stop_refresh(self):
        if self._refresh_id is not None:
            GLib.source_remove(self._refresh_id)
            self._refresh_id = None

    def _tick(self):
        self._fps_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_time
        if elapsed >= 1.0:
            self._display_fps = self._fps_count / elapsed
            self._fps_count   = 0
            self._fps_time    = now
        self.drawing_area.queue_draw()
        return True

    # ── cairo draw ────────────────────────────────────────────────────────────

    def _on_draw_frame(self, widget, ctx):
        alloc = widget.get_allocation()
        win_w, win_h = alloc.width, alloc.height

        ctx.set_source_rgb(0, 0, 0)
        ctx.paint()

        frame = self.mgr.pull_display_frame()
        if frame is None:
            ctx.set_source_rgb(0.3, 0.3, 0.3)
            ctx.select_font_face('Sans', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            ctx.set_font_size(20)
            ctx.move_to(win_w / 2 - 80, win_h / 2)
            ctx.show_text('Waiting for stream…')
            return

        fh, fw = frame.shape[:2]
        surface = cairo.ImageSurface.create_for_data(
            frame, cairo.FORMAT_RGB24, fw, fh, fw * 4)

        scale = min(win_w / fw, win_h / fh)
        off_x = (win_w - fw * scale) / 2
        off_y = (win_h - fh * scale) / 2

        ctx.save()
        ctx.translate(off_x, off_y)
        ctx.scale(scale, scale)
        ctx.set_source_surface(surface, 0, 0)
        ctx.paint()
        ctx.restore()

        # Top info bar
        info = (f'Camera FPS: {self._display_fps:.2f}'
                f'  Invoke Time(ms): {self.mgr.invoke_ms:.2f}')
        ctx.select_font_face('Sans', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(18)
        ctx.set_source_rgba(0, 0, 0, 0.6)
        ctx.rectangle(off_x, off_y, fw * scale, 28)
        ctx.fill()
        ctx.set_source_rgb(1, 1, 1)
        ctx.move_to(off_x + 6, off_y + 20)
        ctx.show_text(info)

        # Detection overlays
        dets   = self.mgr.detected
        labels = self.mgr.labels
        mod_w  = self.cfg['model_width']
        mod_h  = self.cfg['model_height']
        sx     = fw / float(mod_w)
        sy     = fh / float(mod_h)

        pango_layout = PangoCairo.create_layout(ctx)
        pango_layout.set_font_description(
            Pango.FontDescription('Noto Sans CJK TC Bold 12'))

        for (x1, y1, x2, y2, cls, score) in dets:
            rx  = off_x + x1 * sx * scale
            ry  = off_y + y1 * sy * scale
            rw  = (x2 - x1) * sx * scale
            rh  = (y2 - y1) * sy * scale

            ctx.set_source_rgb(1.0, 0.0, 0.0)
            ctx.set_line_width(2)
            ctx.rectangle(rx, ry, rw, rh)
            ctx.stroke()

            name = labels[cls] if 0 <= cls < len(labels) else str(cls)
            pango_layout.set_text(f'{name} {score:.2f}', -1)
            tx, ty = rx + 2, max(ry - 4, off_y + 14)
            ctx.move_to(tx, ty)
            ctx.set_source_rgb(0, 0, 0)
            ctx.set_line_width(3)
            PangoCairo.layout_path(ctx, pango_layout)
            ctx.stroke()
            ctx.move_to(tx, ty)
            ctx.set_source_rgb(1.0, 1.0, 0.0)
            PangoCairo.show_layout(ctx, pango_layout)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, msg):
        logging.info('[status] %s', msg)
        if 'running' in msg:
            self._start_refresh()
        elif 'stopped' in msg or 'Error' in msg:
            self._stop_refresh()
            self.drawing_area.queue_draw()
        return False

    def _on_exit(self, *_):
        self._stop_refresh()
        self.mgr.stop()
        Gtk.main_quit()


# ── Entry point ───────────────────────────────────────────────────────────────

def _load_labels(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    args = argument_parser_init()

    current_folder = os.path.dirname(os.path.abspath(__file__))
    tflite_model   = os.path.join(current_folder, 'yolov8s_mixed.tflite')
    dla            = os.path.join(current_folder, 'yolov8s_mixed.dla')
    label_file     = os.path.join(current_folder, 'mixed_label.txt')

    for path, name in ((tflite_model, 'tflite model'), (label_file, 'label file')):
        if not os.path.exists(path):
            logging.error('cannot find %s [%s]', name, path)
            sys.exit(1)

    labels = _load_labels(label_file)
    logging.info('loaded %d labels', len(labels))
    if len(labels) != pp.NC:
        logging.warning('label count %d != NC %d in yolov8_postprocess',
                        len(labels), pp.NC)

    # Resolve video dimensions (mirror original script logic)
    if args.cam_type == 'uvc':
        default_w, default_h = 640, 480
    elif args.cam_type == 'websocket':
        default_w, default_h = 1280, 720
    else:
        default_w, default_h = 1920, 1080

    vid_w = args.width  if args.width  else default_w
    vid_h = args.height if args.height else default_h

    cfg = dict(
        cam_type    = args.cam_type,
        cam_id      = args.cam,
        cam_rot     = args.rot,
        video_width = vid_w,
        video_height= vid_h,
        model_width = 640,
        model_height= 640,
        framework   = args.framework,
        engine      = args.engine,
        throughput  = args.throughput,
        tflite_model= tflite_model,
        dla         = dla,
        labels      = labels,
        conf_thres  = 0.60,
        iou_thres   = 0.45,
        ws_url      = args.ws_url,
        fullscreen  = args.fullscreen == '1',
    )

    enable_performance(args.performance)

    Gst.init(None)
    app = App(cfg)
    Gtk.main()


if __name__ == '__main__':
    main()
