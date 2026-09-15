import sys
import threading

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Gst', '1.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib, Gst

RTSP_URL = "rtsp://192.168.51.77:8554/live/test"


class MainWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="RTSP Client")
        self.set_default_size(960, 640)
        self.set_resizable(False)
        # A GtkImage's size request tracks the natural size of whatever
        # pixbuf it's showing, so the window kept growing to the decoded
        # frame's raw resolution once playback started. GtkDrawingArea has no
        # such content-driven size request, so painting the scaled frame
        # ourselves in 'draw' keeps the window truly fixed at default_size.
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.connect('draw', self._on_draw)
        self.add(self.drawing_area)
        self.show_all()

        # Only the newest frame is kept; if the GTK main loop can't keep up,
        # older frames are overwritten instead of piling up as queued idle
        # callbacks (which previously caused growing display latency).
        self._lock = threading.Lock()
        self._latest_frame = None
        self._update_pending = False
        self._current_pixbuf = None

    def queue_frame(self, rgb_bytes, width, height):
        with self._lock:
            self._latest_frame = (rgb_bytes, width, height)
            if self._update_pending:
                return
            self._update_pending = True
        GLib.idle_add(self._display_latest)

    def _display_latest(self):
        with self._lock:
            frame = self._latest_frame
            self._latest_frame = None
            self._update_pending = False
        if frame is not None:
            self.display(*frame)
        return GLib.SOURCE_REMOVE

    def display(self, rgb_bytes, width, height):
        self._current_pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
            GLib.Bytes.new(rgb_bytes),
            GdkPixbuf.Colorspace.RGB,
            False, 8, width, height, width * 3,
        )
        self.drawing_area.queue_draw()

    def _on_draw(self, widget, cr):
        if self._current_pixbuf is None:
            return False

        win_w = widget.get_allocated_width()
        win_h = widget.get_allocated_height()
        pw = self._current_pixbuf.get_width()
        ph = self._current_pixbuf.get_height()
        scale = min(win_w / pw, win_h / ph)
        new_w = max(1, int(pw * scale))
        new_h = max(1, int(ph * scale))
        scaled = self._current_pixbuf.scale_simple(
            new_w, new_h, GdkPixbuf.InterpType.BILINEAR,
        )

        Gdk.cairo_set_source_pixbuf(
            cr, scaled, (win_w - new_w) / 2, (win_h - new_h) / 2,
        )
        cr.paint()
        return False


def on_new_sample(sink, window):
    sample = sink.emit('pull-sample')
    if sample is None:
        return Gst.FlowReturn.OK

    buf = sample.get_buffer()
    caps = sample.get_caps()
    structure = caps.get_structure(0)
    width = structure.get_value('width')
    height = structure.get_value('height')

    ok, info = buf.map(Gst.MapFlags.READ)
    if ok:
        try:
            rgb_bytes = bytes(info.data)
        finally:
            buf.unmap(info)
        window.queue_frame(rgb_bytes, width, height)

    return Gst.FlowReturn.OK


def on_bus_message(bus, message, loop_label):
    t = message.type
    if t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"[gst error] {err}: {debug}")
    elif t == Gst.MessageType.EOS:
        print("[gst] end of stream")
    return True


def main():
    Gst.init(None)

    # v4l2h264dec (MediaTek's HW decoder) fails to negotiate against this
    # live RTSP source on this board: h264parse's first caps event (before
    # it has parsed the SPS) carries no width/height, so v4l2h264dec opens
    # the V4L2 device at a placeholder 320x240 and then refuses to
    # renegotiate once h264parse follows up with the real 1280x720 caps,
    # producing a not-negotiated error every time. avdec_h264 (software)
    # doesn't have this failure mode and also tolerates a mid-GOP join, so
    # it's used here instead.
    pipeline_str = (
        f'rtspsrc location={RTSP_URL} '
        'protocols=udp '
        'do-rtcp=false '
        'latency=50 ! '
        'rtph264depay ! '
        'h264parse ! '
        'avdec_h264 ! '
        'videoconvert ! '
        'video/x-raw,format=RGB ! '
        'appsink name=sink '
        'emit-signals=true '
        'sync=false '
        'max-buffers=1 '
        'drop=true'
    )
    pipeline = Gst.parse_launch(pipeline_str)

    window = MainWindow()
    window.connect("destroy", Gtk.main_quit)

    sink = pipeline.get_by_name('sink')
    sink.connect('new-sample', on_new_sample, window)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect('message', on_bus_message, None)

    pipeline.set_state(Gst.State.PLAYING)

    try:
        Gtk.main()
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    main()
