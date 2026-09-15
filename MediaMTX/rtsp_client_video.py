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


def make_drop_until_keyframe_probe():
    """
    v4l2h264dec (MediaTek's HW decoder) errors out fatally ("poll error 1" /
    kernel log "need first seq header") if the very first buffer it receives
    isn't a keyframe -- which happens whenever the client joins the RTSP
    stream mid-GOP. Unlike avdec_h264, it doesn't tolerate/skip leading
    non-keyframe data, so we drop it ourselves before it reaches the decoder.
    """
    state = {'synced': False}

    def probe(pad, info):
        if state['synced']:
            return Gst.PadProbeReturn.OK
        buf = info.get_buffer()
        if buf.has_flags(Gst.BufferFlags.DELTA_UNIT):
            return Gst.PadProbeReturn.DROP
        state['synced'] = True
        return Gst.PadProbeReturn.OK

    return probe


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

    # v4l2h264dec (MediaTek's HW decoder) outputs a tiled MM21 format, which
    # plain videoconvert can't handle -- v4l2convert (MDP3) does the
    # MM21 -> RGB conversion in hardware instead.
    pipeline_str = (
        f'rtspsrc location={RTSP_URL} '
        'protocols=udp '
        'latency=50 '
        'drop-on-latency=true '
        'buffer-mode=none ! '
        'rtph264depay ! '
        'h264parse name=parser ! '
        'v4l2h264dec ! '
        'v4l2convert ! '
        'video/x-raw,format=RGB ! '
        'appsink name=sink '
        'emit-signals=true '
        'sync=false '
        'max-buffers=1 '
        'drop=true'
    )
    pipeline = Gst.parse_launch(pipeline_str)

    parser = pipeline.get_by_name('parser')
    parser.get_static_pad('src').add_probe(
        Gst.PadProbeType.BUFFER, make_drop_until_keyframe_probe(),
    )

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
