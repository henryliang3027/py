import os

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

# Minimal player, no detection/inference branch at all -- just decode and
# display. Used to isolate whether playback speed/lag issues come from the
# decode+sync path itself or from the YOLOv8/Re-ID inference load competing
# for CPU in yolov8s_reidnet_video.py.
VIDEO_FILE = "/py/MediaMTX/video1.avi"


def on_bus_message(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"[gst error] {err}: {debug}")
        loop.quit()
    elif t == Gst.MessageType.EOS:
        print("[gst] end of stream")
        loop.quit()
    elif t == Gst.MessageType.QOS:
        live, running_time, stream_time, timestamp, duration = message.parse_qos()
        print(f"[qos] late buffer, dropping to catch up (live={live})")
    return True


def build_pipeline():
    cmd = (
        f'filesrc location={VIDEO_FILE} ! decodebin ! videoconvert ! '
        f'waylandsink name=sink sync=true qos=true'
    )
    print(f"pipeline: {cmd}")
    return Gst.parse_launch(cmd)


def main():
    Gst.init(None)

    if not os.path.exists(VIDEO_FILE):
        print(f"[error] cannot find video file [{VIDEO_FILE}]")
        return

    pipeline = build_pipeline()

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
