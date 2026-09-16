#!/usr/bin/env python3
"""
Minimal RTP/UDP receiver — decodes H264 from udpsrc and displays via waylandsink.
Use this to verify rtsp_server.py is streaming correctly before running the full demo.

Usage:
  python3 test_rtsp.py [--port 5600]
"""

import argparse
import logging
import signal
import sys

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--port',      default=5600,      type=int)
    parser.add_argument('--multicast', default='224.1.1.1')
    args = parser.parse_args()

    Gst.init(None)

    pipeline_str = (
        f'udpsrc address={args.multicast} port={args.port} '
        f'multicast-iface=eth0 auto-multicast=true '
        f'caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! '
        f'rtph264depay ! h264parse ! avdec_h264 ! '
        f'videoconvert ! waylandsink sync=false'
    )

    logging.info('Pipeline: %s', pipeline_str)
    pipeline = Gst.parse_launch(pipeline_str)

    bus = pipeline.get_bus()
    bus.add_signal_watch()

    loop = GLib.MainLoop()

    def on_message(bus, msg):
        if msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            logging.error('%s\n%s', err, dbg)
            loop.quit()
        elif msg.type == Gst.MessageType.WARNING:
            err, dbg = msg.parse_warning()
            logging.warning('%s\n%s', err, dbg)
        elif msg.type == Gst.MessageType.EOS:
            logging.info('EOS')
            loop.quit()
        elif msg.type == Gst.MessageType.STATE_CHANGED:
            if msg.src == pipeline:
                old, new, _ = msg.parse_state_changed()
                logging.info('State: %s -> %s', old.value_nick, new.value_nick)

    bus.connect('message', on_message)

    pipeline.set_state(Gst.State.PLAYING)
    logging.info('Receiving on port %d ... (Ctrl+C to stop)', args.port)

    def _stop(sig, frame):
        pipeline.set_state(Gst.State.NULL)
        loop.quit()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    loop.run()
    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    main()
