#!/usr/bin/env python3
"""
RTP/UDP Multicast video server — streams a USB webcam to a multicast group.
Any client on the LAN that joins the multicast group can receive the stream.

Usage:
  python3 rtsp_server.py [--device /dev/video5] [--width 640] [--height 480]
                         [--fps 30] [--port 5600]
                         [--multicast 224.1.1.1]

Client receive URL: --rtsp_url 224.1.1.1:5600
"""

import argparse
import logging
import signal
import subprocess
import sys


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--device',    default='/dev/video5')
    parser.add_argument('--width',     default=640,       type=int)
    parser.add_argument('--height',    default=480,       type=int)
    parser.add_argument('--fps',       default=30,        type=int)
    parser.add_argument('--port',      default=5600,      type=int)
    parser.add_argument('--multicast', default='224.1.1.1')
    args = parser.parse_args()

    pipeline = (
        f'gst-launch-1.0 -v '
        f'v4l2src device={args.device} io-mode=mmap ! '
        f'video/x-raw,width={args.width},height={args.height},'
        f'framerate={args.fps}/1,format=YUY2 ! '
        f'videoconvert ! '
        f'v4l2h264enc extra-controls="controls,repeat_sequence_header=1" ! '
        f'video/x-h264,profile=baseline ! '
        f'rtph264pay config-interval=1 pt=96 ! '
        f'udpsink host={args.multicast} port={args.port} '
        f'multicast-iface=eth0 sync=false auto-multicast=true'
    )

    logging.info('Multicast streaming to %s:%d', args.multicast, args.port)
    logging.info('Clients connect with --rtsp_url %s:%d', args.multicast, args.port)
    logging.info('Pipeline: %s', pipeline)

    proc = subprocess.Popen(pipeline, shell=True)

    def _stop(sig, frame):
        logging.info('Stopping...')
        proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    proc.wait()


if __name__ == '__main__':
    main()
