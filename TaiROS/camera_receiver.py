import asyncio
import time

import cv2
import numpy as np
import websockets


WS_URL = "ws://192.168.50.108:8011/ws/camera"


async def main():
    print(f"Connecting to {WS_URL}")

    async with websockets.connect(
        WS_URL,
        max_size=None,
        ping_interval=20,
        ping_timeout=20,
    ) as ws:
        print("Connected. Measuring FPS (Ctrl+C to stop).")

        frame_count = 0
        t0 = time.monotonic()

        async for message in ws:
            if not isinstance(message, bytes):
                continue

            frame_count += 1
            elapsed = time.monotonic() - t0
            if elapsed >= 2.0:
                fps = frame_count / elapsed
                size_kb = len(message) / 1024
                arr = np.frombuffer(message, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    h, w = img.shape[:2]
                    print(f"FPS: {fps:.1f}  frame size: {size_kb:.1f} KB  resolution: {w}x{h}")
                else:
                    print(f"FPS: {fps:.1f}  frame size: {size_kb:.1f} KB  resolution: unknown")
                frame_count = 0
                t0 = time.monotonic()


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
except Exception as exc:
    print(f"Connection failed: {type(exc).__name__}: {exc}")
