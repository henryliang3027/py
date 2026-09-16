#!/usr/bin/env python

import os
import sys
import gi
import re
import logging
import argparse
import subprocess

import numpy as np
import cairo

############################################################################
#
#  RAW-HEAD YOLOv8 demo (FLOAT32 model) for MTK / nnstreamer.
#
#  Model = raw-head YOLOv8 from export_yolov8_rawhead.py -> mtk_pytorch_converter
#  (UNQUANTIZED / float32 this time):
#    INPUT  : float32, NCHW [1,3,640,640], expects normalised [0,1] (i.e. /255)
#    OUTPUT : 3 raw heads float32 [1, 64+nc, {80,40,20}^2] @ strides 8/16/32
#             channel layout per head = [box: 4*reg_max=64][class: nc]
#
#  As with the yolov5 raw-head demo, we CANNOT use tensor_filter
#  framework=python3 (embedded interpreter fights this Python host over the
#  GIL). We pull the raw heads into THIS interpreter via tensor_sink, decode +
#  NMS in numpy (yolov8_postprocess.py), and draw boxes with cairooverlay.
#
#  Differences from the int8 yolov5 raw-head demo:
#    * input transform is typecast:float32,div:255.0 (NOT int8/add:-128),
#      because the float model expects normalised [0,1] input.
#    * tensor_sink buffers are read as float32 (NOT int8).
#    * decoder is yolov8_postprocess (DFL + anchor-free, no objectness).
#
############################################################################

gi.require_version('Gst', '1.0')
gi.require_version('GstGL', '1.0')
gi.require_version('Pango', '1.0')
gi.require_version('PangoCairo', '1.0')

from gi.repository import Gst, GstGL, GLib, Pango, PangoCairo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nnstreamer_example import *
import yolov8_postprocess as pp


class Demo:
  def __init__(self, argv=None):
    self.loop = None
    self.pipeline = None
    self.running = False

    self.VIDEO_WIDTH = 720
    self.VIDEO_HEIGHT = 1280

    self.MODEL_INPUT_HEIGHT = 640
    self.MODEL_INPUT_WIDTH = 640

    self.FULLSCREEN = 0
    self.CAM_ROT = 0
    self.CAM_ID = 0

    self.tflite_model = ''
    self.dla = ''
    self.tflite_label = ''

    self.filter = None
    self.textoverlay = None
    self.invoke_ms = 0

    # Detection state shared between the tensor_sink callback (streaming thread)
    # and the cairooverlay draw callback. Reference swap is atomic under the GIL.
    self.detected = []
    self.labels = []
    self.overlay_valid = False
    self.conf_thres = 0.60
    self.iou_thres = 0.45

    self.rtsp_url = 'rtsp://127.0.0.1:8554/stream'

    if not self.tflite_init():
        raise Exception

    Gst.init(argv)

  def _src(self):
    if self.cam_type == 'uvc':
      return (f'v4l2src name=src device=/dev/video{self.CAM_ID} io-mode=mmap ! '
              f'video/x-raw,width={self.VIDEO_WIDTH},height={self.VIDEO_HEIGHT},format=YUY2 ! tee name=t_raw ')
    elif self.cam_type == 'yuvsensor':
      return (f'v4l2src name=src device=/dev/video{self.CAM_ID} ! '
              f'video/x-raw,width=1920,height=1080,format=UYVY ! tee name=t_raw ')
    elif self.cam_type == 'yuvsensor_d9':
      return (f'v4l2src name=src device=/dev/video{self.CAM_ID} ! '
              f'video/x-raw,width=1920,height=1080,format=YUY2 ! tee name=t_raw ')
    elif self.cam_type == 'rawsensor':
      return (f'v4l2src name=src device=/dev/video{self.CAM_ID} ! '
              f'video/x-raw,width=2048,height=1536,format=YUY2 ! tee name=t_raw ')
    elif self.cam_type == 'rtsp':
      addr, port = self.rtsp_url.rsplit(':', 1)
      return (f'udpsrc address={addr} port={port} '
              f'multicast-iface=eth0 auto-multicast=true '
              f'caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! '
              f'rtph264depay ! h264parse ! avdec_h264 ! '
              f'videoconvert ! video/x-raw,format=BGRx ! tee name=t_raw ')
    return ''

  def build_pipeline(self):
    cmd = self._src()

    # ---------------- Display branch: draw boxes via cairooverlay -------------
    cmd += f't_raw. ! queue leaky=2 max-size-buffers=10 ! '
    if self.cam_type in ('uvc', 'rtsp'):
      cmd += f'videoconvert ! video/x-raw,format=BGRx ! '
    else:
      cmd += (f'v4l2convert output-io-mode=dmabuf-import extra-controls="cid,rotate={self.CAM_ROT}" ! '
              f'video/x-raw,width={self.VIDEO_WIDTH},height={self.VIDEO_HEIGHT},format=ARGB,pixel-aspect-ratio=1/1 ! ')
    cmd += f'cairooverlay name=res ! videoconvert ! '
    if self.THROUGHPUT == '1':
      cmd += f'textoverlay name=info text="" font-desc=Sans,18 valignment=position halignment=position ypos=0.03 ! '
    if self.cam_type in ('uvc', 'rtsp'):
      cmd += (f'fpsdisplaysink name=sink text-overlay=false signal-fps-measurements=true sync=false '
              f'video-sink="waylandsink sync=false qos=false fullscreen={self.FULLSCREEN}" ')
    else:
      cmd += (f'fpsdisplaysink name=sink text-overlay=false signal-fps-measurements=true sync=false '
              f'video-sink="waylandsink sync=false qos=false" ')

    # ---------------- Inference branch: raw heads -> tensor_sink --------------
    cmd += f't_raw. ! queue leaky=2 max-size-buffers=2 ! '
    if self.cam_type in ('uvc', 'rtsp'):
      cmd += (f'videoconvert ! videoscale ! '
              f'video/x-raw,width={self.MODEL_INPUT_WIDTH},height={self.MODEL_INPUT_HEIGHT},format=RGB ! ')
    else:
      cmd += (f'v4l2convert output-io-mode=dmabuf-import capture-io-mode=mmap extra-controls="cid,rotate={self.CAM_ROT}" ! '
              f'video/x-raw,width={self.MODEL_INPUT_WIDTH},height={self.MODEL_INPUT_HEIGHT},format=RGB,pixel-aspect-ratio=1/1 ! ')
    cmd += f'tensor_converter ! '

    # Input reformat for the FLOAT model:
    #   transpose 1:2:0:3         : interleaved RGB [3,W,H,1] -> planar NCHW [W,H,3,1]
    #   typecast:float32,div:255. : uint8 [0,255] -> float32 [0,1] (model expects
    #                               normalised input; the /255 is NOT in the model)
    cmd += f'tensor_transform mode=transpose option=1:2:0:3 ! '
    cmd += f'tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! '

    if self.framework == 'neuronsdk':
      # VERIFY against your compiled .dla. For a FLOAT .dla, types are float32.
      cmd += (f'tensor_filter framework=neuronsdk throughput={self.THROUGHPUT} name=nn '
              f'model={self.dla} inputtype=float32 input=640:640:3:1 '
              f'outputtype=float32,float32,float32 output=80:80:96:1,40:40:96:1,20:20:96:1 !')
    elif self.framework == 'tflite':
      if self.engine == 'cpu':
        cpu_cores = find_cpu_cores()
        cmd += f'tensor_filter framework=tensorflow-lite throughput={self.THROUGHPUT} name=nn model={self.tflite_model} custom=NumThreads:{cpu_cores} ! '
      elif self.engine == 'armnn':
        library = find_armnn_delegate_library()
        cmd += f'tensor_filter framework=tensorflow-lite throughput={self.THROUGHPUT} name=nn model={self.tflite_model} custom=Delegate:External,ExtDelegateLib:{library},ExtDelegateKeyVal:backends#GpuAcc ! '
      elif self.engine == 'stable_delegate':
        cmd += f'tensor_filter framework=tensorflow-lite throughput={self.THROUGHPUT} name=nn model={self.tflite_model} custom=Delegate:Stable,StaDelegateSettingFile:/usr/share/label_image/stable_delegate_settings.json,ExtDelegateKeyVal:backends#GpuAcc ! '
      elif self.engine == 'nnapi':
        logging.error('Not support NNAPI')

    cmd += f'tensor_sink name=res_sink '

    self.pipeline = Gst.parse_launch(cmd)
    logging.info("pipeline: %s" % cmd)

  # ---- tensor_sink: raw float32 heads arrive here; decode in this interpreter
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
    if arrays:
      try:
        self.detected = pp.detect(arrays, self.conf_thres, self.iou_thres)
      except Exception as e:
        logging.warning('[decode] %s', e)

  # ---- cairooverlay: draw the latest detections ------------------------------
  def on_caps_changed(self, overlay, caps):
    self.overlay_valid = True

  def on_draw(self, overlay, context, timestamp, duration):
    if not self.overlay_valid:
      return
    dets = self.detected
    if not dets:
      return
    sx = self.VIDEO_WIDTH / float(self.MODEL_INPUT_WIDTH)
    sy = self.VIDEO_HEIGHT / float(self.MODEL_INPUT_HEIGHT)
    pango_layout = PangoCairo.create_layout(context)
    font_desc = Pango.FontDescription('Noto Sans CJK TC Bold 12')
    pango_layout.set_font_description(font_desc)
    for (x1, y1, x2, y2, cls, score) in dets:
      rx, ry = x1 * sx, y1 * sy
      rw, rh = (x2 - x1) * sx, (y2 - y1) * sy
      context.set_source_rgb(1.0, 0.0, 0.0)
      context.set_line_width(2)
      context.rectangle(rx, ry, rw, rh)
      context.stroke()
      name = self.labels[cls] if 0 <= cls < len(self.labels) else str(cls)
      pango_layout.set_text('%s %.2f' % (name, score), -1)
      tx, ty = rx + 2, max(ry - 4, 14)
      context.move_to(tx, ty)
      context.set_source_rgb(0.0, 0.0, 0.0)
      context.set_line_width(3)
      PangoCairo.layout_path(context, pango_layout)
      context.stroke()
      context.move_to(tx, ty)
      context.set_source_rgb(1.0, 1.0, 0.0)
      PangoCairo.show_layout(context, pango_layout)

  def on_buffer(self, pad, info):
      throughput = self.filter.get_property('throughput')
      if (throughput > 0):
        fps = (throughput/1000.0);
        self.invoke_ms = (1.0/fps) * 1000.0;
      return Gst.PadProbeReturn.OK

  def on_fps_measurement(self, element, fps, droprate, avgfps):
      new_text = f'Camera FPS: {avgfps:.2f}, Invoke Time(ms):{round(self.invoke_ms, 2)}'
      self.textoverlay.set_property('text', new_text)

  def run(self):
      logging.info("Run: YOLOv8 object detection (raw-head, float).")

      self.loop = GLib.MainLoop()

      bus = self.pipeline.get_bus()
      bus.add_signal_watch()
      bus.connect('message', self.on_bus_message)

      res_sink = self.pipeline.get_by_name('res_sink')
      res_sink.connect('new-data', self.on_new_data)

      overlay = self.pipeline.get_by_name('res')
      overlay.connect('draw', self.on_draw)
      overlay.connect('caps-changed', self.on_caps_changed)

      if self.THROUGHPUT == '1':
          self.filter = self.pipeline.get_by_name("nn")
          srcpad = self.filter.get_static_pad("src")
          srcpad.add_probe(Gst.PadProbeType.BUFFER, self.on_buffer)
          self.textoverlay = self.pipeline.get_by_name('info')
          sink = self.pipeline.get_by_name('sink')
          sink.connect('fps-measurements', self.on_fps_measurement)

      self.pipeline.set_state(Gst.State.PLAYING)
      self.running = True
      self.loop.run()

      self.running = False
      self.pipeline.set_state(Gst.State.NULL)
      bus.remove_signal_watch()

  def tflite_init(self):
      tflite_model = 'yolov8s_mixed.tflite'   # rename your converted model to this
      dla = 'yolov8s_mixed.dla'
      tflite_label = 'mixed_label.txt'

      current_folder = os.path.dirname(os.path.abspath(__file__))
      model_folder = os.path.join(current_folder, '')

      self.tflite_model = os.path.join(model_folder, tflite_model)
      self.dla = os.path.join(model_folder, dla)
      if not os.path.exists(self.tflite_model):
          logging.error('cannot find tflite model [%s]', self.tflite_model)
          return False

      self.tflite_label = os.path.join(model_folder, tflite_label)
      if not os.path.exists(self.tflite_label):
          logging.error('cannot find label [%s]', self.tflite_label)
          return False

      with open(self.tflite_label, 'r', encoding='utf-8') as f:
          self.labels = [line.strip() for line in f if line.strip() != '']
      logging.info('loaded %d labels', len(self.labels))
      if len(self.labels) != pp.NC:
          logging.warning('label count %d != NC %d in yolov8_postprocess',
                          len(self.labels), pp.NC)
      return True

  def on_bus_message(self, bus, message):
      if message.type == Gst.MessageType.EOS:
          logging.info('received eos message')
          self.loop.quit()
      elif message.type == Gst.MessageType.ERROR:
          error, debug = message.parse_error()
          logging.warning('[error] %s : %s', error.message, debug)
          self.loop.quit()
      elif message.type == Gst.MessageType.WARNING:
          error, debug = message.parse_warning()
          logging.warning('[warning] %s : %s', error.message, debug)
      elif message.type == Gst.MessageType.STREAM_START:
          logging.info('received start message')
      elif message.type == Gst.MessageType.QOS:
          data_format, processed, dropped = message.parse_qos_stats()
          format_str = Gst.Format.get_name(data_format)
          logging.info('[qos] format[%s] processed[%d] dropped[%d]', format_str, processed, dropped)

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  args = argument_parser_init()

  example = Demo(sys.argv[1:])
  example.CAM_ID = args.cam
  example.FULLSCREEN = args.fullscreen
  example.THROUGHPUT = args.throughput
  example.CAM_ROT = args.rot
  example.cam_type = args.cam_type
  example.framework = args.framework
  example.engine = args.engine
  example.rtsp_url = args.rtsp_url

  if example.cam_type == 'uvc':
    example.VIDEO_WIDTH = 640
    example.VIDEO_HEIGHT = 480
  elif example.cam_type == 'rtsp':
    example.VIDEO_WIDTH = 1280
    example.VIDEO_HEIGHT = 720
  else:
    example.VIDEO_WIDTH = 1920
    example.VIDEO_HEIGHT = 1080

  example.VIDEO_WIDTH = args.width
  example.VIDEO_HEIGHT = args.height

  enable_performance(args.performance)

  example.build_pipeline()
  example.run()
