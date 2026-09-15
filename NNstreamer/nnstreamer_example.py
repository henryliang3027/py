
#!/usr/bin/env python

import os
import sys
import re
import logging
import subprocess
import argparse

available_engine=['cpu', 'TBD', 'armnn']
available_framework=['tflite','neuronsdk']
available_app=['image_classification', 'object_detection', 'object_detection_yolov5','face_detection', 'pose_estimation', 'low_light_image_enhancement', 'monocular_depth_estimation']
toggle_flags = ['0', '1']


def argument_parser_init():
  if find_nnapi_delegate_library() != 'null':
    available_engine[1] = 'nnapi'
  if find_neuron_library() != 'null':
    available_engine[1] = 'stable_delegate'


  parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument(
      '--app',
      default=available_app[0],
      choices=available_app,
      help='Choose a demo app to run. Default: %s' % available_app[0])
  parser.add_argument(
      '--engine',
      default=available_engine[0],
      choices=available_engine,
      help='Choose a backends to inference. Default: %s' % available_engine[0])
  parser.add_argument(
      '--framework',
      default='neuronsdk',
      choices=available_framework,
      help='Choose a framework for inference. Default: %s' % available_framework[0])
  parser.add_argument(
      '--img',
      default=None,
      help='Input a image file path.\n' +
           'Example: /usr/bin/nnstreamer-demo/original.png\n'
           'Note: This paramater is dedicated to low light enhancement app\n')
  parser.add_argument(
      '--cam',
      default=5,
      type=int,
      help='Input a camera node id, ex: 130 .\n' + 
           'Use \'v4l2-ctl --list-devices\' query camera node id.\n' + 
           'Example:\n' + 
           '$ v4l2-ctl --list-devices \n' + 
           '  ... \n' +
           '   C922 Pro Stream Webcam (usb-11290000.xhci-1.2): \n' + 
           '   /dev/video130 \n' + 
           '   /dev/video131 \n' + 
           '  ... \n' +
           'Note: This paramater is for all the apps except low light enhancement app.' 
           '  \n')

  parser.add_argument(
      '--cam_type',
      default='websocket',
      choices=['uvc', 'yuvsensor', 'rawsensor', 'websocket', 'rtsp'],
      help='Choose correct type of camera being used for the demo, ex: yuvsensor\n' +
      'Note: This paramater is for all the apps except low light enhancement app.\n')

  parser.add_argument(
      '--ws_url',
      default='ws://192.168.50.108:8011/ws/camera',
      help='WebSocket URL for camera stream (used with --cam_type websocket)')

  parser.add_argument(
      '--rtsp_url',
      default='rtsp://127.0.0.1:8554/stream',
      help='RTSP URL for camera stream (used with --cam_type rtsp)')

  parser.add_argument(
      '--width',
      default=640,
      type=int,
      help='Input video display width, ex: 640')

  parser.add_argument(
      '--height',
      default=480,
      type=int,
      help='Input video display height, ex: 480')

  parser.add_argument(
      '--performance',
      default=0,
      choices=[0, 1],
      type=int,
      help='Enable to make CPU/GPU/APU run under performance mode, ex: 1')

  parser.add_argument(
        '--fullscreen',
        default='1',
        choices=toggle_flags,
        help='Fullscreen preview.\n'
             '1: Enable\n'
             '0: Disable\n'
             'Note: This paramater is for all the apps except low light enhancement app and monocular depth estimation app.\n')

  parser.add_argument(
        '--throughput',
        default='1',
        choices=toggle_flags,
        help='Print throughput information.\n'
             '1: Enable\n'
             '0: Disable\n')

  parser.add_argument(
      '--rot',
      default=0,
      type=int,
      help='Rotate the camera image by degrees, ex: 90\n'
            'Note: This paramater is for all the apps except low light enhancement app.\n')
  
  parser.add_argument(
      '--export', 
      type=str, 
      help='Path to export the enhanced image (e.g., /path/to/image.png)')

  args = parser.parse_args()

    # Conditionally require the cam_type argument based on the app argument
  if args.app != 'low_light_image_enhancement' and not args.cam_type:
      parser.error("--cam_type is required for the selected app")

  return args

def performance_mode():
  batcmd = 'for i in /sys/devices/system/cpu/cpufreq/policy*; do echo performance > ${i}/scaling_governor; done'
  subprocess.run(batcmd, shell=True, check=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
  batcmd = 'echo performance > /sys/devices/platform/soc/*.gpu/devfreq/*.gpu/governor'
  subprocess.run(batcmd, shell=True, check=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
  batcmd = 'if [ -f /sys/kernel/debug/apusys/power ]; then echo dvfs_debug 0 > /sys/kernel/debug/apusys/power; fi;'
  subprocess.run(batcmd, shell=True, check=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
  batcmd = 'echo disabled > /sys/class/thermal/thermal_zone0/mode'
  subprocess.run(batcmd, shell=True, check=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
  logging.info("Set performance mode")

def enable_performance(option):
  if option == 1:
    performance_mode()
  else:
    logging.info("Performance mode not set")

def find_armnn_delegate_library():
  cmd = 'ls -l /usr/lib/libarmnnDelegate.so.*'
  res = subprocess.run(cmd, shell=True, check=False, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
  result = res.stdout

  pattern = r"ls\: cannot access "
  miss = re.search(pattern, result)
  if miss:
      logging.warn("can't find libneuronusdk_runtime.mtk.so")
      return 'null'

  pattern = r"/usr/lib/libarmnnDelegate\.so\.\d+\.0"
  find = re.search(pattern, result)
  if find:
      logging.info(find.group())
      return find.group()
  else:
      logging.warn("can't find libarmnnDelegate.so")
      return 'null'

def find_nnapi_delegate_library():
  cmd = 'ls -l /usr/lib/nnapi_external_delegate.so'
  res = subprocess.run(cmd, shell=True, check=False, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
  result = res.stdout

  pattern = r"ls\: cannot access "
  miss = re.search(pattern, result)
  if miss:
      logging.debug("can't find nnapi_external_delegate.so")
      return 'null'

  pattern = r"nnapi_external_delegate.so"
  find = re.search(pattern, result)
  if find:
      logging.debug(find.group())
      return find.group()
  else:
      logging.debug("can't find nnapi_external_delegate.so")
      return 'null'

def find_neuron_library():
  cmd = 'ls -l /usr/lib/libneuronusdk_runtime.mtk.so'
  res = subprocess.run(cmd, shell=True, check=False, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
  result = res.stdout

  pattern = r"ls\: cannot access "
  miss = re.search(pattern, result)
  if miss:
      logging.debug("can't find libneuronusdk_runtime.mtk.so")
      return 'null'

  pattern = r"libneuronusdk_runtime.mtk.so"
  find = re.search(pattern, result)
  if find:
      logging.debug(find.group())
      return find.group()
  else:
      logging.debug("can't find libneuronusdk_runtime.mtk.so")
      return 'null'

def find_cpu_cores():
    cmd = 'echo $(grep -c processor /proc/cpuinfo)'
    res = subprocess.run(cmd, shell=True, check=False, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    result = res.stdout
    return result

# Query supported backend of this platform and use the first backend to compile tflite model to dla file later.
# We expect the first backend to be MDLA.
def dla_query_supported_backend():
    batcmd = ('ncc-tflite --arch=?')
    res = subprocess.run(batcmd, shell=True, check=False, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    result = res.stdout

    reg = re.compile("\n.*?- (.*)")
    match = re.findall(reg,result)
    if match:
        output_count = len(match)
    else:
        logging.error("FAIL to find backends")
        exit("")

    return match[0];


# Query input/output tensor information by ncc-tflite.
# Because we can't get input/output tensor information from dla file, so we have to 
# query these information when we compile tflite model to dla.
def dla_parse_tensor_info(content, prefix):
    compile_options = ' '

    # Find Type
    s = prefix + 'type=';
    matches = re.findall(r' Type:\s*(\S+)', content) 
    if matches:
        count = len(matches)
        for i in range(count):
            t = matches[i]
            if (t == 'kTfLiteFloat32'):
                s += 'float32,'
                compile_options += ' --relax-fp32 '
            elif (t == 'kTfLiteInt32'):
                s += 'int32,'
            elif (t == 'kTfLiteUInt8'):
                s += 'uint8,'
            elif (t == 'kTfLiteInt64'):
                s += 'float64,'
            else:
                logging.error("Unknown Type")
                return ""
    else: 
        logging.error("FAIL to find Type")
        return ""

    s = s.rstrip(',')
    logging.debug(s)

    # Find Shape
    s = s + ' ' + prefix + '=';
    matches = re.findall(r' Shape:\s*\{\s*([\d\s,]+)\s*\}', content)
    if matches:
        count = len(matches)
        for i in range(count):
            numbers = matches[i].split(',')
            for j in range(len(numbers)-1, -1, -1):
                s += numbers[j] + ':'

            s = s.rstrip(':')
            s += ','
    else: 
        logging.error("FAIL to find Shape")
        return ""

    s = s.rstrip(',')
    logging.debug(s)
    return s, compile_options

def dla_converter(tflite_file, dla_file):
    # Query supported backend on platform
    arch = dla_query_supported_backend()

    # Get input/output tensor information
    batcmd = ('ncc-tflite --arch=%s %s --show-io-info' % (arch, tflite_file))
    res = subprocess.run(batcmd, shell=True, check=False, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    result = res.stdout

    parts = result.partition("# of output tensors")
    input_part = parts[0]
    output_part = parts[1] + parts[2]

    # Parse and convert input/output tensor information to nnstreamer filter properties
    input_tensor_info_str, compile_options = dla_parse_tensor_info(input_part, 'input')
    output_tensor_info_str, compile_options = dla_parse_tensor_info(output_part, 'output')

    tensor_info_str =  input_tensor_info_str + ' ' + output_tensor_info_str


    # Compile tflite to dla file
    batcmd = ('ncc-tflite --arch=%s %s -o %s %s' % (arch, tflite_file, dla_file, compile_options))
    res = subprocess.run(batcmd, shell=True, check=False, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    result = res.stdout

    return (tensor_info_str)
