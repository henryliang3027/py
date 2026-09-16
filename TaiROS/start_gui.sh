#!/bin/bash
set -e
sudo ip addr flush dev end0
sudo ip addr add 192.168.60.2/24 dev  end0
sudo ip link set end0 up
python3 /demo/gui_ws_nnstreamer_cards_demo.py --ws_url ws://192.168.60.1:8011/ws/camera
