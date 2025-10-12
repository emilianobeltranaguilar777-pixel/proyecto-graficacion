#!/usr/bin/env bash
set -e
sudo systemctl restart droidcam.service
export QT_QPA_PLATFORM=xcb
python -m src.app_launcher_tk
