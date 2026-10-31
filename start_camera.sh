#!/bin/bash
pkill -9 -f ip_camera_stream.py 2>/dev/null
sleep 1
cd "$HOME"
setsid nohup python3 -u ip_camera_stream.py < /dev/null > /tmp/stream.log 2>&1 &
