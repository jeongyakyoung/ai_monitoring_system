## Rasberry Pi5 Pyqt5 install
# sudo apt update
# sudo apt install python3-pyqt5
# python3 -m venv --system-site-packages myenv
# pip install -r requirements.txt

## Rasberry Pi5 Pin reset
# sudo apt remove python3-rpi.gpio
# sudo apt install python3-rpi-lgpio

## setup one click file
# nano "~/Desktop/AI Monitoring System.desktop"
# [Desktop Entry]
# Name=AI Monitoring System
# Exec=/home/pi/ai_monitoring_system_env/bin/python /home/pi/ai_monitoring_system/main.py
# Icon=/home/pi/ai_monitoring_system/icon.png
# Terminal=false
# Type=Application
# Categories=Utility;
## close
# chmod +x ~/Desktop/AI\ Monitoring\ System.desktop


