#!/bin/bash
# One-shot install on Raspberry Pi OS 64-bit (Bookworm) with desktop.
#   cd ~/yachachiq/robot && bash setup_pi.sh
set -e
cd "$(dirname "$0")"
ROBOT=$(pwd); REPO=$(dirname "$ROBOT")

echo "== 1/6 paquetes del sistema"
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev portaudio19-dev libatlas-base-dev \
     chromium-browser unclutter git alsa-utils libgl1 libglib2.0-0

echo "== 2/6 entorno Python"
[ -d venv ] || python3 -m venv venv
./venv/bin/pip install --upgrade pip wheel
./venv/bin/pip install -r requirements.txt

echo "== 3/6 modelo de voz (Whisper)"
./venv/bin/python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8'); print('whisper base listo')"

echo "== 4/6 prueba del sistema (sin hardware)"
./venv/bin/python selftest.py

echo "== 5/6 servicios (arranque automatico)"
sed "s|/home/pi/yachachiq|$REPO|g; s|User=pi|User=$USER|" yachachiq.service | sudo tee /etc/systemd/system/yachachiq.service >/dev/null
sed "s|/home/pi/yachachiq|$REPO|g; s|User=pi|User=$USER|" yachachiq-lsp.service | sudo tee /etc/systemd/system/yachachiq-lsp.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now yachachiq
sudo systemctl enable --now yachachiq-lsp
sudo usermod -aG dialout,video,audio "$USER" || true

echo "== 6/6 kiosko en pantalla completa al iniciar sesion"
chmod +x kiosk.sh
mkdir -p ~/.config/autostart
cat > ~/.config/autostart/yachachiq-kiosk.desktop <<DESK
[Desktop Entry]
Type=Application
Name=Yachachiq kiosk
Exec=$ROBOT/kiosk.sh
X-GNOME-Autostart-enabled=true
DESK

echo
echo "Listo. Reinicia la Pi: sudo reboot"
echo "Camaras por id:  ls /dev/v4l/by-id/     -> ponlas en los .service (PHOTO_CAMERA / ASL_CAMERA)"
echo "Logs:            journalctl -u yachachiq -f   |   journalctl -u yachachiq-lsp -f"
echo "Red con la laptop (cable):  sudo nmcli con add type ethernet ifname eth0 con-name lan ip4 192.168.7.2/24 && sudo nmcli con up lan"
