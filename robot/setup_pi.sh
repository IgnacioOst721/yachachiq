#!/bin/bash
# One-shot install on Raspberry Pi OS 64-bit (Bookworm or Trixie) with desktop.
#   cd ~/yachachiq/robot && bash setup_pi.sh
set -e
cd "$(dirname "$0")"
ROBOT=$(pwd); REPO=$(dirname "$ROBOT")

echo "== 1/7 paquetes del sistema"
sudo apt-get update
# package names differ between Debian 12 and 13; install whichever exists
pkgs="python3-venv python3-dev portaudio19-dev unclutter git alsa-utils libgl1 curl fonts-noto-color-emoji"
for alt in "libopenblas0 libatlas-base-dev" "chromium chromium-browser" "libglib2.0-0t64 libglib2.0-0"; do
  for p in $alt; do apt-cache show "$p" >/dev/null 2>&1 && { pkgs="$pkgs $p"; break; }; done
done
sudo apt-get install -y $pkgs

echo "== 2/7 entorno Python"
# MediaPipe 0.10 (the version the sign model was trained with) needs Python <= 3.12.
# Debian 13 ships 3.13, so fetch a standalone 3.12 build when the system one is too new.
PY=python3
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] <= (3, 12) else 1)'; then
  PY=$HOME/.local/pythons/python/bin/python3
  if [ ! -x "$PY" ]; then
    echo "   python3 es $(python3 -V | cut -d" " -f2): instalando Python 3.12 en ~/.local/pythons"
    URL=$(curl -s https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest \
          | grep -o 'https://[^"]*cpython-3\.12[^"]*aarch64-unknown-linux-gnu-install_only\.tar\.gz' | head -1)
    mkdir -p "$HOME/.local/pythons" && curl -sL "$URL" | tar xz -C "$HOME/.local/pythons"
  fi
fi
echo "   usando $($PY -V) ($PY)"
[ -d venv ] || "$PY" -m venv venv
./venv/bin/pip install --upgrade pip wheel
./venv/bin/pip install -r requirements.txt

echo "== 3/7 modelo de voz (Whisper)"
./venv/bin/python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8'); print('whisper base listo')"

echo "== 4/7 prueba del sistema (sin hardware)"
./venv/bin/python selftest.py

echo "== 5/7 servicios (arranque automatico)"
sed "s|/home/pi/yachachiq|$REPO|g; s|User=pi|User=$USER|" yachachiq.service | sudo tee /etc/systemd/system/yachachiq.service >/dev/null
sed "s|/home/pi/yachachiq|$REPO|g; s|User=pi|User=$USER|" yachachiq-lsp.service | sudo tee /etc/systemd/system/yachachiq-lsp.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now yachachiq
sudo systemctl enable --now yachachiq-lsp
sudo usermod -aG dialout,video,audio "$USER" || true

echo "== 6/7 kiosko en pantalla completa al iniciar sesion"
chmod +x kiosk.sh
# Chromium policy: no translate bubble, sign-in or default-browser prompts on the kiosk screen
sudo mkdir -p /etc/chromium/policies/managed
printf '{\n  "TranslateEnabled": false,\n  "DefaultBrowserSettingEnabled": false,\n  "PasswordManagerEnabled": false,\n  "BrowserSignin": 0,\n  "MetricsReportingEnabled": false\n}\n' \
  | sudo tee /etc/chromium/policies/managed/yachachiq-kiosk.json >/dev/null
# the Touch Display 2 is portrait (720x1280); show the kiosk landscape (rotation set by the user's labwc/kanshi config)
mkdir -p ~/.config/autostart
cat > ~/.config/autostart/yachachiq-kiosk.desktop <<DESK
[Desktop Entry]
Type=Application
Name=Yachachiq kiosk
Exec=$ROBOT/kiosk.sh
X-GNOME-Autostart-enabled=true
DESK

echo "== 7/7 archivo web (clon de main para publicar historias)"
ARCHIVO=$HOME/yachachiq-archivo
if [ ! -d "$ARCHIVO/.git" ]; then
  git clone -q --branch main --single-branch https://github.com/IgnacioOst721/yachachiq.git "$ARCHIVO"
fi
git -C "$ARCHIVO" remote set-url --push origin git@github.com:IgnacioOst721/yachachiq.git
git -C "$ARCHIVO" config user.name "Yachachiq (Raspberry Pi)"
git -C "$ARCHIVO" config user.email "yachachiq@fdr.edu.pe"
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -q -t ed25519 -N "" -C "yachachiq-pi-publish" -f ~/.ssh/id_ed25519
ssh-keyscan -t ed25519 github.com 2>/dev/null >> ~/.ssh/known_hosts
echo "   Para que la Pi pueda publicar, agrega esta llave como Deploy Key (con escritura) en GitHub:"
echo "   $(cat ~/.ssh/id_ed25519.pub)"

echo
echo "Listo. Reinicia la Pi: sudo reboot"
echo "Camaras por id:  ls /dev/v4l/by-id/     -> ponlas en los .service (PHOTO_CAMERA / ASL_CAMERA)"
echo "Logs:            journalctl -u yachachiq -f   |   journalctl -u yachachiq-lsp -f"
echo "Red con la laptop (cable):  sudo nmcli con add type ethernet ifname eth0 con-name lan ip4 192.168.7.2/24 && sudo nmcli con up lan"
