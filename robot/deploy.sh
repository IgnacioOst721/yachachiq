#!/bin/bash
# Update the robot on the Pi in one go:  ssh admin@definitelyawesome.local 'bash ~/yachachiq/robot/deploy.sh'
# Pulls the dev branch, reinstalls the two services and restarts them. The kiosk page reloads itself.
set -e
cd "$(dirname "$0")"; ROBOT=$(pwd); REPO=$(dirname "$ROBOT")
echo "== código"; git -C "$REPO" pull -q --ff-only origin dev && echo "   $(git -C "$REPO" log --oneline -1)"
echo "== dependencias (solo si cambió requirements.txt)"
if [ requirements.txt -nt venv/.deps-ok ] 2>/dev/null || [ ! -f venv/.deps-ok ]; then
  ./venv/bin/pip install -q -r requirements.txt && touch venv/.deps-ok
fi
echo "== servicios"
for s in yachachiq yachachiq-lsp; do
  sed "s|/home/pi/yachachiq|$REPO|g; s|User=pi|User=$USER|" $s.service | sudo tee /etc/systemd/system/$s.service >/dev/null
done
sudo systemctl daemon-reload
sudo systemctl restart yachachiq yachachiq-lsp
sleep 6
echo "   $(systemctl is-active yachachiq yachachiq-lsp | tr '\n' ' ')"
echo "== estado"; curl -s localhost:8877/api/state | python3 -c "import sys,json; d=json.load(sys.stdin); print('   ', d['state'], '|', {k:v for k,v in d['modes'].items() if k in ('plotter','audio','stt','photo','publish')})"
