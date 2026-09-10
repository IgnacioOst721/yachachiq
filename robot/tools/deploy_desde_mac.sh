#!/bin/bash
# Update the robot WITHOUT internet: pack the dev branch into a git bundle, send it to the Pi over
# the local network (phone hotspot or cable) and run deploy.sh there.
#   bash robot/tools/deploy_desde_mac.sh                 # finds the Pi by name or by scanning the network
#   bash robot/tools/deploy_desde_mac.sh 192.168.43.17   # or give its address
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PI="$1"
if [ -z "$PI" ]; then
  for h in definitelyawesome.local 192.168.7.2; do
    if nc -z -G 2 "$h" 22 2>/dev/null; then PI=$h; break; fi
  done
fi
if [ -z "$PI" ]; then
  # no mDNS on this network: look for the one machine that answers on the robot's web port
  ME=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)
  BASE=${ME%.*}
  echo "buscando la Pi en $BASE.0/24 (puerto 8877)..."
  for i in $(seq 1 254); do
    (nc -z -G 1 "$BASE.$i" 8877 2>/dev/null && echo "$BASE.$i") &
  done | head -1 > /tmp/pi_addr.txt; wait
  PI=$(cat /tmp/pi_addr.txt)
fi
[ -n "$PI" ] || { echo "No encuentro la Pi. ¿Está prendida y en la misma red? Pásame su IP: bash $0 <ip>"; exit 1; }
echo "== Pi en $PI"
git -C "$REPO" bundle create -q /tmp/yachachiq.bundle dev
scp -q /tmp/yachachiq.bundle "admin@$PI:/tmp/yachachiq.bundle"
ssh "admin@$PI" 'bash ~/yachachiq/robot/deploy.sh /tmp/yachachiq.bundle'
