#!/bin/bash
# Configure the Mac's USB Ethernet adapter for the direct cable to the Pi (192.168.7.1 <-> 192.168.7.2).
# Run once with the adapter plugged in:  bash cable_mac.sh
set -e
PORT=$(networksetup -listallhardwareports | awk -F': ' '/Hardware Port: .*(USB|Ethernet|LAN)/{p=$2} /Device:/{d=$2} p&&d{print p; exit}')
[ -n "$PORT" ] || { echo "No veo ningún adaptador Ethernet. Conéctalo a la Mac y vuelve a correr esto."; exit 1; }
echo "Adaptador: $PORT"
networksetup -setmanual "$PORT" 192.168.7.1 255.255.255.0
echo "Listo: $PORT = 192.168.7.1. La Pi por cable es 192.168.7.2 (ssh admin@192.168.7.2)."
echo "Para volver a automático (DHCP): networksetup -setdhcp \"$PORT\""
