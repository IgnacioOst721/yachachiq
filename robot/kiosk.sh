#!/bin/bash
# Full-screen browser on the Pi's screen showing the Yachachiq kiosk page.
# Works on both desktops Raspberry Pi OS ships: labwc/Wayland (current) and X11 (older).
if [ -n "$WAYLAND_DISPLAY" ]; then
  PLATFORM=wayland
else
  PLATFORM=x11; export DISPLAY=${DISPLAY:-:0}
  xset s off 2>/dev/null; xset -dpms 2>/dev/null; xset s noblank 2>/dev/null
  command -v unclutter >/dev/null && unclutter -idle 1 -root &
fi
until curl -s http://localhost:8877/api/state >/dev/null; do sleep 1; done
BROWSER=$(command -v chromium || command -v chromium-browser)
exec "$BROWSER" --kiosk --noerrdialogs --disable-infobars --disable-session-crashed-bubble \
     --check-for-update-interval=31536000 --touch-events=enabled --start-maximized \
     --ozone-platform=$PLATFORM http://localhost:8877
