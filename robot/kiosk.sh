#!/bin/bash
# Full-screen browser on the Pi's screen showing the Yachachiq kiosk page.
export DISPLAY=:0
xset s off; xset -dpms; xset s noblank 2>/dev/null
unclutter -idle 1 -root &
until curl -s http://localhost:8877/api/state >/dev/null; do sleep 1; done
exec chromium-browser --kiosk --noerrdialogs --disable-infobars --disable-session-crashed-bubble \
     --check-for-update-interval=31536000 --ozone-platform=x11 http://localhost:8877
