import socket
import threading
import time
import pyautogui

TCP_PORT = 9999
DISCOVERY_PORT = 9998
DISCOVERY_MAGIC = b"ASL_RECEIVER_HERE"

pyautogui.PAUSE = 0.02          # tiny gap between keystrokes
pyautogui.FAILSAFE = True       # slam the mouse into a corner to abort


def get_my_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def broadcaster():
    """Announce 'I'm here' every second so the camera Mac can auto-find us."""
    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    u.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    while True:
        try:
            u.sendto(DISCOVERY_MAGIC, ("255.255.255.255", DISCOVERY_PORT))
        except OSError:
            pass
        time.sleep(1)


def type_token(token):
    if token == "SPACE":
        pyautogui.press("space")
    elif token == "BACK":
        pyautogui.press("backspace")
    elif len(token) == 1:
        pyautogui.write(token)
    # anything else (e.g. MODE) is ignored


def main():
    ip = get_my_ip()
    print("=" * 54)
    print("  RECEIVER READY")
    print("  This Mac's IP:", ip)
    print("  On the camera Mac, just run:   python3 asl_app.py")
    print("  (it auto-connects). If it can't find me, run there:")
    print(f"      python3 asl_app.py {ip}")
    print("  >> Click into TextEdit now so the letters land there. <<")
    print("=" * 54)

    threading.Thread(target=broadcaster, daemon=True).start()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", TCP_PORT))
    s.listen(1)

    while True:
        conn, addr = s.accept()
        print("Connected to camera Mac:", addr)
        buf = ""
        with conn:
            while True:
                data = conn.recv(1024)
                if not data:
                    print("Camera Mac disconnected. Waiting for reconnect...")
                    break
                buf += data.decode("utf-8", errors="ignore")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    token = line.strip()
                    if token:
                        print("typing:", token)
                        type_token(token)


if __name__ == "__main__":
    main()
