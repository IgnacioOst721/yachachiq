"""A pretend GRBL 1.1 board on a pseudo-terminal, so the plotter driver can be exercised
without an Arduino: answers 'ok' per line, '?' with <Idle|...> / <Run|...>, $X, Ctrl-X.
Motion lines take a little 'time' so wait_idle() really has something to wait for."""
import os, pty, re, sys, threading, time

master, slave = pty.openpty()
import tty; tty.setraw(slave)          # no echo: only what the driver sends reaches us
link = sys.argv[1]
try: os.unlink(link)
except FileNotFoundError: pass
os.symlink(os.ttyname(slave), link)
print("fake GRBL on", os.ttyname(slave), "->", link, flush=True)

busy_until = 0.0
alarm = True                          # GRBL powers up in alarm when homing is enabled
hold = False
pos = [0.0, 0.0, 0.0]
received, buf = [], b""
def w(s): os.write(master, s.encode())
w("\r\nGrbl 1.1h ['$' for help]\r\n")
while True:
    try:
        data = os.read(master, 256)
    except OSError:
        break
    for ch in data:
        b = bytes([ch])
        if b == b"?":
            if hold: state = "Hold:0"
            elif time.time() < busy_until: state = "Run"
            else: state = "Alarm" if alarm else "Idle"
            w(f"<{state}|MPos:{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}|FS:0,0>\r\n"); continue
        if b == b"!":
            hold = True; busy_until = 0; continue
        if b == b"~":
            hold = False; continue
        if b == b"\x18":
            # reset while moving loses position (ALARM:3); reset in hold keeps it
            moving = (not hold) and time.time() < busy_until
            busy_until = 0; hold = False; buf = b""
            w("\r\nGrbl 1.1h ['$' for help]\r\n")
            if moving:
                alarm = True; w("ALARM:3\r\n"); pos[:] = [0, 0, 0]
            continue
        buf += b
        if b in (b"\n", b"\r"):
            line = buf.decode(errors="ignore").strip(); buf = b""
            if not line: continue
            received.append(line)
            if line == "$X":
                alarm = False; w("[MSG:Caution: Unlocked]\r\nok\r\n"); continue
            if alarm and re.match(r"^G[01]", line):
                w("error:9\r\n"); continue
            if re.match(r"^G[01]\b", line):
                # like a real GRBL: the planner buffer is small, so 'ok' for a move only comes
                # once there is room, i.e. streaming is paced by the machine (~30 ms per move here)
                time.sleep(0.03)
                busy_until = max(busy_until, time.time()) + 0.05
                for ax, i in (("X", 0), ("Y", 1), ("Z", 2)):
                    m = re.search(ax + r"(-?[\d.]+)", line)
                    if m: pos[i] = float(m.group(1))
            w("ok\r\n")
            with open(link + ".log", "a") as f: f.write(line + "\n")
