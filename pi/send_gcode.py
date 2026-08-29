# GRBL version with servo pen control
import serial, time, sys

# ── Servo pen settings (tune these two numbers for your servo) ──
PEN_UP_CMD   = "M3 S30"    # servo angle for pen UP
PEN_DOWN_CMD = "M3 S90"    # servo angle for pen DOWN
PEN_DELAY    = 0.3         # seconds to wait after pen move

def _wait_for_ok(ser, timeout=120):
    """GRBL replies 'ok' or 'error:N' after each command."""
    start = time.time()
    while time.time() - start < timeout:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue
        if line == "ok":
            return True
        if line.startswith("error"):
            print("   !! GRBL error:", line)
            return False
        print("   <", line)   # status messages, alarms, etc.
    print("   !! timed out waiting for 'ok'")
    return False

def _send_line(ser, line):
    ser.write((line + "\n").encode())
    print("TX:", line)
    return _wait_for_ok(ser)

def send(gcode_file="output.gcode", port="/dev/ttyACM0", baud=115200):
    print("Connecting to GRBL on", port)
    ser = serial.Serial(port, baud, timeout=1)

    # GRBL wake-up: send newlines and wait for boot
    ser.write(b"\r\n\r\n")
    time.sleep(2)
    ser.reset_input_buffer()

    # Since machine is positioned manually, set current position as origin
    _send_line(ser, "G92 X0 Y0")     # zero work coordinates here
    _send_line(ser, "G21")           # millimeters
    _send_line(ser, "G90")           # absolute positioning

    # Start with pen up
    _send_line(ser, PEN_UP_CMD)
    time.sleep(PEN_DELAY)

    pen_is_down = False

    with open(gcode_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(";") or line.startswith("("):
                continue

            upper = line.upper()

            # Translate pen commands to servo commands
            if upper.startswith("M3") or "PEN DOWN" in upper:
                if not pen_is_down:
                    _send_line(ser, PEN_DOWN_CMD)
                    time.sleep(PEN_DELAY)
                    pen_is_down = True
                continue
            if upper.startswith("M5") or "PEN UP" in upper:
                if pen_is_down:
                    _send_line(ser, PEN_UP_CMD)
                    time.sleep(PEN_DELAY)
                    pen_is_down = False
                continue
            if upper.startswith("M2"):
                # End of program: pen up + return to origin
                _send_line(ser, PEN_UP_CMD)
                time.sleep(PEN_DELAY)
                _send_line(ser, "G0 X0 Y0")
                continue

            # G0 = travel move -> make sure pen is UP
            if upper.startswith("G0") and pen_is_down:
                _send_line(ser, PEN_UP_CMD)
                time.sleep(PEN_DELAY)
                pen_is_down = False

            # G1 = drawing move -> make sure pen is DOWN
            if upper.startswith("G1") and not pen_is_down:
                _send_line(ser, PEN_DOWN_CMD)
                time.sleep(PEN_DELAY)
                pen_is_down = True

            _send_line(ser, line)

    # Finish: pen up, return home
    _send_line(ser, PEN_UP_CMD)
    time.sleep(PEN_DELAY)
    _send_line(ser, "G0 X0 Y0")

    ser.close()
    print("DONE")

if __name__ == "__main__":
    send(sys.argv[1] if len(sys.argv) > 1 else "output.gcode")
