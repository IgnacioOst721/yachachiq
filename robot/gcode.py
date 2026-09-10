"""Pen strokes (polylines in mm) -> GRBL G-code for the Yachachiq plotter.

    from_polylines(polylines_mm, name) -> list of G-code lines
    save(lines, path)

Pen lift is the Z stepper by default (config.PEN_MODE = "z"); "servo" uses
grbl-servo's M3 S<angle>. The origin (0,0 = bottom-left of the paper, pen
touching) is set ONCE from the screen ("fijar origen" -> G92) and never reset
here, so a second drawing keeps the same paper origin and pen height.
"""
import config


def pen_up():
    if config.PEN_MODE == "servo":
        return [f"M3 S{int(config.SERVO_UP)}", "G4 P0.25"]
    return [f"G1 Z{config.PEN_UP_Z:.2f} F{int(config.PEN_FEED)}"]


def pen_down():
    if config.PEN_MODE == "servo":
        return [f"M3 S{int(config.SERVO_DOWN)}", "G4 P0.25"]
    return [f"G1 Z{config.PEN_DOWN_Z:.2f} F{int(config.PEN_FEED)}"]


def home_sweep():
    """Back to the physical corner (left + bottom stops) and make that the origin again.
    No limit switches on this machine, so the corner is found by driving into the stops."""
    if not config.HOME_SWEEP:
        return []
    return ["G91", f"G1 X{-abs(config.HOME_SWEEP_X):.1f} Y{-abs(config.HOME_SWEEP_Y):.1f} F{int(config.HOME_SWEEP_FEED)}",
            "G90", "G10 L20 P1 X0 Y0", "G54"]


def from_polylines(polylines, name="story"):
    lines = [f"(Yachachiq - {name} - {len(polylines)} strokes)", "G21", "G90"]
    lines += pen_up()
    for pl in polylines:
        if len(pl) < 2:
            continue
        x0, y0 = pl[0]
        lines.append(f"G0 X{x0:.2f} Y{y0:.2f}")
        lines += pen_down()
        for x, y in pl[1:]:
            lines.append(f"G1 X{x:.2f} Y{y:.2f} F{int(config.DRAW_FEED)}")
        lines += pen_up()
    # al terminar: el lapiz queda arriba (el bucle ya lo subio) y el cabezal vuelve al origen,
    # que es donde se fijo el papel. Asi la camara ve el dibujo despejado y el siguiente
    # dibujo empieza siempre desde el mismo punto conocido.
    lines.append("G0 X0 Y0")
    lines += home_sweep()
    return lines


def save(lines, path):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def is_motion(line):
    s = line.strip().upper()
    return s.startswith("G0") or s.startswith("G1")
