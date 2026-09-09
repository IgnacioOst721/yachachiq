"""Calibrar los pasos/mm del plotter midiendo con una regla.

    python3 tools/calibrar.py mover X 100      -> levanta el lapiz y mueve 100 mm en X
    python3 tools/calibrar.py aplicar X 100 97 -> pediste 100, mediste 97: corrige $100

La cuenta es: pasos_nuevos = pasos_viejos * (pedido / medido). Si pides 100 mm y la maquina
recorre 97, es que da pocos pasos por milimetro y hay que subirlos.
"""
import sys, time
sys.path.insert(0, __file__.rsplit("/tools/", 1)[0])
import config, plotter

EJES = {"X": "$100", "Y": "$101", "Z": "$102"}


def abrir():
    p = plotter.Plotter().connect()
    if p.mode != "real":
        sys.exit("No hay plotter conectado (esta en modo simulacion). Revisa el USB del Arduino.")
    p.unlock()
    return p


def ajustes(p):
    """Lee $$ y devuelve {'$100': 80.0, ...}."""
    with p._lock:
        p.ser.reset_input_buffer()
        p.ser.write(b"$$\n")
        time.sleep(1.5)
        data = p.ser.read(p.ser.in_waiting or 1).decode(errors="ignore")
    out = {}
    for line in data.splitlines():
        if "=" in line and line.startswith("$"):
            k, v = line.strip().split("=", 1)
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


def mover(eje, mm):
    p = abrir()
    print(f"  pasos/mm actuales en {eje}: {ajustes(p).get(EJES[eje], '?')}")
    for l in (["G21", "G90"] + (["G1 Z%.2f F%d" % (config.PEN_UP_Z, config.PEN_FEED)] if eje != "Z" else [])):
        p.send(l)
    p.wait_idle(timeout=60)
    print(f"  moviendo {mm:+.0f} mm en {eje}...")
    p.send("G91")
    p.send(f"G1 {eje}{mm:.2f} F400")
    p.wait_idle(timeout=120)
    p.send("G90")
    print(f"  listo. Mide con la regla cuanto se movio DE VERDAD y corre:")
    print(f"     python3 tools/calibrar.py aplicar {eje} {mm:.0f} <lo_que_mediste>")
    p.close()


def aplicar(eje, pedido, medido):
    if medido <= 0:
        sys.exit("La medida tiene que ser mayor que cero.")
    p = abrir()
    clave = EJES[eje]
    viejo = ajustes(p).get(clave)
    if viejo is None:
        sys.exit(f"No pude leer {clave} de la placa.")
    nuevo = round(viejo * pedido / medido, 3)
    print(f"  {clave}: {viejo} -> {nuevo} pasos/mm  (pediste {pedido} mm, midio {medido} mm)")
    if abs(nuevo / viejo - 1) > 0.5:
        sys.exit("  La correccion es enorme (mas del 50%). Revisa la medida antes de aplicarla.")
    r = p.send(f"{clave}={nuevo}")
    print("  la placa respondio:", r)
    print("  guardado en el Arduino (sobrevive a apagones).")
    print(f"  comprueba: python3 tools/calibrar.py mover {eje} {pedido:.0f}  -> deberia medir {pedido:.0f} mm")
    p.close()


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 3 and a[0] == "mover":
        mover(a[1].upper(), float(a[2]))
    elif len(a) == 4 and a[0] == "aplicar":
        aplicar(a[1].upper(), float(a[2]), float(a[3]))
    else:
        print(__doc__)
