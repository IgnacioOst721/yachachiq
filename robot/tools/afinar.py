"""Afinar velocidad y aceleracion del plotter, con una prueba que delata pasos perdidos.

    python3 tools/afinar.py ver                    -> muestra $110/$111 (vel) y $120/$121 (acel)
    python3 tools/afinar.py probar 1500 300        -> pone vel 1500 mm/min y acel 300 mm/s2 y
                                                      dibuja un cuadrado de 100 mm dos veces
    python3 tools/afinar.py aplicar 1500 300       -> lo deja guardado en el Arduino

Como leer la prueba: el cuadrado se dibuja dos veces seguidas. Si las dos pasadas caen exactamente
una encima de la otra y el trazo cierra en la esquina de salida, la maquina NO pierde pasos a esa
velocidad/aceleracion. Si la segunda pasada sale desplazada o el cuadrado no cierra, es demasiado:
baja la aceleracion primero (es lo que mas pasos hace perder), luego la velocidad.

Referencia de la maquina del equipo: GRBL venia con 500 mm/min y 10 mm/s2 (el minimo). Con eso un
dibujo de 9 min tarda 19. Con 1500 y 300 tarda 7.5. Empieza por 1000 / 150 y sube.
"""
import sys, time
sys.path.insert(0, __file__.rsplit("/tools/", 1)[0])
import config, plotter

CLAVES = {"vel": ("$110", "$111"), "acel": ("$120", "$121")}


def abrir():
    p = plotter.Plotter().connect()
    if p.mode != "real":
        sys.exit("No hay plotter conectado. Revisa el USB del Arduino.")
    p.unlock()
    return p


def ver(p=None):
    p = p or abrir()
    lim = p.limits() or {}
    print("  velocidad maxima: %s mm/min · aceleracion: %s mm/s2" % (lim.get("max_feed", "?"), lim.get("accel", "?")))
    return p


def poner(p, vel, acel):
    for k in CLAVES["vel"]:
        p.send("%s=%g" % (k, vel))
    for k in CLAVES["acel"]:
        p.send("%s=%g" % (k, acel))
    p._limits = None
    print("  puesto: %g mm/min, %g mm/s2" % (vel, acel))


def cuadrado(p, lado=100.0, veces=2):
    up = "G1 Z%.2f F%d" % (config.PEN_UP_Z, config.PEN_FEED)
    down = "G1 Z%.2f F%d" % (config.PEN_DOWN_Z, config.PEN_FEED)
    f = int(config.DRAW_FEED)
    lines = ["G21", "G90", up, "G0 X20 Y20", down]
    for _ in range(veces):
        lines += ["G1 X%g Y20 F%d" % (20 + lado, f), "G1 X%g Y%g" % (20 + lado, 20 + lado),
                  "G1 X20 Y%g" % (20 + lado), "G1 X20 Y20"]
    # una diagonal en zigzag: los cambios bruscos de direccion son donde se pierden pasos
    lines += [up, "G0 X20 Y20", down]
    for i in range(10):
        lines.append("G1 X%g Y%g" % (20 + lado * (i + 1) / 10, 20 + (lado if i % 2 else 0)))
    lines += [up, "G0 X0 Y0"]
    t0 = time.time()
    p.run(lines)
    print("  dibujado en %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["ver"]:
        ver()
    elif len(a) == 3 and a[0] in ("probar", "aplicar"):
        vel, acel = float(a[1]), float(a[2])
        p = abrir()
        antes = p.limits() or {}
        poner(p, vel, acel)
        if a[0] == "probar":
            print("  dibujando el cuadrado de prueba (100 mm, dos pasadas + zigzag)...")
            cuadrado(p)
            print("  MIRA EL PAPEL: si las dos pasadas coinciden y el cuadrado cierra, esta bien.")
            print("  Para dejarlo:  python3 tools/afinar.py aplicar %g %g" % (vel, acel))
            print("  Para volver:   python3 tools/afinar.py aplicar %s %s" % (antes.get("max_feed", 500), antes.get("accel", 10)))
        else:
            print("  guardado en el Arduino (sobrevive a apagones).")
        p.close()
    else:
        print(__doc__)
