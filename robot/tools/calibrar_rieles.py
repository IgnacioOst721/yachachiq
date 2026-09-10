"""Calibracion completa a partir del recorrido REAL de los rieles, medido con wincha.

    python3 tools/calibrar_rieles.py <cm riel de ARRIBA> <cm riel IZQUIERDO> [--pi admin@IP] [--dry]

  1. pasos/mm reales en la placa (por la API del robot):
       riel de arriba  = GRBL Y, llego al tope con 15 mm pedidos -> $101 = 80 * 15 / arriba_mm
       riel izquierdo  = GRBL X, llego al tope con 24 mm pedidos -> $100 = 80 * 24 / izquierdo_mm
  2. arduino/grbl_settings.txt en la Pi con esos pasos/mm y velocidades de plotter en unidades
     reales ($110/$111 2000 mm/min, $120/$121 200 mm/s2, $1=255), cargado a la placa.
  3. papel = recorrido real menos margen (visto de frente: ancho = arriba, alto = izquierdo) y
     velocidades de dibujo reales, en un drop-in de systemd; reinicia el robot.
"""
import json, subprocess, sys, urllib.request

PEDIDO_ARRIBA, PEDIDO_IZQ, PASOS_ACTUALES = 15.0, 24.0, 80.0


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=90).read().decode())


def pi_bash(pi, script):
    """Run a bash script on the Pi via stdin: no quoting problems."""
    return subprocess.run(["ssh", "-o", "BatchMode=yes", pi, "bash", "-s"], input=script, text=True,
                          capture_output=True, timeout=120)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        print(__doc__); sys.exit(1)
    pi = sys.argv[sys.argv.index("--pi") + 1] if "--pi" in sys.argv else "admin@192.168.1.57"
    dry = "--dry" in sys.argv
    arriba_mm, izq_mm = float(args[0]) * 10, float(args[1]) * 10
    y_steps = round(PASOS_ACTUALES * PEDIDO_ARRIBA / arriba_mm, 3)
    x_steps = round(PASOS_ACTUALES * PEDIDO_IZQ / izq_mm, 3)
    margen = 8.0
    W, H = round(arriba_mm - 2 * margen), round(izq_mm - 2 * margen)
    print(f"riel de arriba : {arriba_mm:.0f} mm reales por {PEDIDO_ARRIBA:.0f} pedidos -> x{arriba_mm/PEDIDO_ARRIBA:.1f} -> $101={y_steps}")
    print(f"riel izquierdo : {izq_mm:.0f} mm reales por {PEDIDO_IZQ:.0f} pedidos -> x{izq_mm/PEDIDO_IZQ:.1f} -> $100={x_steps}")
    print(f"papel real     : {W} ancho x {H} alto (margen {margen:.0f}) · dibujo 1200 mm/min, viaje 2500")
    if dry:
        print("(simulacion: no se cambio nada)"); return
    base = f"http://{pi.split('@')[-1]}:8877"
    for axis, cmd, meas in (("Y", PEDIDO_ARRIBA, arriba_mm), ("X", PEDIDO_IZQ, izq_mm)):
        r = post(base, "/api/plotter/calibrate", {"axis": axis, "commanded": cmd, "measured": meas})
        print(f"  placa {axis}: {r.get('old')} -> {r.get('new')} pasos/mm ·", "ok" if r.get("ok") else r.get("error"))
        if not r.get("ok"):
            sys.exit("no se aplico; revisa las medidas")
    script = f"""set -e
cd ~/yachachiq/robot
sed -i 's/^\\$100=.*/$100={x_steps}/; s/^\\$101=.*/$101={y_steps}/; s/^\\$110=.*/$110=2000/; s/^\\$111=.*/$111=2000/; s/^\\$120=.*/$120=200/; s/^\\$121=.*/$121=200/; s/^\\$1=.*/$1=255/' arduino/grbl_settings.txt
printf '[Service]\\nEnvironment=YACHACHIQ_PAPER_W_MM={W}\\nEnvironment=YACHACHIQ_PAPER_H_MM={H}\\nEnvironment=YACHACHIQ_MARGIN_MM={margen:.0f}\\nEnvironment=YACHACHIQ_DRAW_FEED=1200\\nEnvironment=YACHACHIQ_TRAVEL_FEED=2500\\nEnvironment=YACHACHIQ_PEN_FEED=600\\nEnvironment=YACHACHIQ_PEN_UP_Z=3\\n' | sudo tee /etc/systemd/system/yachachiq.service.d/calibracion.conf >/dev/null
sudo systemctl daemon-reload && sudo systemctl restart yachachiq && sleep 9
curl -s -X POST -H 'content-type: application/json' -d '{{}}' localhost:8877/api/plotter/settings
echo
curl -s localhost:8877/api/state | python3 -c "import sys,json; d=json.load(sys.stdin); print('robot:', d['state'], '| plotter:', d['modes']['plotter'])"
"""
    r = pi_bash(pi, script)
    print(r.stdout.strip()); print(r.stderr.strip()[-300:] if r.returncode else "")
    print("LISTO: ejes calibrados, velocidades reales, papel a tamano real. Ahora un dibujo de prueba.")


if __name__ == "__main__":
    main()
