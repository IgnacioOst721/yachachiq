# Teclado ASL en Raspberry Pi

Cómo mover el reconocimiento de lengua de señas de la laptop a una Raspberry Pi
con cámara, para que funcione como un aparato independiente.

> ⚠️ **Sin probar todavía en hardware real.** Este código está en la rama `dev`
> hasta que lo corramos en una Pi con cámara y lo depuremos.

## Qué cambia y qué no

**No cambia nada del sistema.** El programa sigue mandando las letras por red al
`receiver.py`, que las escribe con `pyautogui` en **cualquier aplicación** del
equipo que las recibe (un documento, WhatsApp Web, el pipeline del plotter…).
La única diferencia es qué máquina lleva la cámara: antes la laptop, ahora la Pi.

```
ANTES:  laptop (cámara + modelo) ──red──►  equipo receptor  ──► escribe
AHORA:  Raspberry Pi (cámara + modelo) ──red──►  equipo receptor  ──► escribe
```

## Hardware

- Raspberry Pi 5 (recomendada) o Pi 4 — la Pi 4 detecta más lento.
- Webcam USB apuntando al usuario.
- Tarjeta SD con Raspberry Pi OS de **64 bits** (MediaPipe no existe en 32 bits).

## Instalación (una sola vez, desde la laptop por SSH)

```bash
sudo apt update && sudo apt install -y python3-pip python3-opencv git
git clone https://github.com/IgnacioOst721/yachachiq.git ~/yachachiq
cd ~/yachachiq && git checkout dev
pip3 install --break-system-packages mediapipe joblib scikit-learn numpy
```

Prueba manual antes de automatizar nada:

```bash
cd ~/yachachiq/asl
ASL_HEADLESS=1 python3 asl_app.py        # busca el receptor solo
ASL_HEADLESS=1 python3 asl_app.py <IP>   # o apúntale la IP directamente
```

Debe imprimir cada letra detectada. Si la cámara no abre, revisa `ls /dev/video*`
y prueba `ASL_CAMERA=1`.

## Que arranque solo al prender (ya sin laptop)

```bash
sudo cp ~/yachachiq/asl/yachachiq-asl.service /etc/systemd/system/
sudo systemctl enable --now yachachiq-asl
systemctl status yachachiq-asl      # ver que esté corriendo
journalctl -u yachachiq-asl -f      # ver las letras en vivo
```

Desde aquí la Pi se enchufa y funciona sola: sin monitor, sin teclado, sin laptop.

## Modo headless (sin pantalla)

Sin monitor no hay ventana de preview ni atajos de teclado. Todo se maneja con
gestos, que ya estaban en el sistema:

| Acción | Cómo |
|---|---|
| Escribir una letra | Señarla y mantenerla quieta (~0.4 s) |
| Espacio / borrar / cambiar modo | Señas SPACE, BACK y MODE del modelo |
| Enviar la historia | Ambas palmas abiertas, sostener 2 segundos |

La confirmación visual la da la pantalla del equipo que recibe: las letras
aparecen escribiéndose ahí mientras señas.

## Pendientes por resolver al probar

- **Velocidad.** MediaPipe en Pi corre alrededor de 10–20 fps (Pi 5). El filtro
  de estabilidad exige 8 fotogramas seguidos, así que puede que haya que bajar
  `STABLE_FRAMES` si se siente lento.
- **Carga del modelo.** `model.pkl` pesa 89 MB; medir cuánto tarda en cargar al
  arrancar y si la RAM alcanza cómodamente.
- **Confirmación local opcional.** Si se quiere feedback sin mirar la otra
  pantalla: un beep, un LED o una pantallita OLED.
