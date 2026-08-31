# Teclado LSP en Raspberry Pi

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
cd ~/yachachiq/lsp
ASL_HEADLESS=1 python3 lsp_app.py        # busca el receptor solo
ASL_HEADLESS=1 python3 lsp_app.py <IP>   # o apúntale la IP directamente
```

Debe imprimir cada letra detectada. Si la cámara no abre, revisa `ls /dev/video*`
y prueba `ASL_CAMERA=1`.

## Conectar la Pi a una Mac (la de Clara o cualquiera)

La Pi no se "empareja" con nada: las dos máquinas solo tienen que estar en la
**misma red** (el hotspot del celular, el wifi del colegio, o un cable de red).
De ahí el descubrimiento es automático.

### En la Mac que va a recibir el texto (una sola vez)

```bash
pip3 install pyautogui
```

Luego **dale permiso de Accesibilidad**, o no va a poder escribir:
Ajustes del Sistema → Privacidad y Seguridad → **Accesibilidad** → activar
**Terminal**. Sin esto el programa corre pero no aparece ninguna letra, y es
el error más común.

### Cada vez que se usa

1. **En la Mac:**
   ```bash
   python3 receiver.py
   ```
   Imprime su IP en pantalla. Déjalo abierto.

2. **En la Pi:** ya arranca solo con systemd. Si lo corres a mano:
   ```bash
   python3 lsp_app.py            # busca la Mac sola
   python3 lsp_app.py 192.168.1.5   # o dale la IP que imprimió la Mac
   ```

3. **En la Mac, haz clic donde quieras que escriba** — un documento, el chat,
   la caja de texto del pipeline. Las letras caen donde esté el cursor.

4. Empieza a señar.

### Si no se encuentran solas

El descubrimiento usa un mensaje UDP de difusión que algunas redes bloquean
(wifi de colegios sobre todo). Solución: pasarle la IP a mano, como en el
paso 2. Para dejarla fija en el arranque automático, agrega la IP al final del
`ExecStart` en `yachachiq-lsp.service`.

Otras cosas que revisar: que el firewall de la Mac permita conexiones entrantes,
y que ambas estén en la misma red (no una en el hotspot y otra en el wifi).

## Que arranque solo al prender (ya sin laptop)

```bash
sudo cp ~/yachachiq/lsp/yachachiq-lsp.service /etc/systemd/system/
sudo systemctl enable --now yachachiq-lsp
systemctl status yachachiq-lsp      # ver que esté corriendo
journalctl -u yachachiq-lsp -f      # ver las letras en vivo
```

Desde aquí la Pi se enchufa y funciona sola: sin monitor, sin teclado, sin laptop.

## Con pantalla o sin pantalla

El programa detecta solo si hay monitor conectado y se adapta. Las dos opciones
funcionan sin la laptop.

| | Con pantalla | Sin pantalla (aparato) |
|---|---|---|
| Ventana de reconocimiento | Sí, igual que en la Mac | No |
| Se maneja con | Gestos + teclado USB opcional | Solo gestos |
| Consumo y tamaño | Mayor | Mínimo |
| Bueno para | Demostrar en la competencia | Instalación en comunidad |

**Pantallas que sirven:** la pantalla táctil oficial de 7" de Raspberry, cualquier
monitor pequeño HDMI, o hasta un televisor. Ojo: la Pi 4 y 5 usan **micro-HDMI**,
así que hace falta el cable o adaptador correcto.

Para la WRO conviene la pantalla: el jurado ve en vivo cómo la máquina reconoce
cada seña, que es la parte más impresionante del sistema.

Para activar cada modo en el arranque automático, ver los comentarios dentro de
`yachachiq-lsp.service`.

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

## Velocidad (optimizaciones ya incluidas)

Verificado con los datos reales del proyecto (2,500 muestras):

- **El modelo se recorta a 100 árboles al cargar** — coincide al 100% con el
  modelo completo de 400 y predice 2 veces más rápido (13 ms vs 26 ms por
  fotograma en Mac). Ajustable con `ASL_TREES` (0 = usar los 400).
- **La cámara captura a 640×480 en la Pi** — suficiente para detectar la mano
  y mucho más liviano que 1280×720.
- **Si aun así se siente lento:** `ASL_FAST=1` baja la complejidad del rastreo
  de MediaPipe (más fps, landmarks algo menos precisos). Probar señando el
  abecedario completo antes de dejarlo activado.
- Última palanca: bajar `STABLE_FRAMES` en `lsp_app.py` de 8 a 5–6.

## Pendientes por resolver al probar

- **Fps reales en la Pi.** Medir con `ASL_FAST` apagado y prendido.
- **Carga del modelo.** `model.pkl` pesa 89 MB; medir cuánto tarda al arrancar.
- **Confirmación local opcional.** Si se quiere feedback sin mirar la otra
  pantalla: un beep, un LED o una pantallita OLED.
