# Qué se enchufa dónde

```
                         ┌──────────────── RASPBERRY PI 5 ────────────────┐
  micrófono USB ────────►│ USB                                              │
  cámara 1 (señas) ─────►│ USB      mira a la persona que seña               │
  cámara 2 (dibujo) ────►│ USB      montada arriba de la cama del plotter    │
  Arduino UNO ──────────►│ USB      (cable USB-A ↔ USB-B, el de impresora)   │
  pantalla ─────────────►│ micro-HDMI (cable micro-HDMI → HDMI)             │
  tu Mac (ComfyUI) ─────►│ Ethernet (cable de red directo)                  │
  fuente 5V/5A ─────────►│ USB-C                                            │
                         └──────────────────────────────────────────────────┘

  Arduino UNO + CNC Shield ──► fuente 12 V 3 A (barrel) ──► motores X, Y, Z
  (el pin VIN del Arduino queda doblado hacia afuera: el Arduino se alimenta solo por USB)
```

La Pi 5 tiene 4 puertos USB: micrófono, cámara 1, cámara 2, Arduino. Justo.
No hay parlante: la narración por voz queda desactivada sola (`tts` en modo mock).

## Red con la Mac (sin WiFi, regla de la competencia)

- Mac: Ajustes → Red → adaptador Ethernet/USB-C → IPv4 **Manual**, IP `192.168.7.1`, máscara `255.255.255.0`.
- Pi (una vez):
  `sudo nmcli con add type ethernet ifname eth0 con-name lan ip4 192.168.7.2/24 && sudo nmcli con up lan`
- ComfyUI en la Mac: `python main.py --listen 0.0.0.0 --port 8188`
- Comprobar desde la Pi: `curl http://192.168.7.1:8188/system_stats`

## Las dos cámaras no se deben confundir

Linux las numera al azar al arrancar. Con las dos conectadas:

```
ls /dev/v4l/by-id/
```

Salen dos rutas largas terminadas en `-video-index0`. Copia cada una en su servicio:

- `/etc/systemd/system/yachachiq.service`     → `Environment=YACHACHIQ_PHOTO_CAMERA=/dev/v4l/by-id/...`  (cámara del dibujo)
- `/etc/systemd/system/yachachiq-lsp.service` → `Environment=ASL_CAMERA=/dev/v4l/by-id/...`  (cámara de señas)

Luego `sudo systemctl daemon-reload && sudo systemctl restart yachachiq yachachiq-lsp`.

## Micrófono

Cualquier micrófono USB. Comprobar que la Pi lo ve:

```
arecord -l                      # debe listar el micrófono
arecord -d 3 t.wav && aplay t.wav   # (aplay solo si hay parlante)
```

Si hay más de un dispositivo de audio: `YACHACHIQ_AUDIO_DEVICE=<nombre o número>` en el servicio.

## Orden para encender en la demo

1. Fuente 12 V del plotter → el Arduino parpadea, los motores se traban (normal, GRBL los mantiene).
2. Pi encendida con todo enchufado → aparece la pantalla del kiosko sola (~40 s).
3. Mac con ComfyUI abierto y el cable de red puesto.
4. Papel en la cama. En la pantalla, ⚙ → mover el lápiz a la esquina inferior izquierda hasta que toque → **Fijar origen**.
5. Botón grande → contar la historia.


## La laptop se busca por nombre, no por IP

El robot llega a la Mac de Ignacio como `el-loco-candy.local` (mDNS: avahi en la Pi, Bonjour en la
Mac). Ese nombre resuelve igual en el WiFi de la casa y por el cable Ethernet directo del concurso,
así que no hay que cambiar ninguna IP entre un lugar y otro. Si se usa otra laptop o se renombra la
Mac, basta con `Environment=YACHACHIQ_LAPTOP_HOST=<nombre>.local` (o una IP fija) en
`yachachiq.service`.


## Qué cámara es cuál

Las dos webcams se nombran por su id estable en `/dev/v4l/by-id/` (nunca por `/dev/video0`, que
cambia según el orden en que se enchufan). En la Pi del equipo:

| Función | Cámara | Dispositivo |
|---|---|---|
| Foto del dibujo terminado (mejor calidad, 1080p) | Logitech **Brio 105** | `/dev/v4l/by-id/usb-046d_Brio_105_2549ZB20HA58-video-index0` |
| Lengua de señas + foto de consentimiento (640×480 a 30 fps) | Logitech **C270** | `/dev/v4l/by-id/usb-046d_C270_HD_WEBCAM_200901010001-video-index0` |

La asignación vive en *drop-ins* de systemd, fuera del repo, porque es propia de cada Pi:

```
/etc/systemd/system/yachachiq.service.d/camaras.conf
    [Service]
    Environment=YACHACHIQ_PHOTO_CAMERA=/dev/v4l/by-id/...Brio...-video-index0
    Environment=YACHACHIQ_LSP_CAMERA=/dev/v4l/by-id/...C270...-video-index0
/etc/systemd/system/yachachiq-lsp.service.d/camaras.conf
    [Service]
    Environment=ASL_CAMERA=/dev/v4l/by-id/...C270...-video-index0
```

Para intercambiarlas basta con cruzar las rutas y `sudo systemctl daemon-reload && sudo systemctl
restart yachachiq yachachiq-lsp`. Ver las cámaras conectadas: `ls /dev/v4l/by-id/`.


## Micrófono

El micrófono USB se fija por nombre (sounddevice acepta una parte del nombre), también en un drop-in,
porque la Brio 105 trae su propio micrófono y sin esto el sistema podría elegir ese:

```
/etc/systemd/system/yachachiq.service.d/audio.conf
    [Service]
    Environment=YACHACHIQ_AUDIO_DEVICE=Usb Audio Device
    Environment=YACHACHIQ_AUTO_LISTEN=0
```

Ver los micrófonos: `venv/bin/python -c "import sounddevice as sd; print(sd.query_devices())"`.
La escucha manos-libres (`AUTO_LISTEN`) queda apagada: con un micrófono real se disparaba con el
ruido del ambiente y dejaba al robot ocupado justo cuando alguien intentaba enviar en señas. La voz
se inicia tocando **Con mi voz** en la pantalla.


## Refrigeración (obligatoria en la Pi 5)

La Raspberry Pi 5 **necesita disipador con ventilador**. Sin él, con el robot corriendo, el chip
llegó a 86 °C, se limitó solo (`vcgencmd get_throttled` distinto de `0x0`) y terminó colgándose.
Comprobar en cualquier momento:

```
vcgencmd measure_temp        # sano por debajo de 70 °C
vcgencmd get_throttled       # 0x0 = sin limitación
```

El *Active Cooler* oficial se conecta al zócalo de 4 pines del ventilador y el sistema lo controla
solo. Además, el reconocimiento de señas ya solo hace seguimiento de manos mientras alguien está
señando, lo que bajó el consumo de un núcleo completo a casi nada en reposo.
