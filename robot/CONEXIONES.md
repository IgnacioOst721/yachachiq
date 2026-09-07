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
