# ComfyUI en la Mac de Ignacio (generador de imágenes del robot)

Ya está instalado en `~/ComfyUI` con **DreamShaper 8** (el modelo que el equipo usó antes).
Python 3.12 propio en `~/.local/pythons/python` — no toca el Python del sistema.

## Arrancarlo

```bash
~/ComfyUI/start_yachachiq.sh          # escucha en toda la red: la Pi lo alcanza
~/ComfyUI/start_yachachiq.sh local    # solo esta Mac, para probar sin la Pi
```

Queda en `http://<ip-de-la-mac>:8188`. Déjalo abierto durante la demo.

## Comprobar que funciona

Desde la Mac:  `curl -s http://127.0.0.1:8188/system_stats | head -5`
Desde la Pi:   `curl -s http://192.168.7.1:8188/system_stats | head -5`

## Ajustes del modelo

En `config.py` del robot (o como variables de entorno):

| Ajuste | DreamShaper 8 (actual) | sd_turbo (alternativa rápida) |
|---|---|---|
| `COMFYUI_CHECKPOINT` | `dreamshaper_8.safetensors` | `sd_turbo.safetensors` |
| `COMFYUI_STEPS` | `22` | `4` |
| `COMFYUI_CFG` | `7.0` | `1.0` |
| Tiempo por imagen (M4) | ~8-15 s | ~2 s |

Ejemplo para probar sd_turbo sin editar nada:
```bash
YACHACHIQ_COMFYUI_CHECKPOINT=sd_turbo.safetensors YACHACHIQ_COMFYUI_STEPS=4 YACHACHIQ_COMFYUI_CFG=1 python3 server.py
```

## Red con la Pi (competencia: sin WiFi, cable Ethernet)

- Mac: Ajustes → Red → adaptador Ethernet → IPv4 **Manual**, IP `192.168.7.1`, máscara `255.255.255.0`.
- Pi: `sudo nmcli con add type ethernet ifname eth0 con-name lan ip4 192.168.7.2/24 && sudo nmcli con up lan`
- El robot ya apunta a `192.168.7.1:8188` por defecto.

## Si ComfyUI no responde

El robot no se queda colgado: pasa al respaldo `motifs` (escena andina dibujada por código)
y lo avisa en la pantalla. La demo nunca se detiene por esto.

## Estilo de las imágenes

`STYLE_PROMPT` en `config.py` pide dibujo de línea negra, estilo tabla de Sarhua / retablo
ayacuchano, sin sombras ni color — que es lo que el plotter puede trazar. Si los dibujos salen
muy cargados de detalle, sube `MIN_STROKE_PX` o baja `MAX_STROKES` en `config.py`.
