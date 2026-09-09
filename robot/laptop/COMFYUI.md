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

## Pruebas de prompt hechas (2026-09-07, DreamShaper 8 en la M4)

Se probaron 5 variantes con la misma escena. Resultado: **el `STYLE_PROMPT` que ya estaba
es el mejor**; los intentos de quitar el relleno negro rompieron otra cosa.

| Variante | Monocromo | Compone la escena | Relleno negro | Veredicto |
|---|---|---|---|---|
| **Actual** (Sarhua/retablo) | ✅ | ✅ mejor de todas | 24% | **la que se usa** |
| "outline only, no filled areas" | ❌ salió a color | ❌ ignoró la escena | 7% | peor |
| Prompt de Clara (libro para colorear) | ❌ a color | ❌ muy vacío (19 trazos) | 1% | peor |
| Monocromo forzado + anti-relleno | ✅ | ❌ perdió la escena | 18% | peor |
| Actual + negativos anti-relleno | ❌ a color | — | 0% | peor |

**Aprendizaje:** los negativos tipo "solid black areas / filled shapes" sí quitan el relleno,
pero también empujan al modelo fuera del blanco y negro. Las menciones "Sarhua tabla, Ayacucho
retablo" son las que fuerzan el monocromo, y de paso traen el relleno: van juntas.

El relleno negro no arruina el dibujo — el trazador lo convierte en contornos y el resultado
es una lámina de líneas correcta. Lo que sí se ajustó fue el **tiempo de dibujo**:

| MIN_STROKE_PX | MAX_STROKES | Trazos | Tiempo del plotter |
|---|---|---|---|
| 12 | 900 | 197 | 11.0 min |
| **20** | **350** | **145** | **8.9 min** ← actual |
| 26 | 260 | 122 | 8.0 min |

Para una demo con cola de visitantes, subir `MIN_STROKE_PX` a 26-35 baja el tiempo sin
cambiar el aspecto general.


## Arranque automático en la Mac

`pe.fdr.yachachiq.comfyui.plist` es un *LaunchAgent*: ComfyUI se levanta solo al iniciar sesión y se
reinicia si se cae (log en `~/Library/Logs/yachachiq-comfyui.log`). Instalarlo en otra Mac:

```
cp pe.fdr.yachachiq.comfyui.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/pe.fdr.yachachiq.comfyui.plist
```

Quitarlo: `launchctl bootout gui/$(id -u)/pe.fdr.yachachiq.comfyui`. Las rutas dentro del plist son las
de la Mac de Ignacio (`/Users/ignacioosterling/ComfyUI`); cambiarlas si se usa otra.
