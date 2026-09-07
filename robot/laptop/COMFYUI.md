# ComfyUI en la Mac (generador de imágenes principal)

1. Abre ComfyUI de forma que la Pi lo alcance por la red:
   ```
   python main.py --listen 0.0.0.0 --port 8188
   ```
2. Checkpoint: el workflow `comfyui_workflow.json` trae `sd_turbo.safetensors` (4 pasos, cfg 1, ~2 s por imagen).
   Si prefieres **DreamShaper** (el que usaron antes, mejor calidad, ~10-20 s):
   - en la Pi: `YACHACHIQ_COMFYUI_CHECKPOINT=dreamshaper_8.safetensors  YACHACHIQ_COMFYUI_STEPS=20  YACHACHIQ_COMFYUI_CFG=7`
   - o exporta tu propio workflow en ComfyUI (Settings → Enable Dev mode → **Save (API Format)**) y reemplaza este archivo.
3. Cable Ethernet Pi ↔ Mac. En la Mac: Ajustes → Red → adaptador Ethernet → IPv4 manual `192.168.7.1`, máscara `255.255.255.0`.
   En la Pi: `sudo nmcli con add type ethernet ifname eth0 con-name lan ip4 192.168.7.2/24 && sudo nmcli con up lan`
4. Prueba desde la Pi: `curl http://192.168.7.1:8188/system_stats`
5. El prompt de estilo está en `config.py` (`STYLE_PROMPT`): pide dibujo de línea negra, estilo tabla de Sarhua / retablo ayacuchano.

Si ComfyUI no responde, el robot cae a `motifs` (escena andina dibujada por código) y avisa en pantalla.
