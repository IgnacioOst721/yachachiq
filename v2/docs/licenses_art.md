# Licencias – dominio ART (dibujo)

Todo lo que usa el dibujo es código abierto o datos libres (regla WRO FI 5.3). Verificado el
2026-09-27 en las fichas de Hugging Face (etiqueta `license:`) y en los metadatos de PyPI del
entorno `.venvs/art`.

## Modelos (solo en la MacBook, se descargan una vez y luego funcionan sin internet)

| Modelo | Repositorio que usamos | Tamaño | Licencia | Uso |
|---|---|---|---|---|
| Z-Image-Turbo 6B (Tongyi-MAI, Alibaba) | `mflux-community/z-image-turbo-mflux-q4` (de `Tongyi-MAI/Z-Image-Turbo`) | 5,9 GB | Apache-2.0 | generador de imagen (opción de calidad) |
| FLUX.2 [klein] 4B (Black Forest Labs) | `mflux-community/flux2-klein-4b-mflux-q4` (de `black-forest-labs/FLUX.2-klein-4B`) | 4,6 GB | Apache-2.0 | generador de imagen (opción rápida) |
| FLUX.1 [schnell] 12B (Black Forest Labs) | `mflux-community/flux-1-schnell-mflux-q4` | 9,6 GB | Apache-2.0 | evaluado; demasiado grande para compartir la Mac |
| ERNIE-Image-Turbo (Baidu) | `mflux-community/ernie-image-turbo-mflux-q4` | 6,6 GB | Apache-2.0 | descargado para comparar |
| Qwen3-8B | `mlx-community/Qwen3-8B-4bit` (de `Qwen/Qwen3-8B`) | 4,6 GB | Apache-2.0 | LLM (plan de la escena) |
| Qwen2.5-7B-Instruct | `mlx-community/Qwen2.5-7B-Instruct-4bit` | 4,3 GB | Apache-2.0 | LLM (comparación) |
| Qwen3.5-9B (multimodal) | `mlx-community/Qwen3.5-9B-MLX-4bit` (de `Qwen/Qwen3.5-9B`) | 6,0 GB | Apache-2.0 | LLM + VLM en un solo modelo (comparación) |
| Qwen3-VL-8B-Instruct | `mlx-community/Qwen3-VL-8B-Instruct-4bit` | 5,8 GB | Apache-2.0 | VLM (revisar el dibujo; BOX-ANALYSIS también) |
| Qwen2.5-VL-7B-Instruct | `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` | 5,7 GB | Apache-2.0 | VLM (comparación) |
| DreamShaper 8 (SD 1.5) | `~/ComfyUI/models/checkpoints/dreamshaper_8.safetensors` (de v1) | 2,1 GB | **CreativeML OpenRAIL-M** (licencia abierta con restricciones de uso; no es OSI) | respaldo por ComfyUI, igual que en v1 — **marcado** |

Ninguno de los modelos elegidos es CC-BY-NC. El único con licencia no-OSI es DreamShaper
(solo respaldo).

## Bibliotecas

| Biblioteca | Versión | Licencia | Dónde |
|---|---|---|---|
| mlx / mlx-metal | 0.32.2 | MIT | Mac |
| mlx-lm | 0.31.3 | MIT | Mac |
| mlx-vlm | 0.7.3 | MIT | Mac |
| mflux | 0.20.0 | MIT | Mac |
| transformers | 5.17.0 | Apache-2.0 | Mac (dependencia) |
| torch | 2.14.0 | BSD-3-Clause + Apache-2.0, MIT, BSL-1.0 (componentes) | Mac (dependencia de mflux) |
| torchvision | 0.29.0 | BSD-3-Clause | Mac (lo pide el procesador de Qwen-VL de transformers) |
| huggingface-hub | 1.33.0 | Apache-2.0 | Mac (descargas) |
| segno | 1.6.6 | BSD-3-Clause | Jetson y Mac (QR; opcional) |
| opencv-contrib-python-headless | 5.0 | Apache-2.0 | todo (visión, adelgazamiento Guo-Hall, QRCodeEncoder/Detector) |
| numpy | 2.2 | BSD-3-Clause | todo |
| Pillow | 12 | MIT-CMU (HPND) | todo |
| ComfyUI | (instalado en v1) | GPL-3.0 | respaldo en la Mac, proceso aparte (solo lo llamamos por HTTP) |

## Datos

- **Tipografías Hershey** (`yq/art/fonts/*.jhf`, copiadas de github.com/kamalmostafa/hershey-fonts):
  se pueden usar para cualquier fin siempre que se distribuya el aviso de reconocimiento
  (Dr. A. V. Hershey, U.S. National Bureau of Standards; formato de James Hurt, Cognition Inc.).
  El aviso completo va junto a los archivos en `yq/art/fonts/HERSHEY_NOTICE.txt`.
- **Motivos andinos procedurales** (`yq/art/motifs.py`) y **palabras clave** (`yq/art/offline.py`):
  código propio del equipo (v1), mejorado.
