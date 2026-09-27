# ART – cómo Yachachiq hace el dibujo

## Qué hace (en simple)

1. **Entender la historia** (MacBook, modelo de lenguaje Qwen): lee la historia (en cualquier idioma;
   si es quechua usa también la traducción al castellano) y escribe un **plan de escena**: qué dibujar,
   las cosas que *tienen* que aparecer (los "elementos"), notas culturales (un cóndor andino tiene collar
   blanco, una llama no es una alpaca, nada de sombreros mexicanos…) y la descripción para el generador.
2. **Generar la imagen** (MacBook, generador de imágenes abierto): pide un dibujo de **tinta negra sobre
   papel blanco**, estilo libro para colorear.
3. **Revisar la imagen** (MacBook, modelo de visión Qwen-VL + medidas de píxeles): ¿están todos los
   elementos?, ¿es dibujo de líneas limpio (sin color, sin sombras, sin letras, sin marco)? Si falta algo,
   genera otra vez con otra semilla y pidiendo con más fuerza lo que faltó (hasta 3 intentos) y se queda
   con la mejor.
4. **Convertir a trazos de lápiz** (Jetson o Mac, solo CPU): la imagen se vuelve líneas centrales
   (vectorización), las zonas negras grandes se vuelven contorno + rayado (estilo grabado), se quitan
   marcos y manchitas, y se ordenan los trazos para que el lápiz viaje lo menos posible levantado.
5. **La parte de atrás**: título, la historia en su idioma original, la traducción al castellano si es
   distinta, una línea de créditos y un **código QR** hacia la galería web — todo con letra de un solo
   trazo (Hershey), que el lápiz puede escribir.
6. Sale todo listo para la impresora de Joaquín: `front.svg`, `back.svg`, `front.gcode`, `back.gcode`
   (+ vistas previas PNG para la pantalla).

**Sin la MacBook** (o si algo falla allá) el dibujo igual sale: el Jetson compone la escena con los
motivos andinos dibujados por código de v1 (cóndor, llama, montaña, sol, chakana, andenes…).

## Archivos

| Archivo | Qué hace |
|---|---|
| `yq/art/drawing.py` | `make_drawing(story, out_dir, on_progress)`: todo el proceso (lo llama la interfaz) |
| `yq/art/vectorize.py`, `skeleton.py` | imagen → líneas centrales (umbral adaptativo, esqueleto Guo-Hall, grafo, poda, unión en cruces, suavizado) |
| `yq/art/hatch.py` | rayado y tramado (rellenos, tonos, QR) |
| `yq/art/order.py` | orden del lápiz (vecino más cercano + 2-opt + Or-opt) y modo "línea continua" |
| `yq/art/text.py`, `fonts/` | letras Hershey de un trazo con tildes, ñ, ü, ¿ ¡ y apóstrofos del quechua |
| `yq/art/qr.py` | QR (corrección H) dibujado con rayas; se comprueba que se puede leer |
| `yq/art/layout.py` | ubica el dibujo en la hoja y arma la parte de atrás |
| `yq/art/svg.py`, `gcode.py` | salida para la impresora (mm, un `<path>` por trazo, G-code GRBL, tiempo estimado) |
| `yq/art/motifs.py`, `offline.py` | respaldo sin Mac (motivos de v1 + palabras clave) |
| `yq/art/settings.py` | papel, lápiz, velocidades (ver abajo) |
| `yq/macworker/routes_art.py` | `/story/clean`, `/story/plan`, trabajo `image` en la Mac |
| `yq/macworker/models/llm.py`, `vlm.py` | `chat()` y `ask()` (también los usa BOX-ANALYSIS) |
| `yq/macworker/models/art_*.py` | generador, estilo del prompt, plan, verificación |
| `tools/art_eval.py` | evaluación de punta a punta con 10 historias |

## Instalar

En la **Mac** (una vez, con internet):

```bash
cd ~/yachachiq/v2
~/.local/bin/uv venv --python 3.10 .venvs/art
~/.local/bin/uv pip install --python .venvs/art/bin/python -r requirements-common.txt \
    -r requirements/art-mac.txt --excludes requirements/art-mac-excludes.txt
# modelos (quedan en ~/.cache/huggingface, luego todo funciona sin internet):
for r in mlx-community/Qwen3-8B-4bit mlx-community/Qwen3-VL-8B-Instruct-4bit \
         mflux-community/z-image-turbo-mflux-q4 mflux-community/flux2-klein-4b-mflux-q4; do
  .venvs/art/bin/hf download $r; done
```

En el **Jetson** solo hace falta `requirements-common.txt` (+ `requirements/art-jetson.txt`, opcional).

## Probar

```bash
cd ~/yachachiq/v2
.venvs/art/bin/python -m pytest tests/art                 # rápido, sin modelos (~15 s)
.venvs/art/bin/python -m yq.common.heavylock .venvs/art/bin/python -m pytest -m heavy tests/art
.venvs/art/bin/python -m yq.common.heavylock .venvs/art/bin/python tools/art_eval.py --out /tmp/art_eval
```

Todo tiene simulador: `YQ_MOCK=1` (o `YQ_MOCK_LLM=1`, `YQ_MOCK_VLM=1`, `YQ_MOCK_IMAGEGEN=1`,
`YQ_MOCK_MAC=1`) hace que funcione sin modelos ni Mac.

## Cambiar papel o lápiz (para la impresora de Joaquín)

Todo se cambia con variables de entorno `YQ_ART_...` (o editando `yq/art/settings.py`):

| Variable | Normal | Qué es |
|---|---|---|
| `YQ_ART_PAPER_W_MM`, `YQ_ART_PAPER_H_MM` | 210, 297 | tamaño del papel (A4 vertical). A5: 148, 210 |
| `YQ_ART_MARGIN_MM` | 10 | margen sin dibujo |
| `YQ_ART_PEN_WIDTH_MM` | 0.5 | grosor del lápiz (separa las rayas del QR y de los rellenos) |
| `YQ_ART_PEN_MODE` | `z` | `z` (eje Z), `servo` (`M3 S…`), `none` (lápiz que nunca se levanta: una sola línea continua) |
| `YQ_ART_PEN_UP_Z`, `YQ_ART_PEN_DOWN_Z` | 2, 0 | alturas del lápiz en modo `z` |
| `YQ_ART_SERVO_UP`, `YQ_ART_SERVO_DOWN`, `YQ_ART_SERVO_DELAY_S` | 90, 30, 0.15 | modo `servo` |
| `YQ_ART_DRAW_FEED`, `YQ_ART_TRAVEL_FEED` | 1500, 3000 | velocidades en mm/min |
| `YQ_ART_ACCEL_MM_S2` | 300 | aceleración de la máquina (solo para estimar el tiempo) |
| `YQ_ART_Y_UP`, `YQ_ART_SWAP_XY`, `YQ_ART_ORIGIN_X_MM`, `YQ_ART_ORIGIN_Y_MM` | sí, no, 0, 0 | dónde está el 0,0 de la máquina |
| `YQ_ART_MACHINE_W_MM`, `YQ_ART_MACHINE_H_MM` | 0 | recorrido máximo; si se pone, el G-code se revisa contra él |
| `YQ_ART_QR_SIZE_MM` | 36 | lado del QR (mínimo 30) |
| `YQ_ART_MAX_DRAW_MINUTES` | 25 | si el frente tardaría más, se quitan los trazos más cortitos |
| `YQ_ART_FILL_STYLE` | `hatch` | zonas negras: `outline` (solo contorno), `hatch` (contorno + rayado), `solid` |
| `YQ_ART_TONES` | no | rayado por tonos de gris (más detalle, más tiempo) |
| `YQ_ART_IMAGE_BACKEND` | ver abajo | generador: `z-image-turbo`, `flux2-klein-4b`, `schnell`, `comfyui` |

El G-code: `G21 G90`, origen abajo-izquierda con Y hacia arriba (como v1), `G0` para moverse con el lápiz
arriba, `G1 … F` dibujando, y al final vuelve al origen. Cada archivo empieza con un comentario con el
número de trazos y el tiempo estimado. Si algún punto se sale del papel, **no se genera** el G-code
(error claro), así la máquina nunca choca contra el borde.

## Mediciones

### Vectorizador (exactitud geométrica, medido con `tests/art/test_vectorize.py`)

Dibujamos figuras conocidas (líneas, círculo, curva, espiral, letras con tildes) con grosores de 3, 5 y 8
píxeles, las vectorizamos a 1400 px y medimos la distancia entre la línea central verdadera y los trazos
que salen (en píxeles de trabajo; 1 px ≈ 0,14 mm en A4). Formato: promedio / 95 % / peor punto.

| Figura | 3 px | 5 px | 8 px |
|---|---|---|---|
| línea | 0,41 / 0,68 / 1,00 | 0,46 / 0,72 / 0,78 | 0,47 / 0,74 / 1,00 |
| círculo | 0,23 / 0,51 / 0,77 | 0,22 / 0,58 / 0,89 | 0,26 / 0,61 / 0,86 |
| curva | 0,26 / 0,52 / 0,99 | 0,22 / 0,44 / 0,80 | 0,20 / 0,43 / 0,61 |
| espiral | 0,25 / 0,51 / 1,00 | 0,26 / 0,56 / 1,00 | 0,24 / 0,58 / 1,00 |
| letras "Ñandú ABRKX" | 0,45 / 1,00 / 7,9 | 0,50 / 1,28 / 3,0 | 0,49 / 1,56 / 7,6 |

O sea: el trazo cae a **menos de 1 px (0,14 mm) de la línea real**, menos que el grosor del lápiz.
En letras el peor punto (7-8 px ≈ 1 mm) es un cachito de 1 mm donde el palito de la "a" sale de su
panza: el adelgazamiento no puede distinguirlo de una "rebaba" y a veces lo corta.
Además: una T da 2 trazos, una X da 2 trazos rectos (se cruzan en vez de partirse en 4), un marco
rectangular se elimina, los ojos (puntitos) se conservan y el ruido de 1-4 píxeles se descarta.

Qué mejoró respecto de v1: umbral adaptativo con histéresis a ≥1024 px (v1 trabajaba a 600 px),
esqueleto Guo-Hall (Zhang-Suen borraba líneas a 45°: ¡desaparecía el brazo de una K!), grafo con
cruces, poda de rebabas según el grosor de cada rama, unión recta a través de cruces, esquinas
afiladas otra vez, suavizado que respeta las esquinas y simplificación sub-píxel; rellenos negros →
contorno + rayado según lo oscuro (más oscuro = más denso, cruzado si es negro).

### Orden del lápiz (dibujos reales generados, mm de viaje con el lápiz levantado)

| Dibujo | Trazos | Orden original | Vecino más cercano | + 2-opt + Or-opt (el nuestro) |
|---|---|---|---|---|
| cóndor (Z-Image) | 370 | 12 182 | 1 803 | **1 379** |
| niña y llama (Z-Image) | 329 | 12 488 | 2 096 | **1 654** |
| zorro y luna (Z-Image) | 263 | 12 131 | 1 698 | **1 369** |
| niña y llama (FLUX.2) | 344 | 17 640 | 1 926 | **1 492** |
| zorro y luna (FLUX.2) | 246 | 10 443 | 1 454 | **1 150** |

El viaje queda en ~11 % del original y ~22 % menos que el método de v1 (vecino más cercano). Además
se unen los trazos que terminan donde empieza el siguiente (8-13 % menos levantadas de lápiz).
Modo sin levantar el lápiz (`PEN_MODE=none`): se vuelve sobre la tinta ya dibujada cuando se puede;
en un dibujo típico quedan ~1,2 m de líneas de unión visibles, así que recomendamos un lápiz que se levante.

### Generador de imágenes (medido en esta MacBook M4 16 GB, 768 × 1088 px, 4 escenas)

| Modelo (4 bits, mflux) | Pasos | Segundos por imagen | Pico de memoria MLX | Calidad (vista a ojo) |
|---|---|---|---|---|
| **FLUX.2 [klein] 4B** (elegido) | 4 | **46-52 s** | 12,1 GB* | líneas limpias, escena fiel, cóndor con collar blanco, rellenos negros grandes a veces |
| Z-Image-Turbo 6B | 9 | 168-264 s | 11,5 GB* | muy parecida (algo más de detalle en casas y textiles) |
| FLUX.1 [schnell] 12B | 4 | no medido | — | 9,6 GB solo de pesos: no cabe junto al LLM/VLM en 10,5 GB |
| DreamShaper 8 (SD1.5, ComfyUI, respaldo v1) | 22 | v1 midió 8-15 s a 512 px; hoy no terminó en 240 s (Mac en swap) | ComfyUI retuvo 9 GB | sigue peor la historia; solo respaldo |

\* Pico medido **sin** límite de caché de MLX y con la Mac en swap por los otros cinco trabajos
(16-18 GB de swap usados); por eso los tiempos son pesimistas. Ahora `art_image.py` limita la caché
de MLX a 1 GB (`YQ_ART_MLX_CACHE_GB`) como hace `mflux --low-ram`.
Elegimos **FLUX.2 [klein] 4B**: 3,4 veces más rápido que Z-Image con calidad equivalente para
dibujo de línea, y sus 4,6 GB caben junto al modelo de visión (5,8 GB) dentro del presupuesto de 10,5 GB.
Z-Image sigue disponible: `YQ_ART_IMAGE_BACKEND=z-image-turbo`.
Los dos ignoran a veces "sin color" en un animal (el zorro sale anaranjado); no importa: el
vectorizador trabaja en gris y la verificación marca `not_line_art` si hay mucho color.

### Modelo de lenguaje (plan de la escena) y modelo de visión (revisión)

| Qué | Modelo elegido | Medido |
|---|---|---|
| LLM | **Qwen3-8B 4 bits** (`mlx-community/Qwen3-8B-4bit`, Apache-2.0), sin "modo pensar" | carga 2,1 s, **12-12,6 tokens/s**, pico 4,7 GB (Mac en swap por los otros trabajos) |
| VLM | **Qwen3-VL-8B-Instruct 4 bits** (`mlx-community/Qwen3-VL-8B-Instruct-4bit`, Apache-2.0) | ver tabla de evaluación (tiempo de verificación por imagen) |

Ejemplo real del plan (Qwen3-8B, historia "el cóndor vive en los Apus nevados"): elementos
`condor, snowy mountain, village`; escena "A large condor with a white collar and bare head soars above
a snowy mountain range, with a small village visible below"; nota cultural "Apus refers to sacred snowy
mountains in Andean culture". Es fiel a la historia y la iconografía es correcta (collar blanco, cabeza
calva).

**Comparación pendiente (no se pudo terminar):** estaban preparadas (`tools/art_bench_llm.py`,
`tools/art_bench_vlm.py`) las comparaciones Qwen3-8B vs Qwen2.5-7B-Instruct vs Qwen3.5-9B (LLM) y
Qwen3-VL-8B vs Qwen3.5-9B vs Qwen2.5-VL-7B (VLM, con elementos presentes Y ausentes para medir falsos
positivos). Los tres LLM y los tres VLM están descargados. En la Mac compartida por seis trabajos a la vez,
el swap llegó a 18 GB y un solo plan tardó 5,7 horas en un intento; se priorizó la evaluación de punta a
punta. Correr cuando la Mac esté libre:

```bash
.venvs/art/bin/python -m yq.common.heavylock .venvs/art/bin/python tools/art_bench_llm.py \
    --llms qwen3-8b qwen2.5-7b qwen3.5-9b --out /tmp/llm_bench.json
.venvs/art/bin/python -m yq.common.heavylock .venvs/art/bin/python tools/art_bench_image.py --out /tmp/art_bench
.venvs/art/bin/python -m yq.common.heavylock .venvs/art/bin/python tools/art_bench_vlm.py \
    --vlms qwen3-vl-8b qwen3.5-9b qwen2.5-vl-7b --dir /tmp/art_bench --out /tmp/vlm_bench.json
```

Idea a probar: **Qwen3.5-9B** es multimodal nativo (texto + imagen, Apache-2.0, 6 GB). Si planifica tan
bien como Qwen3-8B y revisa tan bien como Qwen3-VL-8B, un solo modelo haría las dos cosas
(`YQ_ART_LLM=qwen3.5-9b YQ_ART_VLM=qwen3.5-9b`: `llm.py` lo detecta y comparte los pesos) y la Mac
cargaría un modelo menos en cada dibujo.

### Memoria en la Mac

Presupuesto del gestor de modelos: 10,5 GB entre todos los dominios. Un dibujo usa, uno después de otro:
LLM 5,2 GB → generador 5,4 GB → VLM 6,6 GB (→ generador otra vez si hay reintento). El gestor
descarga el menos usado cuando hace falta, así que cada cambio cuesta unos segundos de carga desde el SSD.
