# Licencias – BOX-ANALYSIS (caja de análisis)

Todo es software libre o datos abiertos (regla WRO FI 5.3). Nada necesita internet durante la
competencia: los modelos y el catálogo se descargan una vez en la Mac.

## Bibliotecas

| Paquete | Para qué | Licencia |
|---|---|---|
| numpy, Pillow | imágenes y matemáticas | BSD-3 / MIT-CMU (HPND) |
| opencv-contrib-python-headless 5.0 | ChArUco, calibración, filtros | Apache-2.0 |
| scipy | mínimos cuadrados, filtros | BSD-3 |
| scikit-image | marching cubes, superpíxeles SLIC | BSD-3 |
| trimesh + fast-simplification | malla 3D, GLB, decimación | MIT / MIT |
| torch, torchvision | ejecutar SigLIP 2 en la GPU de la Mac (MPS) | BSD-3 |
| transformers, safetensors, huggingface_hub | cargar SigLIP 2 | Apache-2.0 |
| requests | descargar el catálogo (solo desarrollo) | Apache-2.0 |
| pycolmap | evaluado, NO usado (su MVS necesita CUDA) | BSD-3 |

## Modelos

| Modelo | Uso | Licencia | Descarga |
|---|---|---|---|
| google/siglip2-base-patch16-384 | embeddings de imágenes (catálogo y consultas) | Apache-2.0 | 1,5 GB (Hugging Face) |
| google/siglip2-so400m-patch14-384 | alternativa más grande (evaluada/lista para usar) | Apache-2.0 | 4,5 GB |
| VLM de ART (Qwen3-VL-8B 4-bit, MLX) | razonamiento opcional sobre fotos + datos | Apache-2.0 (Qwen3-VL) — ver docs/licenses_art.md | lo descarga ART |

No usamos ningún modelo de segmentación aprendido: la caja es negra y la sustracción de fondo es
exacta; un modelo no mejoró medidas en las pruebas sintéticas y agregaría otra descarga.

## Datos del catálogo de museos (referencias para identificar)

| Fuente | Qué guardamos | Licencia |
|---|---|---|
| The Metropolitan Museum of Art – Collection API + MetObjects.csv (github.com/metmuseum/openaccess) | solo objetos `isPublicDomain = true`: metadatos + miniatura ≤384 px | CC0 (Open Access) |
| Art Institute of Chicago – API + IIIF | solo `is_public_domain = true`; metadatos CC0 (no guardamos el campo `description`, que es CC-BY) | CC0 |
| Cleveland Museum of Art – Open Access API | solo `share_license_status = CC0` | CC0 |
| Smithsonian Open Access (opcional, no usado aún) | necesita una clave gratuita de api.data.gov | CC0 |

Cada registro guarda su museo, URL y licencia; la interfaz muestra siempre de qué museo viene cada
objeto parecido. Nada tiene licencia no comercial (CC-BY-NC): no hay nada que marcar.

## Código propio

Todo lo demás (calibración, casco visual, fotoconsistencia, RTI/PTM, UV, termografía TSR/PPT,
identificación, contexto del visitante, renderizador de pruebas) es código del equipo.
