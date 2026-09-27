# Caja de análisis – BOX-ANALYSIS (análisis de los escaneos)

La parte de captura (BOX-CAPTURE) toma las fotos, el peso, las luces y el video térmico y los deja
en una carpeta `scans/<id>/` (CONTRACTS.md §4). Este módulo convierte esa carpeta en:
medidas con su margen de error, un modelo 3D, imágenes especiales (RTI, UV, térmica), hallazgos
en español sencillo y una identificación (tipo, material, cultura, época) con objetos parecidos de
museos. Todo funciona **sin internet**.

## 1. Qué le dice cada análisis a un arqueólogo

| Análisis | Qué hace | Qué se aprende |
|---|---|---|
| Peso | promedio de la celda de carga (±0,5 g como mínimo) | masa real del objeto |
| Fotogrametría (2 cámaras × 24 ángulos) | siluetas → casco visual → talla fina por fotoconsistencia | alto, ancho, profundidad, volumen exterior, modelo 3D a escala real |
| Densidad aparente | masa ÷ volumen exterior | macizo o hueco; si es macizo, qué materiales encajan (cerámica 1,6–2,5; piedra 2,3–3,4; metal > 7,5 g/cm³) |
| RTI (8 LED rasantes) | normales de la superficie (estéreo fotométrico) + PTM | incisiones, marcas de herramienta, inscripciones gastadas, huellas; se puede "mover la luz" en la pantalla |
| UV 365 nm | fluorescencia ÷ reflectancia visible, zonas por color | zonas que brillan distinto: posibles restauraciones, adhesivos, consolidantes o pinturas modernas |
| Termografía activa | la halógena calienta, la Lepton filma el enfriamiento (TSR + PPT) | zonas que se enfrían distinto: posibles grietas, huecos, despegues, inclusiones o reparaciones justo bajo la superficie |
| Identificación | comparación con ~decenas de miles de objetos de museos (CC0) + VLM opcional | cultura, época, material y tipo más probables, alternativas y objetos parecidos reales |

**Importante (densidad):** la densidad aparente usa el volumen de la *envolvente exterior*. En una
vasija, botella o figura hueca el aire de adentro cuenta como volumen, así que la densidad aparente
sale mucho menor que la del material. Solo en piezas macizas se parece a la del material.

Todos los hallazgos están redactados con cuidado ("posible…", "confirmar con un especialista"): son
pistas, no diagnósticos.

## 2. Instalar

En la Mac (una vez, con internet):

```
cd ~/yachachiq/v2
~/.local/bin/uv venv --python 3.10 .venvs/box_analysis
~/.local/bin/uv pip install --python .venvs/box_analysis/bin/python -r requirements-common.txt -r requirements/box_analysis-mac.txt
```

En el Jetson (análisis livianos): `-r requirements/box_analysis-jetson.txt` en lugar del archivo de la Mac.

El modelo SigLIP 2 se descarga solo la primera vez (1,5 GB en `~/.cache/huggingface`). Para asegurar
que nunca intente descargar en la competencia: `export YQ_BOX_ALLOW_DOWNLOAD=0` y `HF_HUB_OFFLINE=1`.

## 3. Usar

```
from yq.box.analysis import analyze_scan
r = analyze_scan("~/yq-data/scans/scan-20261105-101500-ab12")     # ScanResult
```

En la Mac el servidor `yq.macworker.app` registra los trabajos `scan_analyze` (recibe `scan.zip`)
e `identify` (recibe fotos). Los resultados y las imágenes quedan en la carpeta `out/` del trabajo
y el Jetson los descarga a `scans/<id>/analysis/`.

Archivos que produce (claves de `ScanResult.artifacts`): `model_glb` (modelo 3D en metros, Y hacia
arriba), `model_view0..3` (vistas), `rti_ptm` + `rti_normals`, `rti_albedo`, `rti_relief`,
`rti_curvature`, `rti_specular`, `rti_relight_000/090/180/270`, `uv_fluorescence`, `uv_overlay`,
`thermal_heating`, `thermal_anomaly`, `thermal_phase`, `thermal_overlay` (si hay registro térmico),
y en `similar/` las miniaturas de los objetos parecidos. `analysis.json` guarda todo el resultado.

## 4. Probar

```
cd ~/yachachiq/v2
.venvs/box_analysis/bin/python -m pytest tests/box_analysis            # pruebas rápidas (sintéticas)
.venvs/box_analysis/bin/python -m yq.common.heavylock .venvs/box_analysis/bin/python -m pytest -m heavy tests/box_analysis
```

Las pruebas crean objetos virtuales de tamaño conocido (cilindro, caja, elipsoide, placa con
incisiones) y los "fotografían" con cámaras virtuales calibradas, con las mismas luces que la caja.
Así sabemos la respuesta correcta y medimos el error de verdad.

## 5. Calibrar la caja (paso a paso, cuando llegue en noviembre)

Sin calibrar, todo funciona con la geometría nominal del CAD R1 (cámaras y LED), pero las medidas
pueden equivocarse varios milímetros y el resultado lo avisa. Calibrar toma ~30 minutos:

1. **Imprimir los tableros**: `python tools/box_analysis_calibrate.py target --out ~/Desktop/tableros`.
   Imprimir los PDF al **100 % (tamaño real)**. Medir la barra de 100 mm con una regla: si no mide
   100 mm, la impresora escaló la hoja; volver a imprimir. Pegar cada hoja sobre cartón plano.
2. **Cámaras (intrínsecos)**: con el tablero `charuco_intrinsics` en la mano, tomar 15–25 fotos con
   la cámara A dentro de la caja, inclinándolo en distintas direcciones y cubriendo todo el cuadro
   (enfoque bloqueado igual que en los escaneos). Luego:
   `python tools/box_analysis_calibrate.py intrinsics --camera A --images "fotos/intrA/*.jpg"`.
   Repetir con la cámara B. Un error RMS menor a 0,5 px es bueno.
3. **Plato giratorio (extrínsecos)**: poner el tablero `charuco_plato` plano sobre el plato
   (fuera del centro está bien) y hacer un escaneo de fotogrametría normal (12–24 ángulos, ambas
   cámaras). Copiar las fotos `camA_000.jpg…` a una carpeta y correr
   `python tools/box_analysis_calibrate.py turntable --folder fotos/plato`.
   Esto encuentra el eje del plato, la posición exacta de cada cámara y el sentido de giro.
4. **Luces RTI (opcional pero recomendado)**: una esfera cromada (rulemán de acero de 1" = radio
   12,7 mm) sobre el plato. En cada posición (3 posiciones, girando el plato 120°) una foto con la
   luz COB (`outline.jpg`) y una por LED (`led1.jpg … led8.jpg`) en `fotos/esfera/pos1/`, `pos2/`…
   `python tools/box_analysis_calibrate.py rti --folder fotos/esfera --radius 12.7`.
5. **Térmica ↔ visible (opcional)**: imprimir `termico_ajedrez`, calentarlo 20–30 s con la halógena,
   guardar un cuadro térmico (`frame.npy`) y la foto de la cámara A en el mismo ángulo:
   `python tools/box_analysis_calibrate.py thermal --thermal frame.npy --visible camA_000.jpg`.
6. `python tools/box_analysis_calibrate.py show` muestra qué está calibrado.

Todo se guarda como JSON en `~/yq-data/calibration/` (formato en `yq/box/analysis/calib.py`).

## 6. Formato web del RTI ("yq-ptm-1") para el visor de la UI

Archivos (en `analysis/`): `rti_ptm.json`, `rti_ptm_c012.png`, `rti_ptm_c345.png`,
`rti_albedo.png`, `rti_normals.png`. Todas las imágenes miden `width × height` del JSON (recorte del
objeto en la foto de la cámara RTI; `crop_xywh` dice de dónde salió).

* Coeficientes PTM de Malzbender (6 por píxel, sobre la luminancia):
  `a[k] = (byte / 255) * scale[k] + bias[k]`; `rti_ptm_c012.png` R,G,B = a0,a1,a2;
  `rti_ptm_c345.png` R,G,B = a3,a4,a5 (PNG de 8 bits, sin gamma).
* Luz: `(lu, lv)` = dirección de la luz proyectada en la imagen, `lu` a la derecha, `lv` hacia arriba,
  con `lu² + lv² ≤ 1`. Luminancia relativa:
  `L = a0·lu² + a1·lv² + a2·lu·lv + a3·lu + a4·lv + a5`.
* Color: `rgb_lineal = srgb_a_lineal(albedo.rgb) · max(L, 0) · gain`, luego volver a sRGB.
  El canal alfa del albedo es la máscara del objeto (0 = fondo).
* Alternativa con normales (mejor para luces muy rasantes): `rti_normals.png`, `n = rgb/255·2−1`
  (x derecha, y arriba, z hacia el observador); sombrear con Lambert/Blinn-Phong.
* `fit_lights` lista las direcciones (lu, lv) de los 8 LED usados en el ajuste: el PTM es confiable
  cerca de esas direcciones; con LED todos a la misma altura, conviene limitar el control a
  `0,5 ≤ |(lu,lv)| ≤ 0,95`.

## 7. Exactitud medida (pruebas sintéticas con respuesta conocida, 27-sep-2026)

Cómo se midió: objetos virtuales de tamaño exacto, "fotografiados" por trazado de rayos con cámaras
virtuales (posiciones del CAD R1, lente con distorsión, ruido de sensor, JPEG), fotos a 776×582
(1/6 de la resolución real de la IMX519: en la caja real el error de píxel será menor), 24 ángulos
por cámara, 2 cámaras, con la calibración conocida. Ver `tests/box_analysis/`.

| Medida | Cilindro Ø80×100 | Elipsoide 100×70×108 | Caja 100×60×80 (girada 20°) |
|---|---|---|---|
| Alto | 100,7 mm (+0,7 %) | 108,4 mm (+0,4 %) | 81,0 mm (+1,3 %) |
| Ancho | 80,2 mm (+0,3 %) | 100,0 mm (0,0 %) | 101,0 mm (+1,0 %) |
| Profundidad | 80,2 mm (+0,3 %) | 70,1 mm (+0,1 %) | 61,4 mm (+2,3 %) |
| Volumen exterior | 498,5 cm³ (−0,8 %) | 421,2 cm³ (−1,5 %) | 481,5 cm³ (+0,3 %) |

* Solo con siluetas (casco visual) el techo plano del cilindro salía 9,7 % más alto y la caja 12 %:
  por eso agregamos la talla por fotoconsistencia (compara parches de textura entre cámaras). Las
  caras planas lisas y sin textura no se pueden tallar: ahí el error puede volver a ser de varios mm
  (la incertidumbre que se informa lo tiene en cuenta).
* Las incertidumbres informadas (1 σ) cubren el error real en todos los casos medidos (a menos de 3 σ).
* Calibración del plato (tablero ChArUco en 12 ángulos): posición de cámaras recuperada con error
  < 0,05 mm y < 0,01°, error de reproyección 0,10 px. Intrínsecos desde 14 fotos: fx dentro del 1 %,
  centro óptico dentro de 4 px, distorsión k1 dentro de 0,03.
* RTI (placa con incisiones de 0,6 mm y un relieve, cámara B, 8 LED del CAD, sombras reales):
  error angular medio de las normales **1,0°** (mediana 0,4°, 90 % < 1,0°); en los píxeles de las
  incisiones y el relieve 4,8°. Sin el modelo 3D como apoyo el error medio sube a 3,1°.
  En objetos altos (más de ~7 cm) los LED quedan por debajo de la parte superior: allí el RTI no
  puede medir la normal y se usa la del modelo 3D (lo indica `rti_constrained_fraction`). En un
  elipsoide de 108 mm de alto visto por la cámara B solo el 2 % de los píxeles quedó bien iluminado;
  con las normales del modelo 3D el error medio fue 3,5° (mediana 2,6°).
* Termografía: defecto (hueco de aire a 1 mm bajo la superficie, radio 6 píxeles Lepton) encontrado
  a 1,4 px de su posición real con confianza 0,78, a pesar de calentamiento desparejo, cuadros
  repetidos y un salto de calibración interna (FFC) de 0,15 °C. Sin defecto: ninguna falsa alarma.
* UV: la zona fluorescente sintética se segmenta con IoU > 0,5 respecto de la verdad.

## 8. Identificación: cómo decide y cómo usa el lugar que dice el visitante

1. **Solo el objeto** (`image_only`): 4 fotos de la cámara A (0°, 90°, 180°, 270°) + 1 de la B,
   recortadas y puestas sobre fondo gris claro (como las fotos de museo) → SigLIP 2 → los 40 objetos
   más parecidos del catálogo → votos ponderados por similitud (softmax, temperatura τ) para cultura,
   material, tipo y región; la época sale de las fechas de los vecinos de esa cultura.
2. **Lugar del visitante** (CONTRACTS.md §9): se interpreta con el chip de región, un diccionario de
   sitios y valles del Perú (Chan Chan, Sipán, Nasca, Chavín, Cusco, Titicaca…), nombres de países,
   y el LLM de ART si está prendido. Regla exacta (`yq/box/analysis/context.py`):
   `m_c = 1,5 ^ (w · k_c)`, donde `w` = confianza del lugar (chip 1,0; sitio conocido 0,9; país 0,8;
   LLM 0,6) y `k_c` = +1 si la cultura es típica de esa zona, +0,5 zona vecina o mismo país, 0 no se
   sabe, −0,5 cultura de otra macro-región. Una cultura con menos del 25 % del voto de la mejor
   **nunca** sube (m ≤ 1): el lugar puede desempatar candidatos cercanos, pero no puede inventar una
   cultura que las fotos no apoyan. El efecto máximo es ×1,5 (y ×0,82 en contra).
3. **VLM** (opcional, de ART): recibe las fotos, las medidas, los 8 registros más parecidos y el lugar
   marcado como "lugar reportado por el visitante; puede ser incorrecto". Solo puede cambiar la
   cultura si su respuesta tiene al menos la mitad del voto de la ganadora; si nombra una cultura sin
   apoyo, se ignora y baja la confianza.
4. **Confianza**: la parte del voto de la cultura ganadora, convertida en "probabilidad de acertar"
   con una tabla medida en objetos del catálogo que el sistema no vio (`calibration.json`); +/−
   según coincidan retrieval y VLM.
5. `context_effect_es` dice en palabras simples qué hizo el lugar: "El lugar ayudó a decidir entre
   Moche y Chimú", "El lugar no cambió la respuesta", "El lugar no coincide con lo que se ve; se
   priorizó la imagen".
6. `similar` solo contiene registros reales del catálogo (con museo, URL y licencia CC0).

### Exactitud de la identificación (medida el 27-sep-2026 con el catálogo PARCIAL)

Cómo se midió ("dejar uno afuera"): se toma un objeto del catálogo como si fuera el objeto
escaneado, se lo saca del catálogo junto con todos los registros del mismo museo con el mismo
título (juegos, fragmentos, duplicados) y se mira si el sistema acierta su cultura, material y tipo.
Índice: SigLIP 2 base 384, **4 500 objetos** ya procesados (de 17 900 descargados en ese momento;
la descarga seguía), 2 000 objetos de prueba al azar, τ = 0,03, 40 vecinos.

| Qué | Top-1 | Top-3 |
|---|---|---|
| Cultura (todas) | 62,6 % | 78,6 % |
| Cultura (solo objetos andinos, n = 469) | 61,4 % | 80,4 % |
| Material exacto (oro, plata, cerámica, jade…) | 77,9 % | 92,0 % |
| Clase de material (cerámica / metal / piedra / textil…) | 84,3 % | 96,8 % |
| Tipo exacto (botella asa estribo, kero, figura…) | 64,6 % | 86,7 % |
| Clase de tipo (vasija / figura / adorno / textil…) | 80,6 % | 94,3 % |
| Región (Andes, Mesoamérica, Egipto…) | 76,8 % | 91,7 % |

Lugar del visitante (1 000 objetos de prueba, otra semilla, τ = 0,01): con el lugar **correcto**
66,5 % frente a 64,9 % sin lugar en esos mismos objetos (+1,6 puntos); con un lugar **equivocado**
64,0 % frente a 64,3 % sin lugar (−0,3 puntos).
En la corrida principal: sin lugar 62,6 %, correcto 64,1 %, equivocado 62,3 %. Es decir: ayuda un
poco cuando es cierto y casi no daña cuando es falso, que es lo que queríamos (el límite ×1,5 lo
garantiza).

La confianza que se muestra está calibrada con estas mismas pruebas (`calibration.json`): cuando el
sistema dice 80 %, acertó ~80 % de las veces en objetos que no había visto.

**Límites honestos:**
* Estas pruebas comparan fotos de museo con fotos de museo. Las fotos de la caja (fondo negro, luz
  distinta, cámaras propias) serán más difíciles: esa diferencia **no se pudo medir** sin la caja.
  Para acercarnos, las fotos de consulta se recortan y se ponen sobre fondo gris claro.
* Retrieval + VLM (el modelo de ART) no se pudo evaluar todavía: la Mac estaba ocupada por los otros
  equipos. Comando para hacerlo: ver la sección 9 (`eval --vlm-n 50`).
* Culturas con pocos objetos abiertos (Recuay, Vicús, Cupisnique, Chachapoyas) se confunden más con
  sus vecinas; por eso siempre se muestran alternativas y objetos parecidos para que la persona decida.

## 9. Catálogo de museos: construirlo, terminarlo y volver a medir

Todo va a `~/yq-data/catalog/` (variable `YQ_CATALOG_DIR`). Se puede cortar y volver a correr en
cualquier momento: sigue donde quedó (`state/done.tsv`).

```
cd ~/yachachiq/v2
# 1) descargar metadatos + miniaturas (horas; respeta los límites de cada museo)
nohup .venvs/box_analysis/bin/python tools/box_analysis_build_catalog.py --log ~/yq-data/catalog/download.log download --workers 4 &
#    para reintentar los que fallaron por red:  ... download --retry-errors
# 2) calcular los embeddings (usa el candado de memoria por tandas de 1 500 imágenes)
nohup .venvs/box_analysis/bin/python tools/box_analysis_build_catalog.py --log ~/yq-data/catalog/embed.log embed --model siglip2-base-384 &
# 3) ver cuántos objetos hay por museo, región, cultura, material
.venvs/box_analysis/bin/python tools/box_analysis_build_catalog.py status
# 4) volver a medir la exactitud (y rehacer la tabla de confianza)
.venvs/box_analysis/bin/python tools/box_analysis_build_catalog.py eval --model siglip2-base-384 --n 2000 --out ~/yq-data/catalog/eval.json
#    con el VLM de ART (dentro del candado de memoria):
.venvs/box_analysis/bin/python -m yq.common.heavylock .venvs/box_analysis/bin/python tools/box_analysis_build_catalog.py eval --n 300 --vlm-n 50
```

Fuentes y prioridad: The Met (primero África/Oceanía/Américas, luego Egipto, Cercano Oriente,
Grecia-Roma, Islam, Asia, Medieval…; filtrado con su CSV abierto `MetObjects.csv` a objetos 3D de
dominio público, ~76 000 candidatos), Art Institute of Chicago (Américas, África, Mediterráneo, Asia,
textiles, artes aplicadas) y Cleveland (13 departamentos). Se excluyen pinturas, grabados, dibujos,
fotos, libros y muebles: la caja nunca los va a ver. Tamaño medido: ~21 KB por objeto
(miniatura 384 px + metadatos; 18 151 objetos = 376 MB) → 100 000 objetos ≈ 2,1 GB, más 0,15 GB de
índice (base, 768 números por objeto) o 0,23 GB (so400m), más el CSV de The Met (317 MB, se puede
borrar al terminar).

Modelo: el índice se hizo con `siglip2-base-384` (rápido; la Mac estaba compartida). El modelo
grande `siglip2-so400m-384` ya está soportado (`--model siglip2-so400m-384`, 4,5 GB de descarga);
conviene medir los dos con `eval` y elegir el mejor con `YQ_BOX_EMBED_MODEL`.
Smithsonian Open Access se puede agregar (CC0) pero necesita una clave gratuita de api.data.gov.

## 10. Límites conocidos

* Casco visual: no ve concavidades (el interior de un cuenco, el hueco bajo un asa se rellena en
  parte). El volumen informado es el de la envolvente; la densidad aparente lo dice.
* Objetos negros brillantes sobre el plato negro: la silueta usa la foto del plato vacío; sin ella,
  la separación por brillo puede fallar (se avisa).
* Objetos transparentes o muy reflectantes (vidrio, metal pulido): siluetas y RTI poco confiables.
* RTI: los 8 LED están a la misma altura (~68 mm sobre el plato); funciona muy bien en piezas bajas
  (placas, tiestos, monedas, textiles) y en la parte baja de las altas; arriba de ~7 cm usa la forma 3D.
* UV y termografía miran una sola cara (la que ve la cámara en ese ángulo del plato).
* La termografía detecta lo que está a pocos milímetros de la superficie; bordes, pintura oscura y
  brillos pueden dar falsas alarmas (la confianza lo refleja).
* Nada de esto reemplaza a un especialista: la interfaz muestra alternativas y objetos parecidos
  para que la persona confirme.

## 11. Jetson, Mac y modo simulado

* **Mac** (`YQ_ROLE=mac`): todo. El servidor `python -m yq.macworker.app` carga este módulo solo
  (`routes_box.py`) y registra el modelo `box-siglip2-base-384` (~0,3 GB) en el administrador de
  memoria compartido; se carga cuando llega el primer trabajo `identify`/`scan_analyze`.
* **Jetson**: `analyze_scan` corre peso, volumen, modelo 3D, RTI, UV y termografía sin la Mac
  (más lento). La identificación necesita el catálogo y el modelo: si no están, el resultado lo
  dice ("La identificación se hace en la Mac") y no inventa nada.
* **Simulado**: `YQ_MOCK=1` o `YQ_MOCK_BOX_ANALYSIS=1` → la identificación devuelve un resultado
  marcado `engine = "mock"` sin cargar modelos. Para ver todo el flujo sin caja:
  `from yq.box.analysis.synthetic_scan import make_scan` crea un escaneo virtual completo.
