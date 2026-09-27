# Lengua de señas en Yachachiq v2 (dominio SIGN)

El visitante cuenta su historia en **Lengua de Señas Peruana (LSP)**, **ASL** o **Señas
Internacionales**. La cámara ve su cuerpo y sus manos, el robot reconoce **letras deletreadas**
y **palabras**, muestra las opciones más probables y **el visitante confirma** tocando la
pantalla. El texto confirmado es la historia que se dibuja.

## 1. Cómo funciona (en corto)

```
cámara (Arducam OV9782, 1280x800 MJPG)
  -> RTMW (pose de cuerpo entero, 133 puntos, en la GPU del Jetson)
  -> filtro One Euro (quita el temblor de los puntos)
  -> 69 puntos útiles: 2 manos x 21, hombros/codos/muñecas/caderas, cara (cejas, nariz, labios)
  -> LETRAS: clasificador de la forma de la mano + detector de movimiento (J, Z, Ñ)
             + "sostén la letra" + autocompletar palabras con diccionario
  -> PALABRAS: detector de inicio/fin de seña + red transformer sobre la secuencia
  -> candidatos en pantalla -> el visitante elige -> texto
```

Cambios frente a v1: ya no usamos MediaPipe (sus puntos temblaban); los puntos se dibujan
sobre **el mismo cuadro** del que salieron (en v1 se mandaban dos cuadros distintos y por eso
"parpadeaban"); la R y la I/J ya no se confunden en la validación (ver números).

## 2. Qué reconoce y qué tan bien (medido)

### Letras

Modelo: red pequeña (MLP) sobre 90 números geométricos de la mano en 2D (x, y), normalizada
(muñeca al centro, tamaño de la palma = 1, mano izquierda espejada a derecha).

**Cómo medimos** (sin trampa): validación cruzada de 5 partes. Cada "toma" de grabación de
Ignacio (100 cuadros seguidos de una pulsación de tecla en v1) queda completa en un lado, y
cada **persona** de ASL-HG también: el acierto de ASL-HG es con **personas que el modelo nunca
vio** en esa parte.

**Paso 1: solo con los datos de Ignacio (1 persona).**

| Modelo | Datos | Acierto por cuadro | Top-3 | v1 (ExtraTrees 3D, mismas particiones) |
|---|---|---|---|---|
| LSP | 12 260 cuadros, 123 tomas, 26 letras + ESPACIO + BORRAR | 95,0 % | 99,97 % | 95,5 % |
| ASL | 12 500 cuadros, 126 tomas | 97,3 % | 99,4 % | 96,7 % |

El modelo nuevo usa solo x, y (lo que da RTMW) y aun así iguala o supera al de v1 (x, y, z).
Pero probado con fotos de **10 personas nuevas** (ASL-HG) solo acertaba ~59 % (tabla de abajo):
un modelo de una sola persona no sirve para visitantes.

**Paso 2 (modelos actuales): Ignacio + 10 personas de ASL-HG (CC BY 4.0).**

| Modelo (`yq/sign/models/`) | Datos | Personas NUEVAS (ASL-HG, top-1 / top-3) | Tomas nuevas de Ignacio (top-1 / top-3) |
|---|---|---|---|
| **LSP** `letters_prl` | 37 164 cuadros (sin la U de ASL-HG) | **95,4 % / 99,8 %** | **99,3 % / 99,96 %** |
| **ASL** `letters_ase` | 38 404 cuadros | **94,1 % / 99,6 %** | **98,0 % / 99,4 %** |

- Letras más débiles: M y N (se confunden entre sí: LSP N→M 13 %, M→N 10 %), S (82 %),
  I (92 %). Son formas de puño muy parecidas en 2D.
- **R vs U/V**: LSP 0 %; ASL R→U 7,5 %. **I vs J**: 2–11 % por cuadro, pero en vivo las
  separa el detector de movimiento (misma forma de mano), no el clasificador.
- Las K, R y T de LSP (antes 71–78 %) mejoraron al sumar manos de otras personas.
- Con ruido extra de 0,05 palmas el acierto total baja solo ~1 punto (LSP 96,7 → 95,8 %).
- Para LSP se usaron fotos de ASL de todas las letras menos la U (según la guía MINEDU son
  iguales; ver límites). Top-3 ≈ 99,8 %: casi siempre la letra correcta está entre las 3
  que ve el visitante.

**Cambio de MediaPipe a RTM** (modelo del paso 1, 957 fotos de 10 personas, letras quietas):

| Puntos de la mano | Acierto (top-1) | Top-3 |
|---|---|---|
| MediaPipe (como v1) | 59,1 % | 70,7 % |
| RTMPose-hand (el afinador de dedos del robot) | 59,6 % | 69,1 % |
| RTMW-m cuerpo entero (fotos que solo muestran la mano: caso pesimista) | 53,8 % | 65,4 % |

Conclusión: **cambiar MediaPipe por RTM casi no cuesta precisión** (0 a −5 puntos); lo que
más importa es entrenar con **muchas personas**.

### Diferencia MediaPipe ↔ RTM (medida)

En 1 037 fotos de manos de 10 personas, distancia media entre los puntos de MediaPipe y los
de RTM, en "palmas" (1 palma ≈ 8 cm en un adulto):
RTMPose-hand 0,114 (puntas de dedos 0,139); RTMW-m 0,134 (mediana 0,097; puntas 0,174).
Si a los datos de prueba se les suma ruido de 0,05 palmas el acierto baja de 95,0 % a 94,8 %;
con 0,10 palmas baja a 90,9 %. Por eso se entrena con ruido y giros aleatorios.

### Letras con movimiento

- ASL: **J** (forma de I + el meñique dibuja una J) y **Z** (el índice dibuja una Z).
- LSP (guía del MINEDU, págs. 69–70; ver `lsp/GUIA_LSP.md`): **J**, **Z** y **Ñ** (= N moviendo
  la mano de lado a lado).
- Se detectan mirando la trayectoria de ~1 s: la mano debe mantener la forma base, la muñeca
  debe moverse y el dibujo debe tener la forma (Z = 2 esquinas; Ñ = ida y vuelta).
- **Precisión real: no medida todavía** (no hay grabaciones de J/Z/Ñ con RTMW). En secuencias
  sintéticas funciona; hay que grabar J, Z y Ñ reales y ajustar los umbrales
  (`yq/sign/letters.py`, `MotionTracker`).

### Palabras

- El programa completo está listo y probado (grabar → entrenar → ONNX → reconocer).
- **LSP y Señas Internacionales: 0 palabras entrenadas**: no existe un dataset abierto que se
  pueda usar (revisamos PUCP-305, AEC, LSP10: licencia no clara o sin licencia). Hay que
  **grabarlas** con el grabador (sección 4).
- **ASL: 0 palabras entrenadas** porque falta la cuenta de Kaggle para bajar asl-signs (250
  señas). Con la cuenta, un comando lo entrena (ver `training/sign/README.md`).
- Validación con datos sintéticos (solo prueba que el flujo funciona, NO dice nada real):
  top-1 85 % con 8 clases y 5 "personas" separadas; paridad PyTorch↔ONNX 1,7e-6.

## 3. Velocidad (medida en el MacBook M4, 1280x800)

| Modelo | CPU | CoreML |
|---|---|---|
| RTMW-m 256x192 (por defecto) | 50 ms (20 fps) | **6,9 ms (146 fps)** |
| RTMW-l 256x192 | 89 ms (11 fps) | 15,4 ms (65 fps) |
| RTMW-l 384x288 | 178 ms (5,6 fps) | 26 ms (38 fps) |
| RTMPose-m mano (afinador) | 34 ms | 4,5 ms |
| YOLOX-tiny persona (solo cada 2 s) | 42 ms | 7,2 ms |
| **Todo el flujo** RTMW-m + afinador de mano | 84 ms (12 fps) | **10 ms (99 fps)** |

**Jetson Orin Nano Super (estimado, NO medido):** no existen medidas publicadas de RTMW en
Jetson. RTMW-m cuesta 4,3 GFLOPs (RTMW-l 7,9; RTMW-l-384 17,7). Tomando la medida oficial de
RTMPose-l (4,5 GFLOPs = 5,7 ms en TensorRT FP16 en una GTX 1660 Ti) y que el Orin Nano Super
tiene ~2,5–3 veces menos GPU y ancho de memoria, esperamos RTMW-m en **~12–18 ms**
(55–80 fps) y RTMW-l en ~20–28 ms. Con el afinador de mano y la cámara, **RTMW-m debería
superar 25 fps con margen**; por eso es el modelo por defecto. Medirlo en el Jetson:
`python tools/sign_benchmark.py --backends tensorrt cuda` y si RTMW-l da ≥ 30 fps, usar
`YQ_SIGN_POSE_MODEL=rtmw-l` (más preciso en manos: AP 59,8 vs 49,1).

## 4. Grabar señas nuevas (palabras o letras)

```bash
cd ~/yachachiq/v2
.venvs/sign/bin/python -m yq.sign.record --lang prl --mode words --signer ignacio --reps 5
```

- Aparece la cámara con tu esqueleto. Arriba dice la palabra (de `yq/sign/vocab_prl.txt`).
- **ESPACIO** = grabar (cuenta 3, 2, 1), **S** = saltar, **R** = borrar la última toma, **Q** = salir.
- Palabras: haz la seña completa y **baja las manos**; se corta sola.
- Letras: `--mode letters --letters krtqg` graba solo esas; mantén la letra quieta 2 s
  (J, Z, Ñ: haz el movimiento dos veces).
- Si algo sale mal (no se ven las manos, muy corto, sin movimiento) dice por qué y repite.
- Se guarda en `~/yq-data/models/sign/recordings/<lengua>/<modo>/<palabra>/`.
- **Consejos:** buena luz de frente, fondo liso, cuerpo de la cintura para arriba, ropa de
  color distinto a la piel. Graba a **varias personas** (≥ 5), en días distintos, un poco
  más cerca y más lejos. Es lo que más sube la precisión con visitantes.

## 5. Entrenar

```bash
.venvs/sign/bin/python -m yq.common.heavylock .venvs/sign/bin/python training/sign/train_words.py --lang prl
.venvs/sign/bin/python -m yq.common.heavylock .venvs/sign/bin/python training/sign/train_letters.py --lang prl --aslhg training/sign/data/aslhg_hands.npz
```
Imprime la precisión medida (sin repetir personas cuando hay ≥ 3). El modelo nuevo queda en
`~/yq-data/models/sign/` y se usa automáticamente. Detalles: `training/sign/README.md`.

## 6. Cómo confirma el visitante

- **Letras:** sostiene cada letra ~0,3 s quieta (barra verde en la vista previa). Para una
  letra doble (LL, RR) hace la letra, mueve un poquito la mano y la hace otra vez.
  Mientras deletrea, la pantalla muestra **palabras candidatas** ("CONDO…" → *cóndor*).
  Al bajar la mano 1,5 s (o con la seña ESPACIO = pulgar arriba) la palabra se cierra con la
  mejor opción y **las otras quedan en pantalla para cambiarla** con un toque.
  BORRAR = pulgar abajo, o el botón.
- **Palabras:** cada seña reconocida se agrega con su mejor opción y se muestran 5
  alternativas para cambiarla.
- Para la interfaz (dominio UI): `state()["candidates"]` = lista de `{"text", "prob"}`;
  `accept(i)` elige; `backspace()`, `clear()`, `text()`.

## 7. Instalar

### Mac
```bash
cd ~/yachachiq/v2
~/.local/bin/uv venv --python 3.10 .venvs/sign
~/.local/bin/uv pip install --python .venvs/sign/bin/python -r requirements-common.txt -r requirements/sign-mac.txt
# una sola OpenCV con ventanas:
~/.local/bin/uv pip uninstall --python .venvs/sign/bin/python opencv-python opencv-contrib-python-headless
~/.local/bin/uv pip install --python .venvs/sign/bin/python --reinstall opencv-contrib-python
.venvs/sign/bin/python -m yq.sign.modelstore download        # ~420 MB, una sola vez, con internet
```

### Jetson (JetPack 6.2)
```bash
sudo nvpmodel -m 2 && sudo jetson_clocks          # modo MAXN SUPER
cd ~/yachachiq/v2 && python3 -m venv .venvs/sign && . .venvs/sign/bin/activate
pip install -r requirements-common.txt
pip install onnxruntime-gpu==1.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
python -c "import onnxruntime as o; print(o.get_available_providers())"   # debe salir TensorrtExecutionProvider
python -m yq.sign.modelstore download               # o copia ~/yq-data/models/sign/pose desde el Mac
python tools/sign_benchmark.py --backends tensorrt cuda --models rtmw-m rtmw-l rtmpose-hand yolox-tiny
```
La primera vez TensorRT "compila" cada modelo (varios minutos) y lo guarda en
`~/yq-data/models/sign/trt_cache/`; después arranca rápido. Si cambias de versión de
onnxruntime o JetPack, borra esa carpeta.

Cámara: `YQ_SIGN_CAMERA=/dev/v4l/by-id/usb-Arducam...-video-index0` (ruta fija). La OV9782
da 1280x800 MJPG hasta 100 fps (no tiene 1080p).

## 8. Probar

```bash
.venvs/sign/bin/python -m pytest tests/sign                   # rápidas (sin cámara ni modelos grandes)
.venvs/sign/bin/python -m yq.common.heavylock .venvs/sign/bin/python -m pytest -m heavy tests/sign
YQ_MOCK_SIGN_CAMERA=1 .venvs/sign/bin/python tools/sign_live.py   # simulador: deletrea "condor sol"
.venvs/sign/bin/python tools/sign_live.py --lang prl              # con cámara de verdad
```
En el Mac, la Terminal necesita permiso de cámara (Ajustes → Privacidad → Cámara).

## 9. Ajustes (variables `YQ_...`, en `yq/sign/settings.py`)

`SIGN_POSE_MODEL` (rtmw-m), `SIGN_BACKEND` (auto), `SIGN_HOLD_S` (0,30 s),
`SIGN_LETTER_MIN_CONF` (0,55), `SIGN_AUTO_SPACE_S` (1,5 s), `SIGN_EURO_MIN_CUTOFF` (0,05) y
`SIGN_EURO_BETA` (80) del filtro (los mismos valores que usa MediaPipe para suavizar la pose), `SIGN_MIRROR` (vista previa como espejo).

## 10. Límites (honestos)

- **Traducir oraciones continuas en lengua de señas NO es confiable** con los modelos abiertos
  de hoy. Por eso el robot trabaja con letras y palabras sueltas y el visitante confirma.
- Las letras se entrenaron con Ignacio + 10 personas de fotos (ASL-HG). Las fotos son de
  manos recortadas y quietas, con puntos de MediaPipe; en el robot habrá cuerpo entero, otra
  cámara y puntos de RTMW: **la precisión real con visitantes hay que medirla** grabando a
  gente nueva con `yq.sign.record --mode letters`.
- Para LSP se usan fotos de ASL de las letras que la guía MINEDU muestra iguales (todas menos
  U y Ñ). Esa comparación la hizo el equipo con los dibujos de la guía: **conviene que un
  intérprete de LSP la revise**.
- J, Z y Ñ: detector por reglas, **sin medir con personas reales**.
- Palabras: hay que grabarlas (LSP, SI) o bajar Kaggle (ASL).
- Señas Internacionales: no hay modelo de letras (no hay datos verificados); solo palabras
  cuando se graben.
- El quechua del diccionario (`story_words.tsv`) lo escribió el equipo: que lo revise un hablante.

## 11. Pendiente (necesita datos, hardware o tiempo de entrenamiento)

| # | Qué falta | Quién / qué necesita | Comando exacto |
|---|---|---|---|
| 1 | Medir fps en el Jetson y elegir RTMW-m o RTMW-l | Jetson + JetPack 6.2 | `python tools/sign_benchmark.py --backends tensorrt cuda --models rtmw-m rtmw-l rtmpose-hand yolox-tiny` |
| 2 | Probar con la cámara Arducam (y con la webcam del Mac: la Terminal no tenía permiso de cámara) | cámara + permiso | `.venvs/sign/bin/python tools/sign_live.py --lang prl` |
| 3 | Grabar J, Z, Ñ reales y medir/ajustar el detector de movimiento | Ignacio + 2–3 amigos | `.venvs/sign/bin/python -m yq.sign.record --lang prl --mode letters --letters jzñ --signer NOMBRE --reps 5` |
| 4 | Medir letras con visitantes reales (RTMW, cuerpo entero) y sumar esas tomas al entrenamiento | ≥ 5 personas | `... -m yq.sign.record --lang prl --mode letters --signer NOMBRE --reps 3` y luego `training/sign/train_letters.py --lang prl --aslhg training/sign/data/aslhg_hands.npz` |
| 5 | Palabras LSP (60 palabras de `yq/sign/vocab_prl.txt`) | ≥ 5 personas × 5 repeticiones (~2 h) | `... -m yq.sign.record --lang prl --mode words --signer NOMBRE --reps 5` y `... -m yq.common.heavylock .venvs/sign/bin/python training/sign/train_words.py --lang prl` |
| 6 | Palabras ASL (250 señas de Kaggle asl-signs) | cuenta de Kaggle (`~/.kaggle/kaggle.json`) + ~1 h de entrenamiento | pasos en `training/sign/README.md`, sección 2 |
| 7 | Señas Internacionales: palabras (y letras, si se decide usar el alfabeto internacional) | grabaciones propias | igual que 5 con `--lang ils` |
| 8 | Revisión de un intérprete de LSP (letras de ASL-HG usadas para LSP; vocabulario) y de un hablante de quechua (`yq/sign/models/story_words.tsv`) | intérprete / hablante | — |
