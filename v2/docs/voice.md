# Voz de Yachachiq (dominio `voice`)

El robot **escucha** la historia del visitante en el idioma que elija, la **escribe** (reconocimiento
de voz), la **traduce** (al español para la pantalla y la impresión, al inglés para dibujar) y puede
**hablar** (síntesis de voz) en ese idioma cuando existe una voz.

- Perú primero: español, quechua (chanka, cusqueño/collao, puneño, Áncash, huanca, Huánuco,
  Cajamarca, Lambayeque, San Martín, kichwa amazónico…), aimara, jaqaru y lenguas amazónicas
  (asháninka, awajún, shipibo-konibo, matsigenka, shawi, wampis, yanesha, kakataibo…).
- Luego el mundo: **1674 idiomas** se pueden reconocer, **1114** se pueden hablar y **598** traducir
  (números de `yq/common/data/languages.json`, ver “Idiomas” abajo).

## 1. Cómo funciona (resumen)

```
micrófono KAYSUDA ──► capture.py (Silero VAD: empieza y termina solo)
                         │ audio 16 kHz
                         ▼
                      asr.py ──► Mac /asr ──► Whisper large-v3-turbo (idiomas “fuertes” de Whisper)
                         │                └─► Omnilingual ASR de Meta (quechua, aimara, amazónicas, 1600 idiomas)
                         │   si la Mac no responde: Whisper large-v3-turbo en la GPU del Jetson
                         │   si tampoco: Whisper small en la CPU del Jetson
                         ▼
                   translate.py ──► Mac /translate (NLLB-200; MADLAD-400 para lo que NLLB no tiene)
                         ▼
                      tts.py ──► voz Piper en el Jetson (español, inglés…) o Mac /tts (MMS: quechua, aimara…)
```

- **Idioma elegido por el visitante** (lo mejor): el kiosco muestra `languages.ui_list("asr")`
  (español, quechuas, aimara, amazónicas primero, luego el mundo) y manda el código (`quy_Latn`…).
- **Idioma automático** (`lang="auto"`): la Mac usa la detección de Whisper para sus idiomas fuertes
  y **MMS-LID** (4017 idiomas) para el resto; si ninguna está segura, usa Omnilingual CTC, que
  transcribe sin saber el idioma. El resultado trae `lang_candidates` para que el visitante confirme.
- Siempre hay **confianza** (0 a 1) por historia y por segmento: si es baja, la pantalla debe pedir
  al visitante que revise o corrija el texto.

## 2. Idiomas

`yq/common/data/languages.json` se genera con `tools/voice_build_languages.py` desde las listas
oficiales de cada motor (Whisper, Omnilingual ASR, MMS-TTS, MMS-LID, NLLB-200, MADLAD-400, Piper)
más ISO 639-3, Glottolog y CLDR para los nombres. Cada idioma dice **qué motor puede qué**:

```python
from yq.common import languages
l = languages.get("quy")          # también sirve "quy_Latn", "qu", "es", "spa", "es-PE"...
l.name_es, l.asr, l.tts, l.translate      # 'Quechua ayacuchano (chanka)', ['omniasr'], ['mms'], True
languages.search("aimara")        # sin importar tildes ni mayúsculas
languages.ui_list("tts")          # lista para el selector del kiosco
```

`popular`: 1 = primero en el selector (español, luego las lenguas del Perú, luego las más habladas
del mundo); 0 = sin ranking (van después, en orden alfabético). Con `peru=True` el kiosco puede
mostrar un grupo “Perú” y otro “Mundo”.

Calidad publicada por los autores (no medida por nosotros) en `l.quality`:
`whisper_large_v3_fleurs_wer` (gráfico oficial de OpenAI, FLEURS) y `omniasr_7b_cer` (tabla de Meta,
modelo 7B, en sus propios datos de prueba, en su mayoría lectura de textos). **Ojo:** Meta reporta
CER muy bajos (0.2–3 %) para quechuas y amazónicas, pero son lecturas limpias parecidas a su
entrenamiento. Nuestras mediciones con voz real (sección 5) son más duras y son las que valen.

## 3. Instalar

### En la MacBook (M4, macOS 15, sin Homebrew)

```bash
cd ~/yachachiq/v2
~/.local/bin/uv venv --python 3.10 .venvs/voice
~/.local/bin/uv pip install --python .venvs/voice/bin/python -r requirements-common.txt -r requirements/voice-mac.txt
.venvs/voice/bin/python tools/voice_download.py --mac      # ~22 GB, una sola vez, con internet
```

- **libsndfile sin Homebrew:** `fairseq2` (lo que usa Omnilingual) viene compilado buscando
  `/opt/homebrew/.../libsndfile.1.dylib`. No hace falta instalar nada: el código arranca Omnilingual
  en un proceso aparte con `DYLD_FALLBACK_LIBRARY_PATH` apuntando a la libsndfile que ya trae el
  paquete `soundfile` (enlace en `~/yq-data/cache/voice_libs/`). Además, al descargar ese proceso
  se libera de verdad toda su memoria.
- Los modelos quedan en `~/.cache/huggingface` y `~/.cache/fairseq2/assets`; en la competencia no se
  usa internet.
- El servidor de la Mac (`python -m yq.macworker.app`) encuentra solo `routes_voice.py`. Para que
  cargue los modelos de voz, arráncalo con el Python de `.venvs/voice` (o uno que tenga
  `requirements/voice-mac.txt`).

### En el Jetson (Orin Nano Super 8 GB, JetPack 6)

```bash
sudo apt install -y libportaudio2 libsndfile1 alsa-utils
cd ~/yachachiq/v2
python3 -m venv .venvs/voice            # o uv, igual que en la Mac
.venvs/voice/bin/pip install -r requirements-common.txt -r requirements/voice-jetson.txt
.venvs/voice/bin/python tools/voice_download.py --jetson   # voces Piper + Whisper turbo/small (~2.5 GB)
```

**GPU (Whisper large-v3-turbo en CUDA) — lo que se sabe y lo que falta probar:**
- `ctranslate2` de PyPI en aarch64 **no trae CUDA** (solo CPU). Con eso el respaldo local funciona,
  pero en CPU (modelo `small`, más lento y menos preciso).
- El índice de NVIDIA Jetson AI Lab (`https://pypi.jetson-ai-lab.io/jp6/cu126`) tiene torch,
  torchaudio y onnxruntime-gpu para cu126, pero **no** ctranslate2. `jp6/cu128` trae ctranslate2
  4.5.0 y `jp6/cu129` trae 4.5.0/4.6.0 y `faster_whisper 1.1.1.post1`; están compilados para CUDA
  12.8/12.9 y **no está verificado** que funcionen con la CUDA 12.6 de JetPack 6.2.
- Camino más seguro (a probar cuando llegue el Jetson): el contenedor de `dusty-nv/jetson-containers`
  `jetson-containers run $(autotag faster-whisper)` (compila ctranslate2 con `-DWITH_CUDA=ON`), o
  compilar ctranslate2 igual que ese contenedor. Prueba: `python -c "import ctranslate2;
  print(ctranslate2.get_cuda_device_count())"` debe dar 1; entonces `asr.py` usa la GPU solo.
- Si el Jetson queda solo con CPU no pasa nada grave: la Mac hace el trabajo fino y el Jetson es respaldo.

## 4. Probar

```bash
cd ~/yachachiq/v2
.venvs/voice/bin/python -m pytest tests/voice                 # pruebas rápidas (sin modelos), ~20 s
.venvs/voice/bin/python -m yq.common.heavylock \
    .venvs/voice/bin/python -m pytest -m heavy tests/voice     # modelos reales: say de macOS -> Whisper/Omnilingual
```

**Probar el micrófono (en el Jetson o en la Mac):**

```bash
.venvs/voice/bin/python -m yq.voice.capture --list    # lista dispositivos; marca el micrófono elegido
.venvs/voice/bin/python -m yq.voice.capture --test    # habla; se detiene solo al callarte y guarda test.wav
aplay test.wav                                        # (Jetson) escucha lo grabado
```

- El micrófono se elige por nombre (`YQ_MIC_NAMES`, por defecto KAYSUDA, SPEAKPHONE, SP300, SP200,
  Speakerphone, USB Audio). En Linux un speakerphone USB aparece como
  `"<nombre del producto>: USB Audio (hw:N,0)"`; `arecord -l` muestra el nombre exacto. Si no
  coincide ninguno, se usa el micrófono por defecto.
- Si corta muy pronto: sube `YQ_VAD_SILENCE_S` (1.8 s por defecto). Si no arranca con voz baja:
  baja `YQ_VAD_THRESHOLD` (0.5). Si el eco del parlante lo activa: el micrófono ya se silencia
  mientras el robot habla (y 0.35 s después).
- Sin hardware: `YQ_MOCK_MIC=1`, `YQ_MOCK_SPEAKER=1`, `YQ_MOCK_ASR=1` o `YQ_MOCK=1` (todo simulado).

**Probar la Mac desde el Jetson:**

```bash
curl -F audio=@test.wav -F lang=auto http://192.168.8.20:8700/asr
curl -H 'Content-Type: application/json' -d '{"text":"Hola","src":"spa_Latn","tgt":"quy_Latn"}' http://192.168.8.20:8700/translate
curl -H 'Content-Type: application/json' -d '{"text":"Allinllachu","lang":"quy_Latn"}' http://192.168.8.20:8700/tts -o hola.wav
```

## 5. Precisión medida (MacBook M4 16 GB)

Cómo se midió: `tools/voice_eval_data.py` prepara frases de prueba públicas; `tools/voice_eval.py`
las pasa por cada motor **diciéndole el idioma** (como en el kiosco, donde el visitante lo elige),
normaliza el texto (minúsculas, sin puntuación, con tildes) y calcula **WER** (palabras mal, %) y
**CER** (letras mal, %). **RTF** = segundos de cómputo / segundos de audio (0.1 = 10 veces más rápido
que hablar), sin contar la carga del modelo. Memoria pico: MLX (Whisper) o proceso hijo RSS + MPS
(Omnilingual). La Mac estaba compartida con otros 5 ingenieros: las velocidades son pesimistas.
Regenerar las tablas: `.venvs/voice/bin/python tools/voice_report.py`.

### 5.1 Reconocimiento de voz (WER/CER)

<!-- EVAL:ASR -->
| Datos | Motor | WER % | CER % | RTF | Memoria pico | Carga |
|---|---|---|---|---|---|---|
| FLEURS español (es-419) (30 frases, 6 min) | whisper-large-v3-turbo | 2.4 | 0.9 | 0.091 | 2.1 GB | 2 s |
| FLEURS español (es-419) (30 frases, 6 min) | whisper-large-v3 | 2.5 | 0.9 | 0.227 | 4.7 GB | 2 s |
| FLEURS español (es-419) (30 frases, 6 min) | omniasr-ctc-1b | 4.9 | 1.4 | 0.051 | 4.6 GB | 7 s |
| FLEURS inglés (en-US) (30 frases, 5 min) | whisper-large-v3 | 5.7 | 2.5 | 0.258 | 4.7 GB | 2 s |
| FLEURS inglés (en-US) (30 frases, 5 min) | whisper-large-v3-turbo | 5.9 | 3.0 | 0.099 | 2.1 GB | 2 s |
| FLEURS inglés (en-US) (30 frases, 5 min) | omniasr-ctc-1b | 11.0 | 4.0 | 0.052 | 4.6 GB | 7 s |
| FLEURS portugués (pt-BR) (30 frases, 7 min) | whisper-large-v3 | 3.8 | 1.1 | 0.231 | 4.7 GB | 2 s |
| FLEURS portugués (pt-BR) (30 frases, 7 min) | whisper-large-v3-turbo | 3.9 | 1.2 | 0.075 | 2.1 GB | 2 s |
| FLEURS portugués (pt-BR) (30 frases, 7 min) | omniasr-ctc-1b | 6.9 | 2.1 | 0.047 | 4.6 GB | 7 s |
| Common Voice quechua de Puno (qxp) (30 frases, 2 min) | omniasr-ctc-1b | 32.5 | 5.0 | 0.070 | 4.6 GB | 7 s |
| Quechua chanka (quy), lectura (30 frases, 4 min) | omniasr-ctc-1b | 27.9 | 4.9 | 0.055 | 4.6 GB | 7 s |

_Archivo: `~/yq-data/eval/voice/asr_20260927_0942.json` (2026-09-27 09:42)._
<!-- /EVAL:ASR -->

**Qué significan estos números**
- Español, inglés y portugués: Whisper **turbo** acierta igual que **large-v3** (2.4 vs 2.5 % de
  palabras mal en español) pero es 2.5 veces más rápido y usa la mitad de memoria → es el modelo por
  defecto (`YQ_MAC_WHISPER_MODEL`). Omnilingual también funciona en español (4.9 %) y queda de respaldo.
- Quechua: Omnilingual CTC 1B se equivoca en ~5 de cada 100 **letras** (CER 5 %). El WER (~30 %)
  se ve alto porque en quechua una palabra es larga (sufijos) y una sola letra distinta la cuenta
  entera como error; además mucho “error” es ortografía (urqu/urku, kayri/kayriy). Ejemplo real:
  referencia “Haqay urqupin tarukaqa kashan”, el robot escribió “haqay urkupin taruka kashan”.
- Por eso, en quechua y aimara la pantalla **siempre** debe mostrar el texto para que el visitante
  lo corrija antes de dibujar.

### 5.2 Detección automática del idioma (`lang="auto"`)

<!-- EVAL:LID -->
_(sin medición de detección automática)_
<!-- /EVAL:LID -->

### 5.3 Traducción (FLORES-200 devtest, chrF++: más alto = mejor)

<!-- EVAL:MT -->
_(pendiente: correr tools/voice_eval_mt.py)_
<!-- /EVAL:MT -->

Primera corrida (solo NLLB-200 distilled 1.3B int8, 100 frases por par, 2026-09-27; se cortó antes de
MADLAD porque la Mac estaba saturada): español→quechua **26.3**, quechua→español **32.4**,
español→aimara **29.2**, aimara→español **30.2**, español→inglés **57.5**, inglés→español **54.6**
(chrF++), ~1.0–1.3 s por frase con la Mac tranquila.

Comparación con lo publicado por Meta para NLLB-200 distilled 1.3B (sus métricas oficiales en
FLORES-200 devtest, chrF++): español→quechua 25.4, español→aimara 28.4, español→inglés 58.3,
inglés→español 53.6. Nuestra versión int8 (CTranslate2) da números muy parecidos, así que la
cuantización no le quita calidad. Un chrF++ de ~26–32 (quechua, aimara) significa que la
traducción **conserva la idea pero con errores**: por eso el visitante siempre ve el texto original
y la traducción es solo una ayuda.

### 5.4 Lo que falta medir (honesto)

- **Detección automática del idioma** con modelos reales (Whisper + MMS-LID): el código y las
  reglas están probados con datos falsos, pero la precisión real aún no está medida.
- **Omnilingual LLM 1B** en quechua (solo probado en una frase en español: perfecta, pero ~20 veces
  más lento que CTC). **Omnilingual CTC 300M** tampoco está medido.
- **MADLAD-400** medido en serio (solo una prueba de 5 frases: en quechua→español y aimara→español
  inventó frases sin relación; por eso NLLB es el traductor principal).
- **Aimara y lenguas amazónicas**: no se encontró ningún conjunto de prueba abierto con audio y texto
  (Common Voice no tiene aimara). Solo existe la cifra publicada por Meta (`l.quality`).
- **Habla real de visitantes** (espontánea, con ruido de feria): todo lo medido es lectura en voz alta.
  Lo ideal: grabar ~50 frases de hablantes reales de quechua y aimara (con permiso) y medir.
- **Todo el lado Jetson** (micrófono KAYSUDA, cancelación de eco, Piper en aarch64, Whisper en CUDA).

Hay una tanda pequeña preparada que mide MT (15 frases), LLM 1B + detección de idioma (15 frases
por conjunto) y corre las pruebas “heavy”; al terminar actualiza solas las tablas de arriba
(`tools/voice_report.py`). Log: `~/yq-data/logs/voice_final_batch.log`.

### 5.5 Pruebas cortas hechas a mano (MacBook M4)

| Prueba | Resultado |
|---|---|
| Omnilingual CTC 1B v2, frase en español dicha por `say` (3.4 s) | texto correcto salvo la tilde de “cóndor”; 0.16 s (RTF 0.05); 3.7 GB; carga 5.5 s |
| Omnilingual LLM 1B v2, misma frase | texto perfecto; 4.1 s con idioma dado (RTF 1.2, incluye el cálculo de confianza); 5.9 GB; carga 29 s |
| faster-whisper `small` int8 en CPU (igual que el respaldo del Jetson) | texto perfecto, idioma detectado 99.7 % español; 4.0 s para 3.4 s de audio |
| Piper `es_MX-claude-high` (CPU) | 4.0 s de voz en 1.1 s |
| MMS-TTS quechua / aimara / español (CPU) | 4.6 s de voz en 4.6 s / 3.2 s en 3.7 s / 2.5 s en 1.6 s |
| Detector de voz Silero con voz sintética | silencio máx. 0.01, voz promedio 0.87 (umbral 0.5) |

## 6. Límites conocidos

- **Quechua y aimara no los entiende Whisper**: para ellos se necesita la Mac (Omnilingual). Si la
  Mac está apagada, el Jetson intenta con Whisper y marca confianza baja: el visitante debe corregir.
- Omnilingual acepta hasta 40 s por pedazo: las historias largas se cortan solas en el silencio más
  cercano a 38 s y se unen.
- La traducción al español de lenguas amazónicas: NLLB no las tiene; MADLAD sí tiene awajún y
  shipibo, pero su calidad no está medida (no hay datos de prueba abiertos). Asháninka, matsigenka
  y jaqaru **no se pueden traducir** hoy: la pantalla muestra el texto original.
- “Llama” (animal) a veces se traduce al inglés como “flame” (fuego): el paso de ART que planifica
  la imagen con el LLM debe revisar el sentido.
- Las voces MMS (quechua, aimara…) suenan robóticas y son CC-BY-NC; las voces Piper suenan mejor.
- Números: los modelos escriben “3” o “tres” según el caso; en la medición cuenta como error.
- Varios pesos son **no comerciales** (NLLB, MMS, algunas voces Piper): ver `licenses_voice.md`.

## 7. Archivos

| Archivo | Qué hace |
|---|---|
| `yq/common/languages.py`, `yq/common/data/languages.json` | lista de idiomas y qué motor sirve |
| `yq/voice/capture.py`, `vad.py`, `audio.py` | micrófono, detección de voz, utilidades de audio |
| `yq/voice/asr.py`, `translate.py`, `tts.py`, `settings.py` | lo que usa el kiosco en el Jetson |
| `yq/macworker/routes_voice.py` | `/asr`, `/lid`, `/translate`, `/tts`, `/voice/info` en la Mac |
| `yq/macworker/models/voice_*.py` | Whisper (MLX), Omnilingual (proceso aparte), MMS-LID, NLLB/MADLAD, MMS-TTS, reglas de ruteo |
| `tools/voice_build_languages.py` | reconstruye la lista de idiomas |
| `tools/voice_download.py` | baja todos los modelos (Mac o Jetson) |
| `tools/voice_eval_data.py`, `voice_eval.py`, `voice_eval_mt.py` | preparan datos y miden WER/CER/velocidad y traducción |

Ajustes (variables `YQ_...`, ver `yq/voice/settings.py`): `MAC_WHISPER_MODEL`, `MAC_OMNI_MODEL`,
`MAC_MMS_LID`, `LID_WHISPER_MIN`, `LID_MMS_MIN`, `ASR_JETSON_MODEL`, `MIC_NAMES`, `VAD_*`,
`PIPER_VOICE_OVERRIDES`, `YQ_MT_PREFERENCE`.
