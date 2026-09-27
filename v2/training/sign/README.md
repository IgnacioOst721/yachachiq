# Entrenamiento de señas (dominio SIGN)

Todo se corre desde `v2/` con el venv `.venvs/sign`. Lo pesado va dentro del candado de
memoria: `.venvs/sign/bin/python -m yq.common.heavylock <comando>`.
Carpetas `data/` y `runs/` están en `.gitignore` (datos bajados y registros).

| Script | Qué hace |
|---|---|
| `letters_data.py` | Lee las grabaciones v1 (CSV de MediaPipe) y las pasa a manos 2D canónicas |
| `train_letters.py` | Entrena el clasificador de letras (MLP), mide con validación cruzada honesta, exporta `.npz` + `.onnx` |
| `build_lexicon.py` | Baja las listas de frecuencia (inglés, castellano) y arma los léxicos |
| `mp_extract.py` | (venv aparte con MediaPipe 0.10.21) puntos de MediaPipe para las fotos ASL-HG |
| `aslhg_keypoints.py` | Manos canónicas de las fotos ASL-HG (10 personas) para entrenar letras |
| `domain_gap.py` | Compara MediaPipe vs RTMPose-hand vs RTMW-m en las mismas fotos |
| `patch_prototypes.py` | Actualiza las "manos ejemplo" del modelo (las usa el simulador) |
| `word_model.py` | Red para palabras (convoluciones + transformer sobre keypoints) |
| `words_data.py` | Datos de palabras: tus grabaciones, Kaggle asl-signs o sintéticos |
| `train_words.py` | Entrena palabras, valida SIN repetir personas, exporta ONNX y prueba paridad |

## 1. Letras (ya entrenadas)

```bash
# datos de 10 personas más (ASL-HG, CC BY 4.0, ~375 MB)
mkdir -p training/sign/data/asl_hg
curl -L -o training/sign/data/asl_hg/train.parquet https://huggingface.co/datasets/juanjodurillo/asl-hg/resolve/main/data/train-00000-of-00001.parquet
curl -L -o training/sign/data/asl_hg/test.parquet  https://huggingface.co/datasets/juanjodurillo/asl-hg/resolve/main/data/test-00000-of-00001.parquet
uv venv --python 3.10 .venvs/sign-mediapipe
uv pip install --python .venvs/sign-mediapipe/bin/python "mediapipe==0.10.21" pyarrow pillow
.venvs/sign-mediapipe/bin/python training/sign/mp_extract.py --all          # ~20 min, puntos MediaPipe
.venvs/sign/bin/python training/sign/aslhg_keypoints.py                    # segundos: manos canónicas
# (opcional, lento en un Mac ocupado: --source rtmpose usa RTMPose-hand en vez de MediaPipe)
# entrenar
.venvs/sign/bin/python -m yq.common.heavylock .venvs/sign/bin/python training/sign/train_letters.py --lang prl --aslhg training/sign/data/aslhg_hands.npz
.venvs/sign/bin/python -m yq.common.heavylock .venvs/sign/bin/python training/sign/train_letters.py --lang ase --aslhg training/sign/data/aslhg_hands.npz
```

Sin ASL-HG también funciona (solo con los datos de Ignacio): quita `--aslhg ...`.
Si grabas letras nuevas con `python -m yq.sign.record --mode letters`, el entrenamiento las
suma solo (carpeta `MODELS_DIR/sign/recordings/<lang>/letters`).

**Cómo se mide:** validación cruzada de 5 partes donde cada "toma" de grabación (100 cuadros
de una pulsación de tecla en v1) y cada PERSONA de ASL-HG queda entera en un lado. Así la
precisión de ASL-HG es con personas que el modelo nunca vio.

## 2. Palabras

### LSP / Señas Internacionales (tus propias grabaciones)

```bash
.venvs/sign/bin/python -m yq.sign.record --lang prl --mode words --signer ignacio --reps 5
.venvs/sign/bin/python -m yq.sign.record --lang prl --mode words --signer joaquin --reps 5
.venvs/sign/bin/python -m yq.common.heavylock .venvs/sign/bin/python training/sign/train_words.py --lang prl
```

Con 3 o más personas la validación es por persona (lo honesto); con menos, por sesión.
Meta: **≥ 5 personas × 5 repeticiones × 60 palabras** (1 500 tomas, ~2 horas en total).

### ASL (Kaggle asl-signs, 250 señas)

Necesita cuenta de Kaggle (gratis):
1. Entra a https://www.kaggle.com/competitions/asl-signs → "Late Submission"/"Join" → acepta las reglas.
2. Perfil → Settings → API → "Create New Token": baja `kaggle.json` y ponlo en `~/.kaggle/kaggle.json`
   (`chmod 600 ~/.kaggle/kaggle.json`).
3. `uv pip install --python .venvs/sign/bin/python kaggle`
4. `.venvs/sign/bin/kaggle competitions download -c asl-signs -p training/sign/data/` (~40 GB descomprimido)
   y `unzip training/sign/data/asl-signs.zip -d training/sign/data/asl-signs`
5. `.venvs/sign/bin/python -m yq.common.heavylock .venvs/sign/bin/python training/sign/train_words.py --lang ase --source kaggle`
   (la primera vez convierte los 94k parquet y guarda un caché; tarda ~1 h)

### Prueba del flujo sin datos

```bash
.venvs/sign/bin/python training/sign/train_words.py --lang ils --source synthetic --epochs 8 --out /tmp/prueba
```
Los números sintéticos NO dicen nada de la precisión real; solo prueban que todo funciona.
