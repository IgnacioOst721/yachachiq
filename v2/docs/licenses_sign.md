# Licencias: dominio SIGN (lengua de señas)

Todo lo que usa el reconocimiento de señas es código abierto o datos con licencia abierta.
Nada necesita internet durante la competencia. Última revisión: 2026-09-27.

## Modelos que corren en el robot

| Qué | Para qué | Licencia | De dónde |
|---|---|---|---|
| **RTMW-m** (`rtmw-dw-l-m_simcc-cocktail14_270e-256x192`), ONNX | Pose de cuerpo entero, 133 puntos (modelo por defecto) | Apache-2.0 (OpenMMLab MMPose) | `https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/...` (espejo: `huggingface.co/Tau-J/RTMPose`) |
| **RTMW-l** 256x192 y 384x288, ONNX | Pose más precisa (opcional, más lento) | Apache-2.0 | igual |
| **YOLOX-tiny** Human-Art (`yolox_tiny_8xb8-300e_humanart`), ONNX | Encontrar a la persona | Apache-2.0 (OpenMMLab; YOLOX de Megvii es Apache-2.0) | `.../rtmposev1/onnx_sdk/` |
| **RTMPose-m hand** (`rtmpose-m_simcc-hand5`), ONNX | Afinar los dedos de la mano que deletrea | Apache-2.0 | `.../rtmposev1/onnx_sdk/` |
| `letters_prl`, `letters_ase` (nuestros) | Letras LSP y ASL | Nuestro (licencia del repo). Entrenados con los datos de Ignacio y con ASL-HG (CC BY 4.0, ver abajo: hay que citarlo) | `yq/sign/models/` |
| `words_<lang>` (nuestros, cuando se entrenen) | Palabras | Nuestro + licencia de los datos usados | `MODELS_DIR/sign/` |

Los archivos de pose se bajan con `python -m yq.sign.modelstore download` (se guarda el sha256 de
cada zip en `pose/<modelo>/source.json`).

## Librerías

| Librería | Licencia | Dónde se usa |
|---|---|---|
| ONNX Runtime (`onnxruntime`, `onnxruntime-gpu` de Jetson AI Lab) | MIT | inferencia (Mac y Jetson) |
| NVIDIA TensorRT (viene con JetPack) | licencia propietaria gratuita de NVIDIA (parte de JetPack, uso permitido en Jetson) | acelerar en el Jetson (opcional; sin TensorRT se usa CUDA) |
| rtmlib 0.0.16 | Apache-2.0 | solo en el Mac, como referencia en un test; `yq/sign/rtm.py` porta su pre/post-proceso (con atribución) |
| OpenCV | Apache-2.0 | cámara, dibujo |
| NumPy | BSD-3 | todo |
| Pillow | MIT-CMU (HPND) | texto con tildes en el grabador |
| PyTorch | BSD-3 | solo entrenamiento en el Mac |
| onnx, onnxscript | Apache-2.0 / MIT | exportar modelos |
| scikit-learn | BSD-3 | validación cruzada |
| pyarrow, pandas | Apache-2.0 / BSD-3 | leer parquet |
| MediaPipe 0.10.21 | Apache-2.0 | solo en el Mac (venv aparte): puntos de las fotos ASL-HG para entrenar y el estudio MediaPipe↔RTM; el robot NO lo usa |
| MediaPipe `hand_landmarker.task` | Apache-2.0 | se bajó para probar la API nueva (falló en este Mac); no se usa |

## Datos

| Datos | Licencia | Uso | Nota |
|---|---|---|---|
| Grabaciones v1 de Ignacio (`lsp/lsp_data.csv`, `~/asl-camera/asl_data.csv`) | propias del equipo | entrenar letras | 1 sola persona |
| **ASL-HG** (Pranto et al., Mendeley Data, doi:10.17632/j4y5w2c8w9.1), copia en Hugging Face `juanjodurillo/asl-hg` | **CC BY 4.0** (hay que citar a los autores) | entrenar y medir letras con 10 personas nuevas | fotos de manos A–Z y 0–9. Para LSP NO se usa la U (es distinta en LSP) |
| scikit-image `astronaut.png` (foto de la NASA, dominio público) | dominio público | imagen de prueba de la pose | |
| FrequencyWords 2018 (Hermit Dave, OpenSubtitles) `en_50k`, `es_50k` | contenido **CC BY-SA 4.0**, código MIT | autocompletar palabras (`lexicon_eng.tsv`, `lexicon_spa.tsv`) | como es BY-SA, estos dos archivos derivados también quedan CC BY-SA 4.0 con atribución |
| `story_words.tsv`, `lexicon_que.tsv`, `vocab_prl.txt` | nuestros | palabras de la historia (castellano, quechua, inglés) | el quechua debe revisarlo un hablante |
| Google ISLR **asl-signs** (Kaggle, 250 señas) | CC BY 4.0 según el paper de PopSign (NeurIPS 2023); la página de Kaggle no se pudo leer sin cuenta | palabras ASL (**no descargado**: faltan credenciales de Kaggle) | hay que unirse a la competencia y aceptar reglas |
| Google **asl-fingerspelling** (Kaggle) / FSboard | CC BY 4.0 (paper FSboard, arXiv 2407.15806) | no usado todavía | Kaggle con cuenta |

### Datos revisados y NO usados

| Datos | Por qué no |
|---|---|
| PUCP-305, PUCP-DGI156, AEC (PUCP, `datos.pucp.edu.pe`) | no se pudo verificar la licencia ni si la descarga es abierta (el sitio no respondió); AEC viene de videos de TV (derechos dudosos) |
| VideoLSP10 (GitHub `videoLSP/VideoLSP10`) | el archivo LICENSE está vacío = sin licencia; solo 10 oraciones con Kinect |
| Leipzig Corpora quechua (`que_wikipedia_2016_10K`) | licencia de Leipzig no verificada |
| Marxulia/asl_sign_languages_alphabets_v03 (HF) | sin licencia declarada |
| sign/popsign-images (HF, CC BY 4.0) | 91 GB de imágenes, sin keypoints; demasiado grande para el Mac compartido |
