# Licencias de VOZ (voice)

Todo lo que usa el dominio de voz es de código abierto o de descarga libre (regla WRO FI 5.3).
**Marcado con ⚠️** = licencia **no comercial** (CC-BY-NC) o **copyleft** (GPL): sirve para la
competencia (uso educativo, no se vende nada), pero hay que decirlo si alguien pregunta y no se
podría vender el robot con esos pesos.

Verificado el 2026-09-27 en las fichas de Hugging Face, PyPI y GitHub.

## Modelos (pesos)

| Qué | Dónde | Licencia | Para qué |
|---|---|---|---|
| Whisper large-v3 y large-v3-turbo (OpenAI) | `mlx-community/whisper-large-v3-mlx`, `mlx-community/whisper-large-v3-turbo` | MIT | Reconocer voz en la Mac (idiomas fuertes de Whisper) |
| Whisper large-v3-turbo / small para CTranslate2 | `dropbox-dash/faster-whisper-large-v3-turbo` (antes `mobiuslabsgmbh/...`), `Systran/faster-whisper-small` | MIT | Respaldo local en el Jetson |
| Meta Omnilingual ASR CTC 300M/1B v2, LLM 1B v2 | `dl.fbaipublicfiles.com/mms/omniASR-*-v2.pt` | Apache-2.0 | Quechua, aimara, lenguas amazónicas y ~1600 idiomas |
| ⚠️ MMS-LID-4017 (Meta) | `facebook/mms-lid-4017` | CC-BY-NC-4.0 | Detectar el idioma cuando Whisper no lo conoce |
| ⚠️ MMS-TTS (Meta, VITS), ~1100 voces | `facebook/mms-tts-<iso3>` | CC-BY-NC-4.0 | Hablar en quechua, aimara, asháninka, awajún, shipibo… |
| ⚠️ NLLB-200 distilled 1.3B (Meta) | `OpenNMT/nllb-200-distilled-1.3B-ct2-int8` | CC-BY-NC-4.0 | Traducir (motor principal, el más preciso en quechua/aimara) |
| MADLAD-400 3B MT (Google) | `Nextcloud-AI/madlad400-3b-mt-ct2-int8` (conversión de `google/madlad400-3b-mt`) | Apache-2.0 | Traducir los pares que NLLB no tiene (~400 idiomas) |
| Silero VAD v6.2.3 | copia en `v2/yq/voice/data/silero_vad.onnx` (+ `SILERO_LICENSE`) | MIT | Saber cuándo empieza y termina de hablar el visitante |
| Voces Piper (Jetson) | `rhasspy/piper-voices` | **cada voz tiene la suya** (abajo) | Hablar en español, inglés y otros idiomas grandes |

Voces Piper que descarga `tools/voice_download.py --jetson` (licencia leída de cada `MODEL_CARD`):

| Idioma | Voz | Licencia |
|---|---|---|
| Español | `es_MX-claude-high` | Apache-2.0 |
| Inglés | `en_US-lessac-high` | ⚠️ licencia del corpus Lessac/Blizzard 2013 (solo investigación, no comercial) |
| Portugués | `pt_BR-cadu-medium` | CC0 |
| Francés | `fr_FR-mls-medium` | CC-BY-4.0 |
| Alemán | `de_DE-thorsten-high` | CC0 |
| Italiano | `it_IT-serena-high` | CC-BY-4.0 |
| Chino | `zh_CN-chaowen-medium` | CC0 |
| Japonés | `ja_JP-hi_fi_captain-medium` | ⚠️ CC-BY-NC-SA-4.0 |
| Ruso | `ru_RU-denis-medium` | CC0 |
| Árabe | `ar_JO-kareem-medium` | ver URL de su MODEL_CARD |
| Hindi | `hi_IN-pratham-medium` | ⚠️ CC-BY-NC-SA-4.0 |
| Turco | `tr_TR-dfki-medium` | ⚠️ CC-BY-NC-SA-4.0 |
| Neerlandés | `nl_BE-nathalie-medium` | CC0 |
| Polaco | `pl_PL-bass-high` | Apache-2.0 |

## Programas (librerías)

| Librería | Licencia | Nota |
|---|---|---|
| mlx, mlx-whisper | MIT | |
| omnilingual-asr | Apache-2.0 | |
| fairseq2 / fairseq2n | MIT | necesita libsndfile (LGPL-2.1), que viene dentro de `soundfile` |
| torch, torchaudio | BSD-3-Clause | |
| transformers, huggingface_hub, sentencepiece | Apache-2.0 | |
| CTranslate2, faster-whisper | MIT | |
| onnxruntime | MIT | |
| ⚠️ piper-tts 1.8.0 (OHF-Voice/piper1-gpl) | GPL-3.0-or-later | trae espeak-ng (GPL-3.0) adentro. Se usa como programa aparte; si algún día se distribuye el robot, hay que publicar el código |
| sounddevice (PortAudio) | MIT | |
| soundfile (libsndfile) | BSD-3-Clause (libsndfile LGPL-2.1) | |
| uroman 1.3.1.1 | tipo MIT, pide citar a Hermjakob et al. (2018) | romaniza texto para voces MMS de alfabetos no latinos |
| jiwer, sacrebleu | Apache-2.0 | solo para medir |
| babel (datos CLDR) | BSD-3-Clause (CLDR: licencia Unicode) | solo para construir la lista de idiomas |

## Datos

| Datos | Licencia | Uso |
|---|---|---|
| Tablas ISO 639-3 (SIL International) | términos de uso de SIL (uso libre de los códigos) | códigos y nombres de idiomas |
| Glottolog CLDF `languages.csv` | CC-BY-4.0 | región (macroárea) y países |
| CLDR (nombres de idiomas en español/inglés/propio) | Unicode License v3 | nombres en la pantalla |
| Listas de idiomas de Whisper, Omnilingual, NLLB, MADLAD, MMS, Piper | las de cada proyecto (arriba) | qué motor sirve para qué idioma |
| Tabla CER por idioma de Omnilingual 7B | Apache-2.0 (repo) | calidad publicada por Meta |
| FLEURS (Google), prueba es_419 / en_us / pt_br | CC-BY-4.0 | medir WER (80 frases cada uno) |
| FLORES-200 devtest (Meta) | CC-BY-SA-4.0 | medir traducción |
| Common Voice quechua de Puno (qxp), vía el espejo `Epiph0nE/quechua-puno-tts` | CC0 (origen Common Voice; el espejo no declara licencia) | medir WER en quechua collao |
| ⚠️ `josemercado/quechua-chanka-asr` | **no declara licencia** | solo para medir en quechua chanka, no se redistribuye ni se entrena con él |

Nada de esto se descarga durante la competencia: `tools/voice_download.py` lo baja todo antes.
