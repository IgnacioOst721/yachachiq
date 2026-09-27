# Pantalla del robot (UI) — Yachachiq v2

La **pantalla táctil** del robot (Jetson, 13,3", 1920x1080) es una página web que muestra
Chromium en pantalla completa. Un servidor en Python (`yq/server/app.py`, puerto 8877)
manda a la pantalla qué mostrar, habla con los otros módulos (voz, señas, dibujo, caja)
y con la impresora y el holograma de Joaquín. Todo funciona **sin internet**.

Dos experiencias:

- **A. Historia:** contar (voz en cualquier idioma, señas LSP/ASL/IS o escribiendo) →
  confirmar el texto (con traducción al español) → foto de consentimiento (10 s; tapar la
  cámara = no publicar) → el robot dibuja → impresora (frente y reverso con QR) +
  holograma + narración → se publica en la web cuando haya internet → QR en pantalla.
- **B. Objeto:** poner el objeto en la caja → revisión (puertas, peso) → escaneo en vivo
  (plato girando, luces, cámara térmica) → resultados: qué es, 3D, luz rasante (RTI),
  UV, calor y medidas con margen de error.

## 1. Probarlo en la Mac (sin hardware)

```bash
cd ~/yachachiq/v2
~/.local/bin/uv venv --python 3.10 .venvs/ui                      # solo la primera vez
~/.local/bin/uv pip install --python .venvs/ui/bin/python -r requirements-common.txt
YQ_MOCK=1 YQ_DATA_DIR=~/yq-data-mock .venvs/ui/bin/python -m yq.server.app
```

Abre <http://localhost:8877>. Con `YQ_MOCK=1` todo se simula: micrófono, cámara,
caja, impresora, holograma y la publicación (no toca GitHub).

- Si los módulos de los otros (voz, señas, dibujo, caja) están instalados, la pantalla
  **los usa de verdad** (en su propio modo simulado). Si alguno falla al importarse, usa
  sus simulaciones propias (`yq/server/mocks*.py`). En el menú ⚙ se ve cuál es cuál.
- `YQ_UI_MOCKS=box` (o `all`, o una lista `voice,sign,art,box,camera,printer,hologram,publish`)
  obliga a usar las simulaciones de la pantalla. Ejemplo: la caja simulada de BOX necesita
  `pyserial`; en la Mac usamos `YQ_UI_MOCKS=box` para ver el escaneo completo.
- En Claude Code: configuración **yachachiq-v2** de `~/.claude/launch.json` (ya hecha).
- Ver cualquier pantalla a mano: abre `http://localhost:8877/?debug` y en la consola del
  navegador escribe `yqRender("story", "listening", {})`.

## 2. Pruebas automáticas

```bash
cd ~/yachachiq/v2 && .venvs/ui/bin/python -m pytest tests/ui
```

Unas 60 pruebas (~25 s): las tres formas de contar una historia de punta a punta (voz, señas,
texto) por el servidor real con WebSocket, el escaneo completo, errores con "Intentar otra
vez", cancelar mientras algo está colgado, el reinicio por inactividad, la clave del menú,
la impresora y el holograma contra sus servidores simulados (HTTP de verdad), y la
publicación en un **repositorio git temporal** (nunca el real ni GitHub).

## 3. Las pantallas

| Pantalla (`screen`) | Qué muestra |
|---|---|
| `home` | Dos puertas grandes: "Cuéntame una historia" y "Analiza un objeto". |
| `method` | Voz / Lengua de señas / Escribiendo. |
| `voice_lang` | Lista de idiomas con buscador (teclado en pantalla). Primero "Detectar automáticamente", luego las lenguas del Perú, luego el resto. |
| `voice_ready` | Botón grande de micrófono ("Toca y habla") y el idioma elegido. |
| `listening` | Anillos que crecen con el volumen, reloj, botón "Ya terminé". |
| `transcribing` | "Escribiendo lo que dijiste…". |
| `sign_lang` | LSP, ASL, IS (las que no tienen modelo salen en gris). |
| `signing` | Cámara con esqueleto, letra actual, palabras candidatas grandes para tocar, texto, Borrar / Espacio / Limpiar, Letras/Palabras, Listo. |
| `typing` | Teclado español (ñ, tildes, ¿¡ y el apóstrofo del quechua). |
| `confirm` | "¿Es esta tu historia?" + traducción al español; si el idioma se detectó solo, botones para corregir el idioma; Corregir (teclado) / Contar otra vez / ¡Sí! |
| `consent`, `consent_result` | Foto para el archivo: cuenta regresiva de 10 s, tapar la cámara = no; botones Sí / No. |
| `making` | Pasos del dibujo con barra tejida y el dibujo apareciendo. |
| `showtime` | El dibujo, la historia con subtítulos que avanzan con la narración, estado de impresora y holograma, "Saltar narración". |
| `done` | QR grande, título, estado de la impresora, "Contar otra historia" / "Terminar". |
| `intro` | 3 pasos para poner el objeto y el nivel de detalle (Rápido / Normal / Detallado). |
| `context` | **Opcional:** "¿Dónde lo encontraron?": chips de zona (Costa norte … Selva, Lima, Otro país, No sé) y el lugar con sus palabras (Escribir con el teclado o Dictar con la voz; se muestra para confirmar). "Saltar" o "Continuar". |
| `preflight`, `preflight_fail` | Revisando la caja; si algo falla, la lista de problemas y "Revisar otra vez". |
| `scanning` | Plato con 48 fotos llenando un anillo, ángulo, luz actual, pasos, cámara térmica, "Detener". |
| `stopping` | "Deteniendo la caja…" (se espera a que apague luces y calor). |
| `results` | Pestañas: ¿Qué es? / 3D / Luz (RTI) / UV / Calor / Medidas. Sello **SIMULADO** si el resultado es de prueba. |
| `error` | Mensaje amable en español + "Intentar otra vez" + "Volver al inicio". |

Arriba siempre: logo, botón **Inicio** (hay que tocarlo dos veces), estado de Mac /
impresora / holograma, idioma **ES / EN / QU** y el engranaje ⚙.

Si nadie toca la pantalla por `IDLE_RESET_SECONDS` (45 s), sale "¿Sigues ahí?" 10 s antes
y luego vuelve al inicio (y al español). Hablar o hacer señas también cuenta como "estar
ahí". Mientras el robot trabaja (dibujando, escaneando) no se reinicia.

## 4. Instalar en la Jetson (arranque automático)

1. Copia el repositorio a la Jetson (`~/yachachiq`) y crea el entorno:
   ```bash
   cd ~/yachachiq/v2
   curl -LsSf https://astral.sh/uv/install.sh | sh          # una vez, con internet
   ~/.local/bin/uv venv --python 3.10 .venvs/ui
   ~/.local/bin/uv pip install --python .venvs/ui/bin/python -r requirements-common.txt -r requirements/ui-jetson.txt
   sudo apt install chromium-browser git v4l-utils
   ```
2. Prueba a mano: `.venvs/ui/bin/python -m yq.server.app` y abre <http://localhost:8877>.
3. Genera los archivos de arranque y sigue las instrucciones que imprime:
   ```bash
   python3 tools/ui_install_jetson.py --out ~/yq-install
   ```
   Crea `yachachiq-kiosk.service` (el servidor arranca con la Jetson y se reinicia solo si
   se cae) y `yachachiq-kiosk.desktop` (abre Chromium en pantalla completa al iniciar sesión,
   con `tools/ui_kiosk.py`, que espera a que el servidor responda).
4. Activa el **inicio de sesión automático** del usuario y apaga el protector de pantalla.
5. Ver qué pasa: `journalctl -u yachachiq-kiosk -f`. Reiniciar: `sudo systemctl restart yachachiq-kiosk`
   (la pantalla se recarga sola cuando el servidor reinicia).

Ajustes útiles (en el `.service`, líneas `Environment=`): `YQ_PRINTER_URL`, `YQ_HOLOGRAM_URL`,
`YQ_PORTRAIT_CAMERA` (cámara que mira al visitante, p. ej. `/dev/v4l/by-id/...`),
`YQ_KIOSK_EXIT_PASSWORD`, `YQ_IDLE_RESET_SECONDS`, `YQ_CONSENT_SECONDS`, `YQ_PUBLISH_REPO_DIR`.

**Ojo con Chromium en JetPack 6:** en Ubuntu 22.04 Chromium viene como *snap*. Si no abre
después de una actualización de `snapd`, prueba `sudo snap refresh --hold snapd` o instala
otra versión; no lo pude probar en la Jetson (ver "Lo que falta verificar").

## 5. Menú ⚙ (solo para el equipo)

Pide la clave (`config.KIOSK_EXIT_PASSWORD`, por defecto `ostra`, igual que v1). Después:

- estado de cada módulo (real o simulado) y si la Mac, la impresora y el holograma responden;
- **Salir del modo kiosko** (cierra Chromium y queda el escritorio de la Jetson);
- **Publicar ahora** (sube las historias pendientes si hay internet);
- **Historias recientes** con botón **Reimprimir**;
- **Recargar pantalla**; en modo simulado, **Tapar cámara (simulada)** para probar el "no".

## 6. Publicación en la web (store-and-forward)

Igual que v1 (`robot/publish.py`), ahora en `yq/publish/`. Cada historia se guarda en
`~/yq-data/stories/<id>/` (story.json, story.txt, scene_1.png, storyteller_photo.jpg,
dibujos SVG, narración). Marcas: `.ready` (terminada y con permiso), `.private` (dijo que
no), `.published` (ya en la web). Cada 5 minutos, y al terminar cada historia, si hay
internet se copia a un **clon aparte** del repositorio (`~/yachachiq-archivo`, rama `main`)
en `docs/stories/<fecha>_<id>/`, se rehace `docs/stories.json`, commit y push. Si falla,
lo intenta la próxima vez; nunca se pierde nada.

Compatibilidad con la galería actual (`docs/index.html`, no se tocó): las entradas nuevas
tienen los mismos campos (`id, images, photos, portrait, story`) y el `story.txt` tiene el
mismo formato (el texto en español va debajo de "— En español —"; `LANG: quechua …` sigue
activando la etiqueta Quechua). Campos **nuevos** (la página actual los ignora): `v`,
`title`, `lang`, `lang_name`, `source` (voice/sign/text), `sign_lang`, `sign_lang_name`,
`text_es`, `qr_url`. Solo se sube: story.txt, scene_1.png, storyteller_photo.jpg e info.json.

**El QR del papel.** ART imprime `PUBLIC_BASE_URL + story_id` (una ruta). Para que ese enlace
funcione en GitHub Pages, al publicar también se escribe `docs/<story_id>/index.html`, una
página mínima que redirige a la galería (`../#<fecha>_<id>`). Si el visitante **no** quiso
publicar, el QR impreso (y el de la pantalla) apunta a la página general de la galería, sin
historia: la pantalla le pide a ART `published=False` (si ART aún no acepta ese parámetro, la
pantalla cambia `QR_URL_FORMAT` de ART a `{base}` solo durante ese dibujo).

Primera vez en la Jetson (con internet):
`git clone https://github.com/IgnacioOst721/yachachiq.git ~/yachachiq-archivo` y configurar
el push (token o llave SSH) en ese clon.

## 7. Eventos del WebSocket `/ws` (servidor → pantalla)

Cada mensaje es JSON `{"type": ..., "t": epoch, ...}`. El servidor manda; la pantalla solo dibuja.

| type | Campos | Cuándo |
|---|---|---|
| `hello` | `boot_id, flow, screen, data, idle_reset_s, ui_language, mock` | Al conectar (si cambia `boot_id`, la página se recarga). |
| `screen` | `flow` (story/scan/null), `screen`, `data` | Cambio de pantalla (lista en §3). |
| `screen_update` | `flow, screen, data` (solo lo que cambió) | Progreso del dibujo, traducción lista, estado de impresora/holograma, narración. |
| `level` | `level` 0..1 | Volumen del micrófono (~15 por segundo). |
| `sign` | `hands_visible, fps, mode, text, buffer, candidates:[{text,prob,kind}], letter:{current,progress}, error` | Estado de señas (~5 por segundo). |
| `sign_status` | `status` | Lo que mande `on_status` del motor de señas. |
| `consent_tick` | `remaining, camera, covered, face, smile` | Cada ~0,5 s en la foto de consentimiento. |
| `scan_live` | `stage, fraction, message_es, detail{platter_deg, camera, light, led, phase, temp_max_c, t, photo_url, thermal_preview_url}` | Progreso de la caja. |
| `scan_photo` | `url, kind` (photogrammetry/rti/uv), `name, camera, angle` | Cada foto nueva en la carpeta del escaneo. |
| `scan_thermal` | `url` | Vista térmica en colores (se genera de `thermal/sequence.npy`). |
| `printer` | `story_id, id, status, progress, message` | Estado del trabajo de impresión. |
| `published` | `story_id, status, new, …` | Resultado de la publicación. |
| `idle_warning` / `idle_clear` | `seconds` | "¿Sigues ahí?". |
| `toast` | `text_es, key, kind` | Aviso corto. |
| `flow_end` | `flow, outcome` (done/cancelled/home/idle/stopped/error) | Terminó una experiencia. |
| `status` | igual que `GET /status` | Cada 10 s. |

Pantalla → servidor por el mismo WebSocket: `{"type":"activity"}` (alguien tocó) y
`{"type":"act","action":...,"payload":{...}}` (igual que `POST /api/act/<action>`).

## 8. REST

- `GET /` página; `GET /status` estado de todo; `GET /api/state` pantalla actual;
  `GET /api/languages?feature=asr` lista de idiomas ya ordenada.
- `POST /api/start {"flow":"story"|"scan", "method"?, "profile"?}` (409 si ya hay otra en curso).
- `POST /api/act/<action> {...}` → 409 si esa acción no vale en esta pantalla. Acciones:
  `choose {method}`, `choose_lang {code}`, `change_lang`, `record`, `stop_listening`,
  `choose_sign {code}`, `sign_accept {index}`, `sign_backspace`, `sign_space`, `sign_clear`,
  `sign_mode {mode}`, `sign_done`, `submit_text {text, lang}`, `confirm`, `edit_text {text}`,
  `retell`, `relang {code}`, `consent {publish}`, `skip_narration`, `finish`, `new_story`,
  `choose_profile {profile}`, `start`, `set_context {found_where?, region_hint?, lang?}`,
  `dictate {ui_lang}`, `stop_dictation`, `continue {found_where, region_hint, lang?}`, `skip`,
  `stop_scan`, `new_scan`, `retry`, `back`, `home`.
- `POST /api/cancel` (volver al inicio), `POST /api/touch`.
- `POST /api/kiosk/exit {password}` (403 con clave mala).
- `GET /sign/preview.mjpeg` (y `.jpg`) desde `SignEngine.latest_jpeg()`;
  `GET /camera/preview.mjpeg` (cámara del consentimiento).
- `GET /files/stories/<id>/<archivo>`, `/files/scans/<id>/<archivo>`, `/files/uicache/...`
  (no se puede salir de esas carpetas).
- Menú ⚙ (todas piden `password`): `POST /api/admin/check`, `/stories`, `/reprint {story_id}`,
  `/publish`, `/refresh`, `/mock_camera {covered}`.

## 9. Cambiar textos y traducciones

Los textos están en `yq/server/static/i18n/es.json` (base), `en.json` y `qu.json`.
Cambia el texto, guarda y recarga la pantalla (no hay que compilar nada). Si una clave falta
en inglés o quechua, se muestra la de español. `{n}` o `{lang}` se reemplazan solos.
Los mensajes que vienen de otros módulos (`message_es`, p. ej. "Luz 3 de 8") solo están en
español.

### Quechua: TODO debe revisarlo un hablante

Lo escribí yo sin ser hablante (quechua sureño, ortografía chanka de 3 vocales). **Revisar
cada línea** antes de la competencia; lo que no está traducido sale en español.

| Clave | Español | Quechua (borrador, REVISAR) |
|---|---|---|
| home | Inicio | Qallariyman |
| idle_title | ¿Sigues ahí? | ¿Kaypiraqchu kachkanki? |
| home_title | ¡Hola! ¿Qué hacemos hoy? | ¡Allillanchu! ¿Imatam kunan ruwasun? |
| door_story | Cuéntame una historia | Huk willakuyta willaway |
| door_scan | Analiza un objeto | Ñawpa kaqta qawasun |
| start | Empezar | Qallarisun |
| method_title | ¿Cómo quieres contarla? | ¿Imaynatam willakuyta munanki? |
| method_voice / method_text | Con mi voz / Escribiendo | Simiywan / Qillqaspa |
| voice_lang_title | ¿En qué idioma vas a hablar? | ¿Ima simipim rimanki? |
| tap_talk | Toca y habla | Llamiy hinaspa rimay |
| listening | Te escucho… | Uyarichkaykim… |
| done_talking | Ya terminé | Tukuniñam |
| confirm_title | ¿Es esta tu historia? | ¿Kaychu willakuyniyki? |
| confirm_yes | ¡Sí, es mi historia! | ¡Arí, willakuyniymi! |
| consent_title | ¿Compartimos tu historia? | ¿Willakuyniykita llapanwan rakinakusunchu? |
| consent_yes / consent_no | Sí, compartir / No, gracias | Arí / Manam, sulpayki |
| consent_thanks | ¡Gracias por compartir! | ¡Añay, sulpayki! |
| making_title | Dibujando tu historia | Willakuyniykita siqichkani |
| done_title | ¡Listo! | ¡Tukunñam! |
| scan_intro_title | Pon tu objeto en la caja | Kaqniykita cajaman churay |
| results_title | Esto descubrimos | Kaytam tarinchik |
| ctx_title | ¿Dónde lo encontraron? | ¿Maypim tarirqanku? |
| ctx_skip / ctx_continue | Saltar / Continuar | Pasay / Qatiy |
| ctx_type / ctx_dictate | Escribir / Dictar | Qillqay / Rimay |
| region_no_se / region_otro_pais | No sé / Otro país | Manam yachanichu / Huk suyu |
| region_selva / region_altiplano | Selva / Altiplano | Sacha-sacha / Qullaw pampa |

Lista completa en `qu.json` (69 claves). El resto (errores, menú ⚙, medidas) sale en español.

## 10. Formato RTI (luz rasante) — PROPUESTA

BOX-ANALYSIS todavía no publicó su formato (`docs/box_analysis.md` no existía). El visor
(`static/js/viewers/rti.js`) lee este formato; si BOX-ANALYSIS elige otro, hay que adaptar
el visor o convertir. Archivos en una carpeta (la ruta del `ptm.json` va en
`ScanResult.artifacts["rti_ptm"]`):

- `ptm.json`: `{"format": "yq-ptm-lrgb-v1", "width", "height", "coeff_images": ["ptm_coeffs_0.png", "ptm_coeffs_1.png"],
  "scale": [6 números], "bias": [6 números], "albedo": "ptm_albedo.png", "px_per_mm"?}`
- `ptm_coeffs_0.png` (RGB = a0, a1, a2) y `ptm_coeffs_1.png` (RGB = a3, a4, a5), 8 bits sin
  corrección de color. Decodificar: `a_i = bias[i] + scale[i] * (byte / 255)`.
- `ptm_albedo.png`: color de la superficie (sRGB).
- Luz: vector unitario `(lu, lv, lw)`, `lu` hacia la derecha de la imagen, `lv` hacia
  arriba de la imagen. Brillo `L = a0·lu² + a1·lv² + a2·lu·lv + a3·lu + a4·lv + a5`;
  color = albedo × L. Modo "Relieve" = gris × L.
- La prueba `tests/ui/test_mock_assets.py` comprueba que esto reproduce el sombreado de una
  superficie conocida (error medio < 0,05).

Nombres de `artifacts` que usa la pantalla (tolera otros parecidos): `model_glb` (.glb, en
metros, Y arriba), `rti_ptm`, `uv_image`, `visible_image`, `uv_overlay`, `thermal_max`,
`thermal_anomaly`, `photo_front`. `findings[].image` se muestra en la pestaña de su `analysis`.

## 10b. "¿Dónde lo encontraron?" (CONTRACTS.md §9)

Paso opcional entre `intro` y la revisión de la caja. Lo que diga el visitante viaja como
`ScanRequest.context = {"found_where", "region_hint", "notes", "lang"}` (`found_where` máx.
200 letras; `region_hint` uno de `costa_norte, costa_central, costa_sur, sierra_norte,
sierra_central, sierra_sur, altiplano, selva, lima, otro_pais, no_se`; `lang` = idioma del
dictado según la transcripción, o el idioma de la pantalla si lo escribió). "Saltar" manda `{}`.

- **Dictar** usa `yq.voice` (Recorder + transcribe, máx. 25 s). Idioma: el de la pantalla
  (ES → español, EN → inglés, QU → detección automática). Si falla, sale un aviso y se puede
  repetir o escribir; nunca una pantalla de error (el paso es opcional).
- **Es una pista, no una prueba.** En la ficha de resultados se muestra "Lugar según el
  visitante", la frase `Identification.context_effect_es` y, si la respuesta solo con la
  imagen (`image_only`) es otra, las dos: "Solo por la imagen: Moche (48 %) · Con el lugar:
  Chimú (57 %)".
- La caja simulada de la pantalla imita esto (costa norte ayuda, "Chan Chan" cambia a Chimú,
  Selva/Altiplano/Otro país contradicen y gana la imagen) solo para probar; la de verdad la
  hace BOX-ANALYSIS.

## 11. Impresora y holograma (propuesta para Joaquín)

**Impresora** (`yq/printer/client.py`): `POST {PRINTER_URL}/jobs` multipart, una parte por
archivo con el nombre del archivo como campo: `front.svg`, `back.svg`, opcionales
`front.gcode`/`back.gcode`, y `job.json` `{"story_id","paper":{"w_mm","h_mm"},"pens":[...],
"flip":"long-edge","qr_url"}` → `{"id"}`. Opcional (si existe, la pantalla muestra el avance):
`GET /jobs/<id>` → `{"status": queued|drawing_front|flipping|drawing_back|done|error|cancelled,
"progress": 0..1}`, `POST /jobs/<id>/cancel`, `GET /health`. Simulador para probar:
`python -m yq.printer.mock_server --port 8900`.

**Holograma** (`yq/hologram/client.py`): `POST {HOLOGRAM_URL}/stories` multipart con
`story.json` `{"story_id","title","lang","text","text_es","scenes":[{"image","caption_es","audio"}]}`
y los archivos que nombra (scene_1.png, narration_1.wav) → `{"ok","id","duration_s"?}`;
lo reproduce enseguida. Opcional: `GET /status` → `{"playing","remaining_s"}`.
Simulador: `python -m yq.hologram.mock_server --port 8950`.

Si la impresora o el holograma no responden, la historia sigue igual y la pantalla lo dice;
desde el menú ⚙ se puede reimprimir después.

## 12. Cómo se protege el robot de quedarse trabado

- Cada experiencia corre en su propio hilo; cada llamada lenta (transcribir, traducir,
  dibujar, escanear, narrar) tiene tiempo máximo y se puede cancelar: el botón Inicio
  vuelve en menos de 1 s aunque un módulo esté colgado (probado en `test_robustness.py`).
- Cualquier error muestra un mensaje amable y "Intentar otra vez" (vuelve al paso correcto).
- Detener un escaneo espera (máx. 20 s) a que la caja apague luces y calor antes de salir.
- Una sola experiencia a la vez; la caja no empieza un escaneo nuevo mientras la anterior se detiene.

## 13. Lo que no se pudo verificar

- Nada se probó en la Jetson ni en la pantalla táctil real (solo navegador de la Mac a
  1920x1080). Falta: Chromium kiosk en JetPack 6, gestos táctiles reales, micrófono, cámara.
- La impresora y el holograma de Joaquín: solo contra simuladores (las interfaces son propuestas).
- La publicación: probada con repositorios git temporales, no contra GitHub.
- Las traducciones al quechua (ver §9).
- El formato RTI real de BOX-ANALYSIS (ver §10).
- Detección de cara/sonrisa: OpenCV 5 (Mac) no trae los modelos; en la Jetson (OpenCV 4)
  debería funcionar. Sin ellos el consentimiento funciona igual (tapar = no).

## 14. Números medidos

- Unas 60 pruebas automáticas en ~25 s (Mac M4, con otros procesos pesados corriendo).
- Tiempos de la pantalla, medidos en modo simulado: cambiar de pantalla < 0,1 s;
  "Inicio" con un módulo colgado vuelve en < 2,5 s (prueba automática).
- Escaneo simulado completo (perfil rápido): ~25 s con la Mac libre; con la Mac muy
  cargada (load 50) tardó 16 min, por eso los simuladores de la caja respetan
  `YQ_MOCK_SPEED`.

## 15. Archivos

`yq/server/` (app.py, routes_media.py, kiosk.py, bus.py, adapters.py, camera.py, settings.py,
mocks.py, mocks_sign.py, mocks_art_box.py, mock_assets.py, flows/), `yq/server/static/`
(index.html, css/, js/, i18n/, vendor/), `yq/printer/`, `yq/hologram/`, `yq/publish/`,
`tools/ui_kiosk.py`, `tools/ui_install_jetson.py`, `tests/ui/`, `requirements/ui-*.txt`,
licencias en `docs/licenses_ui.md`.
