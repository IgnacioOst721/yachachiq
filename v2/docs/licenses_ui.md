# Licencias — dominio UI (pantalla, impresora, holograma, publicación)

Todo es software libre o propio del equipo. No se usa ninguna API paga ni ningún servicio en
internet mientras el robot funciona (regla WRO FI 5.3).

## Copiado dentro del proyecto (funciona sin internet)

| Qué | Versión | Licencia | Dónde | De dónde se bajó |
|---|---|---|---|---|
| three.js (`three.module.js`, `three.core.js`) | 0.186.1 | MIT | `yq/server/static/vendor/three/` (+ `LICENSE`) | `npm pack three@0.186.1` |
| three.js addons: GLTFLoader, OrbitControls, BufferGeometryUtils, SkeletonUtils | 0.186.1 | MIT | `vendor/three/addons/` | mismo paquete (`examples/jsm/`) |
| qrcode-generator (`qrcode.mjs`), Kazuhiko Arase | 2.0.4 | MIT | `vendor/qrcode/` (+ `LICENSE.txt`) | `npm pack qrcode-generator@2.0.4` |
| Fuente Baloo 2 (600, 800; latin + latin-ext, woff2) | Fontsource 5.3.0 | SIL OFL 1.1 | `vendor/fonts/` (+ `OFL-Baloo2.txt`) | `npm pack @fontsource/baloo-2@5.3.0` |
| Fuente Atkinson Hyperlegible (400, 700; latin + latin-ext, woff2) | Fontsource 5.3.0 | SIL OFL 1.1 | `vendor/fonts/` (+ `OFL-AtkinsonHyperlegible.txt`) | `npm pack @fontsource/atkinson-hyperlegible@5.3.0` |

Para volver a bajarlos (con internet, en cualquier carpeta):
`npm pack three@0.186.1 qrcode-generator@2.0.4 @fontsource/baloo-2@5.3.0 @fontsource/atkinson-hyperlegible@5.3.0`
y copiar los archivos de la tabla.

"QR Code" es marca registrada de DENSO WAVE INCORPORATED (solo se menciona, como pide la licencia).

## Paquetes de Python (de `requirements-common.txt`, no se agregó ninguno)

| Paquete | Licencia | Para qué |
|---|---|---|
| FastAPI, Starlette | MIT, BSD-3 | servidor web y WebSocket |
| Uvicorn | BSD-3 | correr el servidor |
| python-multipart | Apache-2.0 | subir archivos (simuladores de impresora/holograma) |
| requests | Apache-2.0 | clientes de impresora y holograma |
| numpy | BSD-3 | audio, datos RTI y térmicos |
| Pillow | MIT-CMU (HPND) | imágenes simuladas, vistas térmicas |
| opencv-contrib-python-headless | Apache-2.0 | cámara del consentimiento, detección de cara opcional |
| httpx, pytest | BSD-3, MIT | pruebas |

## Programas del sistema (Jetson)

Chromium (BSD-3 y otras libres) para la pantalla completa; git (GPL-2.0) para publicar.

## Contenido propio

Los íconos SVG, el diseño (colores, patrones tocapu/awayu, chakana), los textos en
español/inglés/quechua, los datos simulados (vasija 3D, relieve RTI, imágenes térmicas) y todo
el código de `yq/server`, `yq/printer`, `yq/hologram`, `yq/publish` son del equipo.
No se usan modelos de IA en este dominio.
