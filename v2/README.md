# Yachachiq v2

Robot de WRO 2026 Future Innovators, "Robots Meet Culture". Colegio FDR, Lima.

Un visitante cuenta una historia **por voz, en cualquiera de 485 idiomas que el robot escucha y
entiende**, o **en lengua de señas** (LSP, ASL o Seña Internacional). El robot la entiende, planea un
dibujo, lo genera con IA y lo revisa. La impresora de Joaquín lo dibuja con lápiz: en el frente la
imagen, y en el reverso la historia y un QR. El holograma la cuenta con sonido.

Aparte, la **caja de análisis** estudia un objeto arqueológico sin tocarlo: lo pesa, lo fotografía
alrededor, lo ilumina con luz rasante (RTI), lo mira con luz UV y lo calienta un poco para verlo con la
cámara térmica. Con eso arma el modelo 3D, mide el volumen y la densidad, e identifica tipo, material,
cultura y época con un catálogo de 14,075 piezas de museos, todo **sin internet**.

La v1 (`../robot/`, `../lsp/`) sigue intacta: era para la Raspberry Pi.

## Las partes

| Parte | Carpeta | Corre en | Manual |
|---|---|---|---|
| Base común: configuración, contratos, servidor de la Mac | `yq/common`, `yq/macworker` | las dos | [CONTRACTS.md](CONTRACTS.md) |
| Pantalla y recorridos (kiosko) | `yq/server` | Jetson | [docs/ui.md](docs/ui.md) |
| Voz: escuchar, traducir, hablar | `yq/voice` + `yq/macworker/models/voice_*` | las dos | [docs/voice.md](docs/voice.md) |
| Lengua de señas | `yq/sign`, `training/sign` | Jetson | [docs/sign.md](docs/sign.md) |
| Dibujo | `yq/art` + `yq/macworker/models/art_*`, `llm`, `vlm` | las dos | [docs/art.md](docs/art.md) |
| Caja: firmware, cámaras, escaneo | `firmware/box_esp32`, `yq/box` | ESP32 + Jetson | [docs/box.md](docs/box.md) |
| Caja: análisis e identificación | `yq/box/analysis` | Mac (y Jetson sin identificar) | [docs/box_analysis.md](docs/box_analysis.md) |
| Impresora, holograma, web | `yq/printer`, `yq/hologram`, `yq/publish` | Jetson | [docs/ui.md](docs/ui.md) |

## Máquinas y red

Todo va conectado al router GL.iNet Beryl AX. Mejor con cable, si se puede. Pon estas IP fijas en el router:

| Máquina | IP | Qué hace |
|---|---|---|
| Jetson Orin Nano Super | 192.168.8.10 | pantalla, micrófono, señas, caja, impresora, holograma |
| MacBook M4 | 192.168.8.20 | modelos grandes de IA, puerto 8700 |
| Impresora (Joaquín) | 192.168.8.30 | puerto 8900 (a confirmar con Joaquín) |
| Holograma (Joaquín) | 192.168.8.40 | puerto 8950 (a confirmar con Joaquín) |

## Probarlo ya en la Mac, sin hardware

```bash
cd ~/yachachiq/v2
YQ_MOCK=1 YQ_UI_MOCKS=box YQ_DATA_DIR=~/yq-data-mock .venvs/ui/bin/python -m yq.server.app
```

Abre http://localhost:8877. Todo está simulado: la caja, las cámaras y los modelos.

## La Mac: servidor de IA

Se instala una vez, con internet:

```bash
cd ~/yachachiq/v2
uv venv --python 3.10 .venvs/mac
uv pip install --python .venvs/mac/bin/python -r requirements-common.txt -r requirements/mac-worker.txt \
    --excludes requirements/art-mac-excludes.txt
```

El quechua (Omnilingual de Meta) necesita otras versiones de librerías, así que corre aparte con
`.venvs/voice`. Los modelos se bajan con los comandos de cada manual: [voz](docs/voice.md),
[dibujo](docs/art.md) y [catálogo de museos](docs/box_analysis.md).

Para prenderlo:

```bash
cd ~/yachachiq/v2 && .venvs/mac/bin/python -m yq.macworker.app
```

Al arrancar carga la voz y el planificador (unos 5 GB). Comprueba que funcione en
http://localhost:8700/health: tiene que decir `routes_art`, `routes_box` y `routes_voice` en `ok`.

## El Jetson (cuando llegue en noviembre)

Un solo entorno para todo: [requirements/jetson.txt](requirements/jetson.txt) tiene los pasos. Lo más
delicado es que ONNX Runtime venga del índice de NVIDIA (Jetson AI Lab) y no de PyPI; si no, las
señas corren sin tarjeta gráfica. Después, en este orden:

1. `python tools/sign_benchmark.py --backends tensorrt cuda`: mide los cuadros por segundo de las señas.
2. `python -m yq.voice.check`: comprueba micrófono y parlante.
3. `python -m yq.box.cli ping`, luego `status`, `weigh` y `rotate 90`: prueba la caja paso a paso ([docs/box.md](docs/box.md)).
4. `python -m yq.box.cli capture-background` y las calibraciones de [docs/box_analysis.md](docs/box_analysis.md).
5. `python tools/ui_install_jetson.py`: el kiosko arranca solo al prender el Jetson.

## Día de la competencia

1. La Mac **enchufada**, **sin** modo de bajo consumo y con la tapa abierta. Prende el servidor de IA.
2. Prende el router, y luego el Jetson.
3. En la pantalla, en el menú ⚙, **Estado**: todo tiene que decir real, no simulado, y la Mac tiene
   que aparecer alcanzable.
4. Si la Mac falla, el robot sigue funcionando, pero más simple: la voz la reconoce el Jetson, el dibujo
   se hace sin IA y la caja no identifica.

## Qué está probado y qué no (27 de septiembre de 2026)

Probado en la Mac: 315 pruebas automáticas rápidas (más las pesadas que cargan los modelos reales) y un
recorrido completo con la caja simulada que
escanea, manda a la Mac, mide, arma el 3D, hace RTI, UV y térmica, e identifica con el catálogo y el
modelo de visión. También dibujos reales con IA y voz real.

Medido:

| Qué | Resultado |
|---|---|
| Voz en español, inglés y portugués | 2.4, 5.9 y 3.9 palabras mal de cada 100 |
| Voz en quechua | 5 letras mal de cada 100 (28 a 32 palabras con algún error) |
| Letras LSP / ASL con personas nuevas | 95 % / 94 % (la correcta está entre 3 opciones el 99.6 % de las veces) |
| Identificar la cultura (catálogo, sin modelo de visión) | 65.5 % a la primera, 81 % entre 3 |
| Identificar el material | 85.6 % / 96.8 % |
| Dibujo de punta a punta (Mac cargada) | 1.3 a 4 minutos, según si reintenta con Z-Image |

Sin probar hasta que llegue el hardware: el Jetson, las cámaras reales, la cámara térmica, el ESP32 con
motor, balanza y luces, la precisión real de las medidas, y la impresora y el holograma de Joaquín.

Pendientes de Ignacio: cuenta de Kaggle (palabras ASL), grabar palabras LSP con 5 personas, revisión del
quechua de la pantalla, intérprete de LSP, confirmar con Joaquín las interfaces de la impresora y el
holograma, y confirmar con WRO Perú el uso del router.
