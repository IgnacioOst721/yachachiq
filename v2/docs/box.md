# Caja de análisis – captura (BOX-CAPTURE)

Esta parte hace que la caja **mida y fotografíe** el objeto:

1. **Pesa** el objeto (celda de carga TAL220B + HX711).
2. Lo **gira** en el plato (motor NEMA17 + driver TMC2209) y lo **fotografía** con las
   cámaras A (baja, 20°) y B (alta, 55°) desde 18/36/72 ángulos (fotogrametría 3D).
3. Lo ilumina con **8 LED rasantes**, uno por uno (RTI: relieve y marcas).
4. Le saca fotos con **luz ultravioleta** (restauraciones, pegamentos).
5. Lo **calienta suavemente** con la lámpara halógena y filma cómo se enfría con la
   **cámara térmica** (defectos escondidos).
6. Guarda todo en una carpeta `~/yq-data/scans/<id>/` y se la pasa a BOX-ANALYSIS
   (la Mac hace el 3D y la identificación; si la Mac no está, se hace un análisis local
   reducido o solo se informa el peso).

Piezas del software:

| Archivo | Qué hace |
|---|---|
| `firmware/box_esp32/` | Programa del ESP32 (motor, balanza, luces, puertas, seguridad). Protocolo: `PROTOCOL.md` |
| `yq/box/device.py` | Habla con el ESP32 por USB (`get_box()`) |
| `yq/box/sim.py` | Un ESP32 "de mentira" que se comporta igual (para probar sin la caja) |
| `yq/box/cameras.py`, `uvcctl.py` | Cámaras IMX519: foco/exposición bloqueados, fotos revisadas |
| `yq/box/thermal.py` | Cámara térmica PureThermal 3 + Lepton 3.5 (temperaturas en °C) |
| `yq/box/scan.py` | El escaneo completo (`preflight()` y `run_scan()`) |
| `yq/box/layout.py` | Revisa que la carpeta del escaneo esté completa |
| `yq/box/cli.py` | Comandos para probar cada parte |

---

## 1. Conexiones (cableado)

**Regla de oro:** todas las tierras (GND) juntas: GND del ESP32, GND de la fuente de 12 V,
GND del TMC2209 y el "−" de la señal de cada módulo MOSFET. Sin tierra común nada funciona.
El ESP32 se alimenta por su cable USB desde el Jetson (ese mismo cable lleva los datos).

### 1.1 ESP32 (DevKitC-32E, 38 pines) – tabla pin por pin

| Pin ESP32 | Va a | Notas |
|---|---|---|
| **3V3** | HX711 VCC y VDD, TMC2209 VCC_IO, resistencias pull-up (ver abajo) | 3,3 V, nunca 5 V |
| **GND** | tierra común (fuente 12 V −, TMC2209 GND, HX711 GND, "−" de las señales MOSFET) | |
| **GPIO 25** | TMC2209 **STEP** | |
| **GPIO 26** | TMC2209 **DIR** | |
| **GPIO 27** | TMC2209 **EN** + resistencia **10 kΩ de EN a 3V3** | la resistencia deja el motor APAGADO mientras el ESP32 arranca |
| **GPIO 17** (TX2) | resistencia **1 kΩ** → TMC2209 **PDN_UART** | UART de un solo cable |
| **GPIO 16** (RX2) | TMC2209 **PDN_UART** (directo) | el mismo pin que arriba |
| **GPIO 34** | HX711 **DOUT** (DAT) | pin solo-entrada |
| **GPIO 5** | HX711 **SCK** (CLK) | |
| **GPIO 35** | reed de la **puerta de acrílico** (frente) | + **10 kΩ de GPIO35 a 3V3** |
| **GPIO 39** (VN) | reed del **obturador negro** | + **10 kΩ de GPIO39 a 3V3** |
| **GPIO 4, 13, 14, 18, 19, 21, 22, 23** | MOSFET de los LED rasantes **1 a 8** (en ese orden) | PWM |
| **GPIO 32** | MOSFET de la **luz UV** | + **10 kΩ de GPIO32 a GND** (obligatoria) |
| **GPIO 33** | MOSFET de la **halógena** | + **10 kΩ de GPIO33 a GND** (obligatoria) |
| **GPIO 2** | MOSFET de la **tira COB** | |
| **GPIO 15** | MOSFET del **ventilador** | |

No uses GPIO 0, 1, 3, 6–11 ni 12 (arranque, USB y memoria flash del ESP32).

¿Por qué estos pines? UV y halógena están en pines que **no se mueven al arrancar** y
tienen resistencia a GND, así que nunca se encienden mientras el ESP32 se reinicia o se
programa. GPIO 2 y 15 son pines "de arranque", pero el módulo MOSFET solo los tira a GND,
lo que no molesta al arranque. GPIO 14 y 5 pueden parpadear unos milisegundos al arrancar:
a lo sumo el LED rasante 3 da un destello.

### 1.2 Driver TMC2209 (BIGTREETECH V1.3)

| Pin TMC2209 | Va a |
|---|---|
| VM | +12 V de la fuente, con **condensador 100 µF / 35 V** entre VM y GND, pegado al driver (la patita "−" del condensador a GND) |
| GND (lado VM) | GND de la fuente |
| A1, A2 | una bobina del motor |
| B1, B2 | la otra bobina (si el motor vibra sin girar, cambia A2 por A1) |
| VCC_IO | 3V3 del ESP32 |
| GND (lado lógica) | GND común |
| STEP / DIR / EN | GPIO 25 / 26 / 27 |
| PDN_UART (4° pin del lado izquierdo, ya conectado de fábrica en la V1.3) | GPIO 16 directo y GPIO 17 con 1 kΩ |
| MS1, MS2 | GND (dirección UART 0) |

La corriente del motor la pone el ESP32 por UART (1,0 A RMS al girar, 30 % quieto).
El potenciómetro del driver **no** se usa. **Nunca** conectes o desconectes el motor con
la fuente de 12 V encendida: se quema el driver.

### 1.3 Balanza (HX711 + TAL220B 5 kg)

| HX711 | Celda TAL220B |
|---|---|
| E+ | rojo |
| E− | negro |
| A− | blanco |
| A+ | verde |

Si al poner peso el número baja en vez de subir, no pasa nada: la calibración lo corrige
sola (el factor sale negativo).

### 1.4 Módulos MOSFET (D4184) y cargas de 12 V

Cada módulo tiene dos lados:
- **Señal:** `PWM/TRIG` al GPIO de la tabla, `GND` al GND común.
- **Potencia:** `VIN+` a +12 V, `VIN−` a GND de la fuente; la carga va entre `OUT+` y `OUT−`
  (el módulo corta el lado negativo).

| Carga | Conexión entre OUT+ y OUT− |
|---|---|
| LED rasante 3 W (×8) | LED en serie con **resistencia 15 Ω 10 W** (≈0,58 A, ≈5 W en la resistencia: se calienta, déjala al aire) |
| LED UV 365 nm | según su driver/resistencia; el filtro ZWB2 delante de la cámara B cuando llegue |
| Halógena MR16 35 W | directo (≈2,9 A; al encender pide un pico mayor, es normal). Cables de 0,75 mm² o más |
| Tira COB | directo (12 V) |
| Ventilador 40 mm | directo, con **diodo 1N4007 o 1N5819 en paralelo** (raya del diodo hacia +12 V) |

Consumo máximo real: el escaneo nunca enciende la halógena junto con la COB; con la
fuente de 12 V 10 A sobra margen.

### 1.5 Sensores de puerta (reed)

Cada reed: un cable al GPIO (35 o 39) y el otro a **GND**; más la resistencia de 10 kΩ
del GPIO a 3V3. Puerta cerrada = imán cerca = el GPIO lee 0 = "cerrada". Puerta abierta
**o cable cortado** = lee 1 = "abierta": si un cable se suelta, la caja se protege sola.
Si los cables son largos, un condensador de 100 nF del GPIO a GND quita ruido.

---

## 2. Instalar

### En la Mac (desarrollo, simulador, programar el ESP32)

```bash
cd ~/yachachiq/v2
~/.local/bin/uv venv --python 3.10 .venvs/box
~/.local/bin/uv pip install --python .venvs/box/bin/python -r requirements-common.txt -r requirements/box-mac.txt
~/.local/bin/uv tool install platformio          # una sola vez
```

### En el Jetson

```bash
sudo apt install v4l-utils                        # v4l2-ctl para bloquear foco/exposición
sudo usermod -aG dialout,video $USER              # permiso para el USB del ESP32 y las cámaras (cerrar sesión y volver a entrar)
sudo apt remove brltty                            # en Ubuntu 22.04 este programa "roba" los adaptadores USB-serie CH340
cd ~/yachachiq/v2
uv venv --python 3.10 .venvs/box
uv pip install --python .venvs/box/bin/python -r requirements-common.txt -r requirements/box-jetson.txt
```

## 3. Programar (flashear) el ESP32

**Desde la Mac:** conecta el ESP32 por USB y:

```bash
cd ~/yachachiq/v2/firmware/box_esp32
~/.local/bin/pio run                 # compila (la primera vez descarga ~1 GB y tarda)
~/.local/bin/pio run -t upload       # lo carga en el ESP32 (encuentra el puerto solo)
~/.local/bin/pio device monitor      # ver lo que dice (Ctrl+C para salir)
```

Al arrancar debe aparecer una línea `{"event":"boot","fw":"yq-box",...}`.
Si la carga falla con "Failed to connect": mantén apretado el botón **BOOT** del ESP32
mientras empieza a cargar. Si no aparece ningún puerto en la Mac, instala el driver
CP210x de Silicon Labs (o CH340 si tu placa es un clon).

**Desde el Jetson:** igual que en la Mac (`uv tool install platformio` y los mismos
comandos; PlatformIO tiene compilador para Linux ARM64). El puerto será `/dev/ttyUSB0`.

**Antes de flashear:** apaga la fuente de 12 V (así nada de potencia se mueve mientras el
ESP32 se reprograma). El entorno `qemu` de `platformio.ini` es solo para el emulador: **no
lo cargues nunca en la placa** (no mueve el motor).

## 4. Probar cada parte, paso a paso (antes de un escaneo completo)

Todo se hace con `python -m yq.box.cli ...` desde `~/yachachiq/v2` con
`.venvs/box/bin/python`. Agrega `--mock` para practicar con el simulador (sin caja).

1. **Comunicación:** `ping` → "Caja OK en /dev/ttyUSB0". Luego `status`.
   Si dice "TMC2209 NO RESPONDE": revisa 12 V en VM, el cable PDN_UART y la resistencia de 1 kΩ.
2. **Puertas:** con `status`, abre y cierra cada puerta; debe cambiar "cerradas"/"ABIERTA(S)".
3. **Luces una por una** (el número final es cuántos milisegundos queda prendida, y
   `--hold` mantiene vivo el programa ese tiempo; si el programa se cierra, el ESP32 apaga
   todo a los 3 s por seguridad):
   `light rake1 1.0 2000 --hold 2`, … hasta `rake8`; `light cob 1.0 3000 --hold 3`;
   `light fan 1.0 3000 --hold 3`.
4. **UV y halógena** (puertas cerradas): `light uv 1.0 2000 --hold 2`, `light halogen 1.0 3000 --hold 3`.
   Prueba de seguridad: prende la halógena con `--hold 10` y abre una puerta: debe apagarse al instante.
   Con la puerta abierta el comando debe dar error "Hay una puerta abierta".
5. **Motor:** marca con cinta una rayita en el plato y otra fija en la caja.
   `rotate 90` (debe girar un cuarto de vuelta suave), `rotate 270` (vuelve a la marca).
   `rotate 360` diez veces: la marca no debe correrse. Si se corre, la correa salta:
   baja la aceleración (`config accel=30 --save`) o tensa la correa.
6. **Balanza:** ver sección 5.
7. **Cámaras:** ver sección 6. `capture-background` con el plato **vacío**.
8. **Térmica:** `thermal-test --seconds 5` → ~8,7 fps y temperatura ambiente razonable.
9. **Escaneo completo:** objeto en el plato, puertas cerradas: `scan --profile quick`.
   La carpeta queda en `~/yq-data/scans/`.

## 5. Calibrar la balanza

1. Plato **vacío**, puertas cerradas, sin tocar la mesa: `tare`.
2. Pon un peso **conocido** en el centro (ideal 500 g a 1000 g; por ejemplo una botella de
   agua pesada antes en una balanza de cocina buena): `calibrate-scale 500` (el número es
   el peso real en gramos).
3. Comprueba: `weigh` cinco veces seguidas; y con el mismo peso en el centro y en el borde
   del plato. Anota las diferencias aquí abajo (sección 8).

La tara y el factor se guardan **dentro del ESP32** (memoria NVS): no se pierden al
apagar. Repite la tara si cambias algo del plato. El motor y el ventilador se apagan
solos mientras pesa (vibran). "No estable" = algo vibra (ventilador, mesa, aire).

## 6. Cámaras

- Las dos IMX519 se llaman igual por USB, así que se identifican por el **puerto USB**
  donde están enchufadas: `ls -l /dev/v4l/by-path/` y anota cuál es la A (baja) y la B
  (alta). Luego, en el servicio del robot:
  `export YQ_BOX_CAMERA_A=/dev/v4l/by-path/...-video-index0` (y `_B`). No cambies los
  cables de puerto después.
- Ver sus controles: `v4l2-ctl -d $YQ_BOX_CAMERA_A --list-ctrls-menus`.
- **Foco:** con un objeto en el plato y la COB prendida: `focus-sweep A` y `focus-sweep B`.
  Elige el mejor número y guárdalo: `export YQ_BOX_FOCUS_A=...`, `YQ_BOX_FOCUS_B=...`.
  Durante el escaneo el autofoco está **apagado** y el foco fijo (si no, el 3D sale mal).
- Exposición, ganancia y balance de blancos también quedan fijos (`YQ_BOX_EXPOSURE_A_US`,
  etc.). La exposición UVC de esta cámara va en pasos de 0,1 ms y su máximo es 200 ms.
- Nunca se usan las dos cámaras a la vez (el USB no da abasto): se abre una, se toman
  sus fotos y se cierra.
- Fotos de calibración para BOX-ANALYSIS: `capture-calib-set` (tablero ChArUco en el plato)
  y `capture-rti-sphere` (esfera negra brillante en el plato). Quedan en `~/yq-data/calibration/box/`.

## 7. Seguridad (léelo antes de encender la halógena y la UV)

- **Halógena MR16 35 W:** se pone MUY caliente (más de 200 °C en el vidrio). Solo dentro de
  su luminaria metálica, con la placa de aluminio y el espacio de aire al techo de PETG.
  No la toques hasta 10 minutos después. El firmware la deja prendida **máximo 45 s** y
  luego la obliga a enfriarse **3 veces** ese tiempo; el escaneo usa 15 s.
- **Luz UV 365 nm:** daña los ojos aunque "no se vea fuerte". **Nunca** la mires. Solo
  prende con las dos puertas cerradas (el firmware lo impide si no) y se apaga sola al
  abrir una puerta. **Nunca puentees los sensores de puerta.**
- LED rasantes: máximo 20 s encendidos y luego descanso del doble (no tienen disipador).
- Si el Jetson se cuelga o se desconecta el USB, a los **3 s** el ESP32 apaga todo y suelta el motor.
- Objetos arqueológicos reales: pregunta a un conservador antes de calentar una pieza;
  empieza con piezas modernas de cerámica. La subida de temperatura real hay que medirla.
- Cambios de cables siempre con la fuente de 12 V **apagada**.

## 8. Resultados medidos (y lo que falta medir)

Medido en la Mac, sin la caja (simulador + cámaras sintéticas):
- 59 pruebas automáticas pasan (`tests/box` + `tests/common`, ~32 s).
- Firmware compilado: RAM 8,9 % (29 268 B), Flash 27,6 % (361 805 B).
- Firmware real corriendo en el emulador QEMU de Espressif: protocolo, luces con tiempo
  máximo, enfriamiento (2× → 586 ms tras 300 ms), exclusión de rasantes, interlock
  "puertas cerradas", límites duros de configuración y errores de protocolo: 22 comprobaciones
  OK (`tools/box_qemu.py`). Guardar la configuración en NVS + reinicio NO se pudo comprobar en
  QEMU (el emulador se reinicia al escribir la flash): hay que probarlo en la placa.
- Resolución del plato: 44 800 pasos por vuelta = 0,008° por paso.
- Escaneo "quick" simulado completo: ~8 s a 50× de velocidad (en la caja real: varios minutos).

**Falta medir con la caja real (noviembre):** precisión y repetibilidad de la balanza
(objetivo ±0,5 g), si la correa salta, nitidez real de las fotos, foco óptimo, tiempo real
de un escaneo, temperatura máxima del objeto con 15 s de halógena, latencia real del corte
UV/halógena al abrir la puerta (el código la corta en la interrupción, microsegundos).
