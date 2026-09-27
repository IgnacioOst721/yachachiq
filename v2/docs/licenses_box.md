# Licencias – BOX-CAPTURE (captura de la caja de análisis)

Todo es software libre. No hay modelos de IA ni datasets en esta parte.
Nada necesita internet cuando el robot funciona (solo para instalar).

## Firmware del ESP32 (`firmware/box_esp32`)

| Componente | Versión | Licencia | Uso |
|---|---|---|---|
| Arduino-ESP32 (core) | 2.0.17 (vía `platformio/espressif32` 6.13.0) | LGPL-2.1-or-later | framework del ESP32 |
| ESP-IDF | 4.4.7 (dentro del core) | Apache-2.0 | watchdog, GPIO, NVS |
| PlatformIO espressif32 platform | 6.13.0 | Apache-2.0 | compilación |
| PlatformIO Core | 6.2.0 | Apache-2.0 | herramienta de compilación (no va en el robot) |
| FastAccelStepper (gin66) | 1.3.4 | MIT | pulsos STEP/DIR con aceleración |
| TMCStepper (teemuatlut) | 0.7.3 | MIT | configurar el TMC2209 por UART |
| ArduinoJson (bblanchon) | 7.4.3 | MIT | protocolo JSON |
| HX711 (bogde) | 0.7.5 | MIT | leer la celda de carga |

## Python (Jetson y Mac)

| Paquete | Versión | Licencia | Uso |
|---|---|---|---|
| pyserial | 3.5 | BSD-3-Clause | puerto serie del ESP32 |
| numpy | (requirements-common) | BSD-3-Clause | imágenes, secuencia térmica |
| opencv-contrib-python-headless | (requirements-common) | Apache-2.0 | cámaras UVC, JPEG |
| v4l-utils (`v4l2-ctl`, paquete apt) | la de Ubuntu 22.04 | GPL-2.0 (herramienta) / LGPL-2.1 (libv4l) | bloquear foco/exposición |

## Herramientas de desarrollo (no van en el robot)

| Herramienta | Licencia | Uso |
|---|---|---|
| Espressif QEMU (esp-develop-9.2.2-20260417) | GPL-2.0 | probar el firmware real sin placa (`tools/box_qemu.py`) |
| uv | Apache-2.0 / MIT | instalar Python y PlatformIO |

## Referencias usadas para verificar datos (no se copió código)

- GroupGets `purethermal1-uvc-capture` (conversión centikelvin → °C): MIT.
- Documentación de Arducam (unidades de exposición UVC 100 µs), BIGTREETECH (R_sense 0,11 Ω,
  pin PDN_UART), Watterott (pull-downs de MS1/MS2), hoja de datos TMC2209 (Analog Devices).
