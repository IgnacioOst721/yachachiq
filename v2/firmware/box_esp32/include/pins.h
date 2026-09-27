// Pin map of the analysis box controller: ESP32-DevKitC-32E (38 pins, ESP32-WROOM-32E).
// Wiring table and the reasons for every choice: v2/docs/box.md.
//
// Boot behaviour checked against the ESP32 datasheet / strapping-pin notes:
//  * UV (32) and HALOGEN (33) are NOT strapping pins and are high-impedance
//    during reset, bootloader and flashing -> a 10k pull-down on their MOSFET
//    inputs keeps them off until the firmware drives them LOW.
//  * GPIO 2 (COB) and 15 (FAN) are strapping pins; a MOSFET module input
//    (LED + 1k to GND, or a 10k pull-down) only pulls them LOW, which is a legal
//    boot level for both (GPIO15 LOW just silences the ROM boot log).
//  * GPIO 5 (HX711 SCK) and 14 (RAKE3) may toggle for a few ms at boot:
//    harmless (HX711 resets itself, a raking LED blinks once).
//  * 34, 35, 39 are input-only without internal pull-ups: the reeds need
//    external 10k pull-ups to 3V3. GPIO 39 can see an ~80 ns LOW glitch when
//    the ADC/Wi-Fi power up (ESP32 errata); we use neither, and a "closed"
//    reading must be stable for 50 ms before it counts.
//  * 6..11 (SPI flash), 0/1/3 (boot/USB serial) and 12 (flash voltage strap)
//    are not used.
#pragma once
#include <stdint.h>

#define PIN_STEP 25
#define PIN_DIR 26
#define PIN_EN 27            // TMC2209 ENN: LOW = driver on. External 10k pull-up to 3V3.
#define PIN_TMC_TX 17        // Serial2 TX -> 1k resistor -> PDN_UART (single wire)
#define PIN_TMC_RX 16        // Serial2 RX <- PDN_UART (direct)
#define PIN_HX_DOUT 34       // HX711 DOUT (input only)
#define PIN_HX_SCK 5         // HX711 PD_SCK
#define PIN_REED_FRONT 35    // front acrylic door reed (closed = LOW), 10k pull-up
#define PIN_REED_SHUTTER 39  // opaque shutter reed (closed = LOW), 10k pull-up
#define PIN_UV 32            // UV 365 nm LED MOSFET (plain GPIO, cut by the door ISR)
#define PIN_HALOGEN 33       // MR16 35 W halogen MOSFET (plain GPIO, cut by the door ISR)
#define PIN_COB 2            // COB strip MOSFET (PWM)
#define PIN_FAN 15           // 40 mm fan MOSFET (PWM)

#define NUM_RAKE 8
static const uint8_t RAKE_PINS[NUM_RAKE] = {4, 13, 14, 18, 19, 21, 22, 23};

// Every output pin, in the order they are forced LOW at boot.
static const uint8_t ALL_OUTPUT_PINS[] = {PIN_UV, PIN_HALOGEN, PIN_COB, PIN_FAN,
                                          4, 13, 14, 18, 19, 21, 22, 23,
                                          PIN_STEP, PIN_DIR};
