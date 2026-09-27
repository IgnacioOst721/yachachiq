// Yachachiq analysis box controller (ESP32-DevKitC-32E, Arduino core 2.0.17).
// JSON-lines protocol: PROTOCOL.md. Safety summary:
//  1. all outputs LOW before setup() (global constructor + initVariant());
//  2. UV + halogen door interlock (GPIO interrupt + loop re-check);
//  3. per-channel max on-time and cool-down;
//  4. host heartbeat watchdog (3 s);
//  5. ESP-IDF task watchdog on the loop (5 s, panic -> reset; sdkconfig of the
//     Arduino core: CONFIG_ESP_TASK_WDT_PANIC=y, TIMEOUT_S=5).
#include <Arduino.h>
#include <driver/gpio.h>
#include <esp_rom_gpio.h>
#include <esp_log.h>
#include <esp_task_wdt.h>

#include "boxconfig.h"
#include "commands.h"
#include "lights.h"
#include "motor.h"
#include "pins.h"
#include "proto.h"
#include "safety.h"
#include "scale.h"

// Runs from the C++ global-constructor table, before app_main()/initArduino():
// plain register writes through the GPIO driver, no allocation, no logging.
static void forceOutputsSafe() {
  for (uint8_t pin : ALL_OUTPUT_PINS) {
    esp_rom_gpio_pad_select_gpio(pin);
    gpio_set_level((gpio_num_t)pin, 0);
    gpio_set_direction((gpio_num_t)pin, GPIO_MODE_OUTPUT);
  }
  esp_rom_gpio_pad_select_gpio(PIN_EN);
  gpio_set_level((gpio_num_t)PIN_EN, 1);  // TMC2209 disabled
  gpio_set_direction((gpio_num_t)PIN_EN, GPIO_MODE_OUTPUT);
}

__attribute__((constructor)) static void earlySafeOutputs() { forceOutputsSafe(); }

// Weak hook called by initArduino() right before setup(): do it again with the
// Arduino API in case the core touched a pin in between.
void initVariant() {
  for (uint8_t pin : ALL_OUTPUT_PINS) {
    digitalWrite(pin, LOW);
    pinMode(pin, OUTPUT);
  }
  digitalWrite(PIN_EN, HIGH);
  pinMode(PIN_EN, OUTPUT);
}

static char s_line[520];

void setup() {
  esp_log_level_set("*", ESP_LOG_NONE);  // ESP-IDF logs share UART0: keep the JSON stream clean
  Serial.setRxBufferSize(1024);
  Serial.setTxBufferSize(4096);          // a full status reply (~1.2 kB) must not block the loop
  Serial.begin(115200);
  cfg.load();
  lightsBegin();    // attaches LEDC with duty 0; UV/halogen stay LOW
  safetyBegin();    // reed pins + interrupt
  scaleBegin();
  motorBegin();     // TMC2209 over UART, stepper disabled
  enableLoopWDT();  // adds the loop task to the task watchdog (reset on hang)

  JsonDocument d = protoEventDoc("boot");
  d["fw"] = FW_NAME;
  d["version"] = FW_VERSION;
  d["proto"] = PROTO_VERSION;
  d["reset_reason"] = commandsResetReason();
  protoSend(d);
}

void loop() {
  uint32_t now = millis();
  safetyLoop(now);  // doors first: interlock cut, door events, heartbeat
  bool tooLong = false;
  if (protoPoll(s_line, sizeof s_line, &tooLong)) {
    if (tooLong) protoError(-1, "line_too_long");
    else commandsHandle(s_line, millis());
  }
  now = millis();
  lightsLoop(now);
  motorLoop(now);
  scaleLoop(now);
  delay(1);  // lets the idle task run (and feed its own watchdog)
}
