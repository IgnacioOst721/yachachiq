// Door interlocks: closed door = magnet = reed closed to GND = LOW.
// Open door or broken wire = HIGH through the external 10k pull-up = OPEN.
// The interrupt cuts UV and halogen by writing the GPIO "write 1 to clear"
// register for pins 32..39 directly (IRAM-safe, no driver call).
#include "safety.h"

#include <Arduino.h>
#include <soc/gpio_struct.h>

#include "boxconfig.h"
#include "lights.h"
#include "motor.h"
#include "pins.h"
#include "proto.h"

static_assert(PIN_UV >= 32 && PIN_HALOGEN >= 32, "ISR cut uses the GPIO32..39 registers");
static_assert(PIN_REED_FRONT >= 32 && PIN_REED_SHUTTER >= 32, "ISR reads the GPIO32..39 input register");

static const uint32_t CUT_BITS = (1u << (PIN_UV - 32)) | (1u << (PIN_HALOGEN - 32));
static const uint32_t REED_BITS = (1u << (PIN_REED_FRONT - 32)) | (1u << (PIN_REED_SHUTTER - 32));
static const uint32_t CLOSE_DEBOUNCE_MS = 50;

static volatile uint32_t s_cutMask = 0;
static const uint8_t REED_PINS[2] = {PIN_REED_FRONT, PIN_REED_SHUTTER};
static bool s_closed[2] = {false, false};
static uint32_t s_lowSince[2] = {0, 0};

static uint32_t s_hbLast = 0;
static bool s_hbArmed = false;
static bool s_hbLost = false;

static void IRAM_ATTR onReedEdge() {
  if (GPIO.in1.val & REED_BITS) {           // at least one reed reads HIGH = open
    uint32_t on = GPIO.out1.val & CUT_BITS;  // which of UV/halogen were on
    GPIO.out1_w1tc.val = CUT_BITS;           // cut both, unconditionally
    uint32_t mask = 0;
    if (on & (1u << (PIN_UV - 32))) mask |= 1;
    if (on & (1u << (PIN_HALOGEN - 32))) mask |= 2;
    s_cutMask |= mask;
  }
}

void safetyBegin() {
  for (int i = 0; i < 2; i++) {
    pinMode(REED_PINS[i], INPUT);  // input-only pins: external pull-ups
    attachInterrupt(REED_PINS[i], onReedEdge, CHANGE);
  }
}

bool anyDoorRawOpen() { return (GPIO.in1.val & REED_BITS) != 0; }

uint32_t takeIsrCutMask() {
  noInterrupts();
  uint32_t m = s_cutMask;
  s_cutMask = 0;
  interrupts();
  return m;
}

bool doorClosed(int door) { return s_closed[door]; }
bool doorsClosed() { return s_closed[0] && s_closed[1]; }

static void emitDoor(int i) {
  JsonDocument d = protoEventDoc("door");
  d["door"] = DOOR_NAMES[i];
  d["closed"] = s_closed[i];
  protoSend(d);
}

void safetyLoop(uint32_t now) {
  for (int i = 0; i < 2; i++) {
    bool low = digitalRead(REED_PINS[i]) == LOW;
    if (!low) {
      s_lowSince[i] = 0;
      if (s_closed[i]) {
        s_closed[i] = false;
        emitDoor(i);
        if (cfg.door_stops_motor) motorStop(false);
      }
    } else {
      if (s_lowSince[i] == 0) s_lowSince[i] = now ? now : 1;
      if (!s_closed[i] && now - s_lowSince[i] >= CLOSE_DEBOUNCE_MS) {
        s_closed[i] = true;
        emitDoor(i);
      }
    }
  }
  // Second path: if the ISR was missed (e.g. during a flash write) cut here,
  // and report what the ISR already cut.
  lightsInterlock(anyDoorRawOpen() || !doorsClosed(), takeIsrCutMask(), now);

  if (s_hbArmed && !s_hbLost && now - s_hbLast > cfg.hb_timeout_ms) {
    s_hbLost = true;
    lightsAllOff("watchdog");
    motorStop(true);
    motorEnable(false);
    JsonDocument d = protoEventDoc("watchdog");
    d["reason"] = "heartbeat";
    protoSend(d);
  }
}

void hbTouch(uint32_t now) {
  s_hbLast = now;
  s_hbArmed = true;
  s_hbLost = false;
}
bool hbLost() { return s_hbLost; }
bool hbArmed() { return s_hbArmed; }
uint32_t hbAgeMs(uint32_t now) { return s_hbArmed ? now - s_hbLast : 0; }
