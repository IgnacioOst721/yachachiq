// Runtime configuration (limits, motion, driver) with safe defaults and hard caps.
// Mirrored in v2/yq/box/settings.py (FIRMWARE_DEFAULTS) and the simulator.
#pragma once
#include <ArduinoJson.h>
#include <stdint.h>

#define FW_NAME "yq-box"
#define FW_VERSION "2.0.0"
#define PROTO_VERSION 1

enum LightKind : uint8_t { K_RAKE = 0, K_UV, K_HALOGEN, K_COB, K_FAN, K_COUNT };
static const char* const KIND_NAMES[K_COUNT] = {"rake", "uv", "halogen", "cob", "fan"};

// Hard caps: config_set can never go beyond these (compiled in).
static const uint32_t HARD_MAX_ON_MS[K_COUNT] = {30000, 60000, 60000, 1800000, 0};
static const float HARD_MIN_COOL[K_COUNT] = {1.0f, 1.0f, 2.0f, 0.0f, 0.0f};

struct BoxConfig {
  uint32_t max_on_ms[K_COUNT] = {20000, 30000, 45000, 900000, 0};  // 0 = unlimited (fan)
  float cool_factor[K_COUNT] = {2.0f, 2.0f, 3.0f, 0.5f, 0.0f};
  uint32_t hb_timeout_ms = 3000;
  float dps = 30.0f;          // turntable cruise speed, deg/s
  float accel = 45.0f;        // turntable acceleration, deg/s^2
  float max_dps = 90.0f;
  float max_accel = 180.0f;
  uint16_t motor_steps = 200;
  uint16_t microsteps = 16;
  float ratio = 14.0f;        // 280T crown / 20T pulley
  uint16_t run_ma = 1000;     // TMC2209 RMS run current
  uint8_t hold_pct = 30;      // hold current, % of run
  bool door_stops_motor = true;
  bool allow_no_uart = false;

  uint32_t stepsPerRev() const { return (uint32_t)motor_steps * microsteps * ratio + 0.5f; }
  void clamp();
  void toJson(JsonObject o) const;
  // Applies the keys present in `in`; returns false (and sets err) on a bad value.
  bool fromJson(JsonObjectConst in, const char** err);
  void load();   // from NVS (namespace "yqbox"), keeps defaults for missing keys
  void save() const;
};

extern BoxConfig cfg;
