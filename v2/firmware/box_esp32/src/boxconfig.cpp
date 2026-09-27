// Configuration: defaults, clamping to hard caps, JSON and NVS persistence.
#include "boxconfig.h"

#include <Preferences.h>

BoxConfig cfg;

static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

void BoxConfig::clamp() {
  for (int k = 0; k < K_COUNT; k++) {
    if (HARD_MAX_ON_MS[k] > 0) {
      if (max_on_ms[k] == 0 || max_on_ms[k] > HARD_MAX_ON_MS[k]) max_on_ms[k] = HARD_MAX_ON_MS[k];
    }
    if (cool_factor[k] < HARD_MIN_COOL[k]) cool_factor[k] = HARD_MIN_COOL[k];
    if (cool_factor[k] > 20.0f) cool_factor[k] = 20.0f;
  }
  if (hb_timeout_ms < 500) hb_timeout_ms = 500;
  if (hb_timeout_ms > 10000) hb_timeout_ms = 10000;
  max_dps = clampf(max_dps, 1.0f, 120.0f);
  max_accel = clampf(max_accel, 1.0f, 360.0f);
  dps = clampf(dps, 0.5f, max_dps);
  accel = clampf(accel, 0.5f, max_accel);
  if (motor_steps != 200 && motor_steps != 400) motor_steps = 200;
  if (microsteps != 8 && microsteps != 16 && microsteps != 32 && microsteps != 64) microsteps = 16;
  ratio = clampf(ratio, 1.0f, 100.0f);
  if (run_ma < 300) run_ma = 300;
  if (run_ma > 1400) run_ma = 1400;
  if (hold_pct > 60) hold_pct = 60;
}

void BoxConfig::toJson(JsonObject o) const {
  JsonObject on = o["max_on_ms"].to<JsonObject>();
  JsonObject cf = o["cool_factor"].to<JsonObject>();
  for (int k = 0; k < K_COUNT; k++) {
    on[KIND_NAMES[k]] = max_on_ms[k];
    cf[KIND_NAMES[k]] = cool_factor[k];
  }
  o["hb_timeout_ms"] = hb_timeout_ms;
  o["dps"] = dps;
  o["accel"] = accel;
  o["max_dps"] = max_dps;
  o["max_accel"] = max_accel;
  o["motor_steps"] = motor_steps;
  o["microsteps"] = microsteps;
  o["ratio"] = ratio;
  o["steps_per_rev"] = stepsPerRev();
  o["run_ma"] = run_ma;
  o["hold_pct"] = hold_pct;
  o["door_stops_motor"] = door_stops_motor;
  o["allow_no_uart"] = allow_no_uart;
}

bool BoxConfig::fromJson(JsonObjectConst in, const char** err) {
  BoxConfig next = *this;
  JsonObjectConst on = in["max_on_ms"];
  JsonObjectConst cf = in["cool_factor"];
  for (int k = 0; k < K_COUNT; k++) {
    if (!on.isNull() && !on[KIND_NAMES[k]].isNull()) {
      if (!on[KIND_NAMES[k]].is<float>()) { *err = "max_on_ms"; return false; }
      float v = on[KIND_NAMES[k]].as<float>();
      if (v < 0) { *err = "max_on_ms"; return false; }
      next.max_on_ms[k] = (uint32_t)v;
    }
    if (!cf.isNull() && !cf[KIND_NAMES[k]].isNull()) {
      if (!cf[KIND_NAMES[k]].is<float>()) { *err = "cool_factor"; return false; }
      next.cool_factor[k] = cf[KIND_NAMES[k]].as<float>();
    }
  }
#define NUM_KEY(name, type)                                   \
  if (!in[#name].isNull()) {                                  \
    if (!in[#name].is<float>()) { *err = #name; return false; } \
    next.name = (type)in[#name].as<float>();                  \
  }
  NUM_KEY(hb_timeout_ms, uint32_t)
  NUM_KEY(dps, float)
  NUM_KEY(accel, float)
  NUM_KEY(max_dps, float)
  NUM_KEY(max_accel, float)
  NUM_KEY(motor_steps, uint16_t)
  NUM_KEY(microsteps, uint16_t)
  NUM_KEY(ratio, float)
  NUM_KEY(run_ma, uint16_t)
  NUM_KEY(hold_pct, uint8_t)
#undef NUM_KEY
  if (!in["door_stops_motor"].isNull()) next.door_stops_motor = in["door_stops_motor"].as<bool>();
  if (!in["allow_no_uart"].isNull()) next.allow_no_uart = in["allow_no_uart"].as<bool>();
  next.clamp();
  *this = next;
  return true;
}

void BoxConfig::load() {
  Preferences p;
  if (p.begin("yqbox", true)) {
    for (int k = 0; k < K_COUNT; k++) {
      char key[16];
      snprintf(key, sizeof key, "on_%s", KIND_NAMES[k]);
      max_on_ms[k] = p.getUInt(key, max_on_ms[k]);
      snprintf(key, sizeof key, "cf_%s", KIND_NAMES[k]);
      cool_factor[k] = p.getFloat(key, cool_factor[k]);
    }
    hb_timeout_ms = p.getUInt("hb_ms", hb_timeout_ms);
    dps = p.getFloat("dps", dps);
    accel = p.getFloat("accel", accel);
    max_dps = p.getFloat("max_dps", max_dps);
    max_accel = p.getFloat("max_accel", max_accel);
    motor_steps = p.getUInt("msteps", motor_steps);
    microsteps = p.getUInt("usteps", microsteps);
    ratio = p.getFloat("ratio", ratio);
    run_ma = p.getUInt("run_ma", run_ma);
    hold_pct = p.getUInt("hold_pct", hold_pct);
    door_stops_motor = p.getBool("door_stop", door_stops_motor);
    allow_no_uart = p.getBool("no_uart", allow_no_uart);
    p.end();
  }
  clamp();
}

void BoxConfig::save() const {
  Preferences p;
  if (!p.begin("yqbox", false)) return;
  for (int k = 0; k < K_COUNT; k++) {
    char key[16];
    snprintf(key, sizeof key, "on_%s", KIND_NAMES[k]);
    p.putUInt(key, max_on_ms[k]);
    snprintf(key, sizeof key, "cf_%s", KIND_NAMES[k]);
    p.putFloat(key, cool_factor[k]);
  }
  p.putUInt("hb_ms", hb_timeout_ms);
  p.putFloat("dps", dps);
  p.putFloat("accel", accel);
  p.putFloat("max_dps", max_dps);
  p.putFloat("max_accel", max_accel);
  p.putUInt("msteps", motor_steps);
  p.putUInt("usteps", microsteps);
  p.putFloat("ratio", ratio);
  p.putUInt("run_ma", run_ma);
  p.putUInt("hold_pct", hold_pct);
  p.putBool("door_stop", door_stops_motor);
  p.putBool("no_uart", allow_no_uart);
  p.end();
}
