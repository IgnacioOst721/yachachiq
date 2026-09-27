// Request dispatcher: one JSON object in, exactly one reply out (see PROTOCOL.md).
#include "commands.h"

#include <Arduino.h>
#include <esp_system.h>

#include "boxconfig.h"
#include "lights.h"
#include "motor.h"
#include "proto.h"
#include "safety.h"
#include "scale.h"

static const char* resetReason() {
  switch (esp_reset_reason()) {
    case ESP_RST_POWERON: return "poweron";
    case ESP_RST_SW: return "software";
    case ESP_RST_PANIC: return "panic";
    case ESP_RST_INT_WDT: return "int_wdt";
    case ESP_RST_TASK_WDT: return "task_wdt";
    case ESP_RST_WDT: return "wdt";
    case ESP_RST_BROWNOUT: return "brownout";
    case ESP_RST_EXT: return "external";
    default: return "other";
  }
}

const char* commandsResetReason() { return resetReason(); }

static JsonDocument okDoc(long id) {
  JsonDocument d;
  d["id"] = id;
  d["ok"] = true;
  return d;
}

static void statusInto(JsonDocument& d, uint32_t now) {
  d["uptime_ms"] = now;
  d["hb_age_ms"] = hbAgeMs(now);
  d["hb_lost"] = hbLost();
  d["tx_dropped"] = protoDropped();
  JsonObject doors = d["doors"].to<JsonObject>();
  doors["front"] = doorClosed(DOOR_FRONT);
  doors["shutter"] = doorClosed(DOOR_SHUTTER);
  d["doors_closed"] = doorsClosed();
  motorStatus(d["motor"].to<JsonObject>());
  lightsStatus(d["lights"].to<JsonObject>(), now);
  scaleStatus(d["scale"].to<JsonObject>());
  JsonArray f = d["faults"].to<JsonArray>();
  if (!motorTmcOk()) f.add("tmc_uart");
  if (!scalePresent()) f.add("hx711");
}

static void motionReply(long id, const char* err, uint32_t eta, double target) {
  if (err) return protoError(id, err);
  JsonDocument d = okDoc(id);
  d["target_deg"] = target;
  d["eta_ms"] = eta;
  protoSend(d);
}

void commandsHandle(const char* line, uint32_t now) {
  JsonDocument req;
  DeserializationError e = deserializeJson(req, line);
  if (e || !req.is<JsonObject>()) return protoError(-1, "bad_json", e ? e.c_str() : "not an object");
  long id = req["id"] | -1L;
  const char* cmd = req["cmd"] | "";
  hbTouch(now);

  if (!strcmp(cmd, "ping") || !strcmp(cmd, "hb")) {
    JsonDocument d = okDoc(id);
    if (cmd[0] == 'p') d["pong"] = true;
    d["uptime_ms"] = now;
    return protoSend(d);
  }
  if (!strcmp(cmd, "info")) {
    JsonDocument d = okDoc(id);
    d["fw"] = FW_NAME;
    d["version"] = FW_VERSION;
    d["proto"] = PROTO_VERSION;
    d["board"] = "esp32dev";
    JsonObject t = d["tmc"].to<JsonObject>();
    t["ok"] = motorTmcOk();
    t["version"] = motorTmcVersion();
    d["hx711"] = scalePresent();
    d["steps_per_rev"] = cfg.stepsPerRev();
    JsonArray ch = d["channels"].to<JsonArray>();
    for (int i = 0; i < NUM_CHANNELS; i++) ch.add(lightsName(i));
    d["reset_reason"] = resetReason();
    return protoSend(d);
  }
  if (!strcmp(cmd, "status")) {
    JsonDocument d = okDoc(id);
    statusInto(d, now);
    return protoSend(d);
  }
  if (!strcmp(cmd, "motor")) {
    if (!req["enable"].is<bool>()) return protoError(id, "bad_param", "enable: bool");
    motorEnable(req["enable"].as<bool>());
    JsonDocument d = okDoc(id);
    d["enabled"] = motorEnabled();
    return protoSend(d);
  }
  if (!strcmp(cmd, "rotate_to") || !strcmp(cmd, "rotate_by")) {
    if (!req["deg"].is<float>()) return protoError(id, "bad_param", "deg: number");
    if (scaleBusy()) return protoError(id, "scale_busy");
    if (cfg.door_stops_motor && (!doorsClosed() || anyDoorRawOpen())) return protoError(id, "interlock", "door open");
    uint32_t eta = 0;
    double target = 0, deg = req["deg"].as<double>();
    const char* err = cmd[7] == 't' ? motorRotateTo(deg, req["wrap"] | false, id, &eta, &target)
                                    : motorRotateBy(deg, id, &eta, &target);
    return motionReply(id, err, eta, target);
  }
  if (!strcmp(cmd, "speed")) {
    if (!req["dps"].isNull() && !req["dps"].is<float>()) return protoError(id, "bad_param");
    if (!req["accel"].isNull() && !req["accel"].is<float>()) return protoError(id, "bad_param");
    if (!req["dps"].isNull()) cfg.dps = req["dps"].as<float>();
    if (!req["accel"].isNull()) cfg.accel = req["accel"].as<float>();
    cfg.clamp();
    motorApplySpeed();
    JsonDocument d = okDoc(id);
    d["dps"] = cfg.dps;
    d["accel"] = cfg.accel;
    return protoSend(d);
  }
  if (!strcmp(cmd, "stop")) {
    motorStop(req["hard"] | false);
    JsonDocument d = okDoc(id);
    d["deg"] = motorDeg();
    return protoSend(d);
  }
  if (!strcmp(cmd, "zero")) {
    const char* err = motorZero();
    if (err) return protoError(id, err);
    JsonDocument d = okDoc(id);
    d["deg"] = 0;
    return protoSend(d);
  }
  if (!strcmp(cmd, "scale_tare") || !strcmp(cmd, "scale_cal") || !strcmp(cmd, "weigh")) {
    int op = !strcmp(cmd, "weigh") ? SCALE_WEIGH : (!strcmp(cmd, "scale_cal") ? SCALE_CAL : SCALE_TARE);
    const char* err = scaleStart(op, id, req["n"] | 20, req["stable_g"] | 0.5f, req["timeout_ms"] | 8000UL,
                                 req["grams"] | 0.0f, now);
    if (err) protoError(id, err);
    return;  // the reply comes from scaleLoop() when the measurement ends
  }
  if (!strcmp(cmd, "light")) {
    int ch = lightsFind(req["ch"] | "");
    if (ch < 0 || !req["level"].is<float>()) return protoError(id, "bad_param", "ch, level");
    uint32_t offIn = 0, wait = 0;
    const char* err = lightsSet(ch, req["level"].as<float>(), req["max_ms"] | 0UL, now, &offIn, &wait);
    if (err) {
      JsonDocument d;
      d["id"] = id;
      d["ok"] = false;
      d["error"] = err;
      if (wait) d["wait_ms"] = wait;
      return protoSend(d);
    }
    JsonDocument d = okDoc(id);
    d["ch"] = lightsName(ch);
    d["level"] = req["level"].as<float>() > 0 && (ch == CH_UV || ch == CH_HALOGEN) ? 1.0f : req["level"].as<float>();
    d["off_in_ms"] = offIn;
    return protoSend(d);
  }
  if (!strcmp(cmd, "all_off") || !strcmp(cmd, "estop")) {
    JsonDocument d = okDoc(id);
    JsonArray cut = d["cut"].to<JsonArray>();
    lightsAllOff(nullptr, &cut);
    if (cmd[0] == 'e') {
      motorStop(true);
      motorEnable(false);
      scaleAbort();
    }
    return protoSend(d);
  }
  if (!strcmp(cmd, "config_get") || !strcmp(cmd, "config_set")) {
    bool reboot = false;
    if (cmd[7] == 's') {
      // Drive geometry and currents are programmed into the TMC2209 at boot:
      // they are saved (with "save": true) but only used after a reboot.
      BoxConfig before = cfg;
      const char* bad = nullptr;
      if (!cfg.fromJson(req.as<JsonObjectConst>(), &bad)) return protoError(id, "bad_param", bad);
      reboot = cfg.motor_steps != before.motor_steps || cfg.microsteps != before.microsteps ||
               cfg.ratio != before.ratio || cfg.run_ma != before.run_ma || cfg.hold_pct != before.hold_pct;
      if (req["save"] | false) cfg.save();
      cfg.motor_steps = before.motor_steps;
      cfg.microsteps = before.microsteps;
      cfg.ratio = before.ratio;
      cfg.run_ma = before.run_ma;
      cfg.hold_pct = before.hold_pct;
      motorApplySpeed();
    }
    JsonDocument d = okDoc(id);
    cfg.toJson(d["config"].to<JsonObject>());
    if (reboot) d["reboot_required"] = true;
    return protoSend(d);
  }
  if (!strcmp(cmd, "reboot")) {
    lightsAllOff(nullptr);
    motorEnable(false);
    JsonDocument d = okDoc(id);
    protoSend(d);
    Serial.flush();
    delay(100);
    ESP.restart();
  }
  protoError(id, "unknown_cmd", cmd);
}
