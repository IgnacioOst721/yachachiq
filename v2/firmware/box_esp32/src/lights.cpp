// Light channels. RAKE1..8, COB and FAN use LEDC PWM (1 kHz, 10 bit);
// UV and HALOGEN are plain GPIO outputs so the door ISR can clear them
// directly in the GPIO register.
#include "lights.h"

#include <Arduino.h>

#include "boxconfig.h"
#include "pins.h"
#include "proto.h"
#include "safety.h"

#define PWM_FREQ 1000
#define PWM_BITS 10
#define PWM_MAX ((1 << PWM_BITS) - 1)

struct Channel {
  const char* name;
  uint8_t pin;
  LightKind kind;
  int8_t ledc;         // LEDC channel, -1 = plain GPIO
  float level;         // 0 = off
  uint32_t onSince;
  uint32_t limitMs;    // 0 = unlimited
  uint32_t coolUntil;  // millis() before which it may not turn on
  bool cooling;
};

static Channel CH[NUM_CHANNELS] = {
    {"rake1", 4, K_RAKE, 0}, {"rake2", 13, K_RAKE, 1}, {"rake3", 14, K_RAKE, 2},
    {"rake4", 18, K_RAKE, 3}, {"rake5", 19, K_RAKE, 4}, {"rake6", 21, K_RAKE, 5},
    {"rake7", 22, K_RAKE, 6}, {"rake8", 23, K_RAKE, 7},
    {"uv", PIN_UV, K_UV, -1}, {"halogen", PIN_HALOGEN, K_HALOGEN, -1},
    {"cob", PIN_COB, K_COB, 8}, {"fan", PIN_FAN, K_FAN, 9},
};
static bool s_fanHeld = false;

static void writeOut(Channel& c, float level) {
  if (c.ledc < 0) {
    digitalWrite(c.pin, level > 0 ? HIGH : LOW);
  } else {
    uint32_t duty = (uint32_t)(level * PWM_MAX + 0.5f);
    if (duty > PWM_MAX) duty = PWM_MAX;
    ledcWrite(c.ledc, duty);
  }
}

void lightsBegin() {
  for (int i = 0; i < NUM_CHANNELS; i++) {
    Channel& c = CH[i];
    digitalWrite(c.pin, LOW);
    pinMode(c.pin, OUTPUT);
    if (c.ledc >= 0) {
      ledcSetup(c.ledc, PWM_FREQ, PWM_BITS);
      ledcWrite(c.ledc, 0);
      ledcAttachPin(c.pin, c.ledc);
    }
    c.level = 0;
    c.cooling = false;
  }
  static_assert(CH_UV == 8 && CH_HALOGEN == 9 && CH_COB == 10 && CH_FAN == 11, "channel order");
}

int lightsFind(const char* name) {
  if (!name) return -1;
  for (int i = 0; i < NUM_CHANNELS; i++)
    if (strcmp(CH[i].name, name) == 0) return i;
  return -1;
}

const char* lightsName(int ch) { return CH[ch].name; }

static void turnOff(int i, uint32_t now, const char* reason, JsonArray* cut) {
  Channel& c = CH[i];
  if (c.level <= 0) return;
  writeOut(c, 0);
  uint32_t onFor = now - c.onSince;
  c.level = 0;
  float f = cfg.cool_factor[c.kind];
  if (f > 0) {
    c.coolUntil = now + (uint32_t)(f * onFor);
    c.cooling = true;
  }
  if (cut) cut->add(c.name);
  if (reason) {
    JsonDocument d = protoEventDoc("light_off");
    d["ch"] = c.name;
    d["reason"] = reason;
    d["on_ms"] = onFor;
    protoSend(d);
  }
}

const char* lightsSet(int i, float level, uint32_t maxMs, uint32_t now, uint32_t* offInMs, uint32_t* waitMs) {
  *offInMs = 0;
  *waitMs = 0;
  if (i < 0 || i >= NUM_CHANNELS) return "bad_param";
  if (!(level >= 0.0f && level <= 1.0f)) return "bad_param";
  Channel& c = CH[i];
  if (level == 0) {
    turnOff(i, now, nullptr, nullptr);
    return nullptr;
  }
  if (c.kind == K_UV || c.kind == K_HALOGEN) {
    if (!doorsClosed() || anyDoorRawOpen()) return "interlock";
    level = 1.0f;  // on/off only
  }
  if (c.level <= 0) {  // turning on
    if (c.cooling && (int32_t)(now - c.coolUntil) < 0) {
      *waitMs = c.coolUntil - now;
      return "cooldown";
    }
    c.cooling = false;
    if (c.kind == K_RAKE)
      for (int j = 0; j < NUM_CHANNELS; j++)
        if (j != i && CH[j].kind == K_RAKE) turnOff(j, now, "exclusive", nullptr);
    uint32_t kindMax = cfg.max_on_ms[c.kind];
    uint32_t lim = kindMax;
    if (maxMs > 0 && (lim == 0 || maxMs < lim)) lim = maxMs;
    c.onSince = now;
    c.limitMs = lim;
  } else if (maxMs > 0) {  // already on: may only shorten the limit
    uint32_t elapsed = now - c.onSince;
    if (c.limitMs == 0 || elapsed + maxMs < c.limitMs) c.limitMs = elapsed + maxMs;
  }
  c.level = level;
  if (!(i == CH_FAN && s_fanHeld)) writeOut(c, level);
  if (c.limitMs > 0) *offInMs = c.limitMs - (now - c.onSince);
  return nullptr;
}

void lightsLoop(uint32_t now) {
  for (int i = 0; i < NUM_CHANNELS; i++) {
    Channel& c = CH[i];
    if (c.level > 0 && c.limitMs > 0 && now - c.onSince >= c.limitMs) turnOff(i, now, "max_on", nullptr);
  }
}

void lightsAllOff(const char* reason, JsonArray* cut) {
  uint32_t now = millis();
  for (int i = 0; i < NUM_CHANNELS; i++) turnOff(i, now, reason, cut);
}

void lightsInterlock(bool doorOpen, uint32_t isrMask, uint32_t now) {
  bool uvOn = CH[CH_UV].level > 0, halOn = CH[CH_HALOGEN].level > 0;
  if (!isrMask && !(doorOpen && (uvOn || halOn))) {  // fast path, runs every loop
    if (doorOpen) {
      digitalWrite(PIN_UV, LOW);
      digitalWrite(PIN_HALOGEN, LOW);
    }
    return;
  }
  bool any = false;
  JsonDocument d = protoEventDoc("interlock");
  JsonArray cut = d["cut"].to<JsonArray>();
  const int chans[2] = {CH_UV, CH_HALOGEN};
  for (int k = 0; k < 2; k++) {
    Channel& c = CH[chans[k]];
    bool isrCut = isrMask & (1u << k);
    if (c.level > 0 && (doorOpen || isrCut)) {
      turnOff(chans[k], now, "interlock", &cut);
      any = true;
    } else if (doorOpen) {
      digitalWrite(c.pin, LOW);  // belt and braces: output must be LOW while open
    }
  }
  if (any) protoSend(d);
}

void lightsHoldFan(bool hold) {
  s_fanHeld = hold;
  writeOut(CH[CH_FAN], hold ? 0 : CH[CH_FAN].level);
}

bool lightsAnyOn() {
  for (int i = 0; i < NUM_CHANNELS; i++)
    if (CH[i].level > 0) return true;
  return false;
}

void lightsStatus(JsonObject o, uint32_t now) {
  for (int i = 0; i < NUM_CHANNELS; i++) {
    Channel& c = CH[i];
    JsonObject j = o[c.name].to<JsonObject>();
    j["level"] = c.level;
    uint32_t onMs = c.level > 0 ? now - c.onSince : 0;
    j["on_ms"] = onMs;
    j["left_ms"] = (c.level > 0 && c.limitMs > 0 && c.limitMs > onMs) ? c.limitMs - onMs : 0;
    j["cool_ms"] = (c.cooling && (int32_t)(now - c.coolUntil) < 0) ? c.coolUntil - now : 0;
  }
}
