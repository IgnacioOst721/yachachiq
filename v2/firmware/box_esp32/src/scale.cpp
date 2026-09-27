// Load cell. bogde/HX711 0.7.5: read() waits for DOUT LOW and then shifts 24
// bits inside a critical section (~60 us); we only call it when is_ready() is
// already true, so the loop never blocks. The SparkFun HX711 board runs at
// 10 samples/s (RATE pin LOW).
#include "scale.h"

#include <Arduino.h>
#include <HX711.h>
#include <Preferences.h>
#include <math.h>

#include "lights.h"
#include "motor.h"
#include "pins.h"
#include "proto.h"

#define MAX_N 64
#define DEFAULT_FACTOR 420.0f  // ~counts per gram for a 5 kg / 1 mV/V cell at gain 128

static HX711 hx;
static long s_offset = 0;
static float s_factor = NAN;
static bool s_tared = false;
static bool s_present = false;
static uint32_t s_lastReady = 0;

static int s_op = 0;
static long s_id = -1;
static int s_n = 20;
static float s_stableG = 0.5f;
static float s_grams = 0;
static uint32_t s_t0 = 0, s_timeout = 8000;
static long s_buf[MAX_N];
static int s_count = 0;
static bool s_skipFirst = true;

void scaleBegin() {
  hx.begin(PIN_HX_DOUT, PIN_HX_SCK, 128);
  Preferences p;
  if (p.begin("yqscale", true)) {
    s_offset = p.getLong("offset", 0);
    s_tared = p.getBool("tared", false);
    s_factor = p.getFloat("factor", NAN);
    p.end();
  }
}

static void save() {
  Preferences p;
  if (!p.begin("yqscale", false)) return;
  p.putLong("offset", s_offset);
  p.putBool("tared", s_tared);
  if (!isnan(s_factor)) p.putFloat("factor", s_factor);
  p.end();
}

bool scaleBusy() { return s_op != 0; }
bool scalePresent() { return s_present; }
static bool calibrated() { return !isnan(s_factor) && fabsf(s_factor) > 1.0f; }
static float factorOrDefault() { return calibrated() ? s_factor : DEFAULT_FACTOR; }

const char* scaleStart(int op, long id, int n, float stableG, uint32_t timeoutMs, float grams, uint32_t now) {
  if (s_op) return "scale_busy";
  if (!s_present) return "hx711_missing";
  if (motorMoving()) return "moving";
  if (n < 3 || n > MAX_N || !(stableG > 0) || timeoutMs < 500 || timeoutMs > 60000) return "bad_param";
  if (op == SCALE_CAL && (!(grams > 0) || !s_tared)) return !s_tared ? "not_tared" : "bad_param";
  if (op == SCALE_WEIGH && !s_tared) return "not_tared";
  if (op == SCALE_WEIGH && !calibrated()) return "not_calibrated";
  s_op = op;
  s_id = id;
  s_n = n;
  s_stableG = stableG;
  s_grams = grams;
  s_t0 = now;
  s_timeout = timeoutMs;
  s_count = 0;
  s_skipFirst = true;
  lightsHoldFan(true);  // the fan shakes the cell
  return nullptr;
}

void scaleAbort() {
  if (!s_op) return;
  protoError(s_id, "scale_timeout", "aborted");
  s_op = 0;
  lightsHoldFan(false);
}

// mean / sigma / half-window drift of the last n samples, in grams
static void windowStats(int n, double* meanRaw, double* sigmaG, double* driftG) {
  double f = fabs(factorOrDefault());
  double sum = 0, sq = 0, h1 = 0, h2 = 0;
  int start = s_count - n, half = n / 2;
  for (int i = 0; i < n; i++) {
    double v = s_buf[start + i];
    sum += v;
    if (i < half) h1 += v;
    else h2 += v;
  }
  double m = sum / n;
  for (int i = 0; i < n; i++) {
    double d = s_buf[start + i] - m;
    sq += d * d;
  }
  *meanRaw = m;
  *sigmaG = sqrt(sq / (n > 1 ? n - 1 : 1)) / f;
  *driftG = fabs(h1 / half - h2 / (n - half)) / f;
}

static void finish(bool stable) {
  int n = s_count < s_n ? s_count : s_n;
  JsonDocument d;
  d["id"] = s_id;
  // the samples are in: give the fan back BEFORE replying, so the host never sees it still held
  lightsHoldFan(false);
  if (n < 3) {
    protoError(s_id, "scale_timeout", "too few samples");
    s_op = 0;
    return;
  }
  double m, sig, drift;
  windowStats(n, &m, &sig, &drift);
  if (s_op == SCALE_TARE) {
    if (!stable) { protoError(s_id, "scale_timeout", "not stable"); goto done; }
    s_offset = lround(m);
    s_tared = true;
    save();
    d["ok"] = true;
    d["offset"] = s_offset;
    d["sigma_raw"] = sig * fabs(factorOrDefault());
    d["stable"] = true;
    protoSend(d);
  } else if (s_op == SCALE_CAL) {
    double f = (m - s_offset) / s_grams;
    if (!stable) { protoError(s_id, "scale_timeout", "not stable"); goto done; }
    if (fabs(f) < 1.0) { protoError(s_id, "bad_param", "no load change: is the weight on the platter?"); goto done; }
    s_factor = (float)f;
    save();
    d["ok"] = true;
    d["factor"] = s_factor;
    d["raw_mean"] = m;
    protoSend(d);
  } else {
    d["ok"] = true;
    d["grams"] = (m - s_offset) / s_factor;
    d["sigma_g"] = sig;
    d["stable"] = stable;
    JsonArray a = d["samples"].to<JsonArray>();
    for (int i = s_count - n; i < s_count; i++) a.add(roundf((s_buf[i] - s_offset) / s_factor * 100.0f) / 100.0f);
    d["tare_raw"] = s_offset;
    d["tare_g"] = s_offset / s_factor;
    d["factor"] = s_factor;
    d["raw_mean"] = m;
    protoSend(d);
  }
done:
  s_op = 0;
  lightsHoldFan(false);
}

void scaleLoop(uint32_t now) {
  if (hx.is_ready()) {
    long raw = hx.read();
    s_present = true;
    s_lastReady = now;
    if (s_op) {
      if (s_skipFirst) {
        s_skipFirst = false;  // converted before the request (maybe with the fan on)
      } else {
        if (s_count == MAX_N) {  // keep the newest MAX_N
          memmove(s_buf, s_buf + 1, sizeof(long) * (MAX_N - 1));
          s_count--;
        }
        s_buf[s_count++] = raw;
        if (s_count >= s_n) {
          double m, sig, drift;
          windowStats(s_n, &m, &sig, &drift);
          if (sig <= s_stableG && drift <= s_stableG) finish(true);
        }
      }
    }
  } else if (s_present && now - s_lastReady > 1000) {
    s_present = false;  // DOUT stuck: board unplugged or not powered
    JsonDocument d = protoEventDoc("fault");
    d["what"] = "hx711";
    d["msg"] = "HX711 stopped answering";
    protoSend(d);
  }
  if (s_op && now - s_t0 > s_timeout) finish(false);
}

void scaleStatus(JsonObject o) {
  o["busy"] = s_op != 0;
  o["tared"] = s_tared;
  o["calibrated"] = calibrated();
  if (calibrated()) o["factor"] = s_factor;
  else o["factor"] = nullptr;
  o["offset"] = s_offset;
  o["hx711"] = s_present;
}
