// HX711 + TAL220B 5 kg load cell, sampled without blocking the loop.
// Stability: the last `n` samples have sigma <= stable_g AND the means of the
// two halves of that window differ by <= stable_g (no drift). Same rule in
// v2/yq/box/scalemath.py (simulator + tests).
#pragma once
#include <ArduinoJson.h>
#include <stdint.h>

#define SCALE_TARE 1
#define SCALE_CAL 2
#define SCALE_WEIGH 3

void scaleBegin();
void scaleLoop(uint32_t now);
bool scaleBusy();
bool scalePresent();
// Starts an asynchronous measurement; the reply is sent when it ends.
// Returns nullptr when started, else an error code (reply not sent).
const char* scaleStart(int op, long id, int n, float stableG, uint32_t timeoutMs, float grams, uint32_t now);
void scaleAbort();
void scaleStatus(JsonObject o);
