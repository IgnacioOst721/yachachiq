// Turntable: NEMA17 + TMC2209 (UART config, STEP/DIR pulses by FastAccelStepper).
#pragma once
#include <ArduinoJson.h>
#include <stdint.h>

void motorBegin();
bool motorTmcOk();
uint8_t motorTmcVersion();
void motorEnable(bool on);
bool motorEnabled();
bool motorMoving();
// Return nullptr on success, else an error code.
const char* motorRotateTo(double deg, bool wrap, long ref, uint32_t* etaMs, double* targetDeg);
const char* motorRotateBy(double deg, long ref, uint32_t* etaMs, double* targetDeg);
const char* motorZero();
void motorApplySpeed();          // after cfg.dps / cfg.accel changed
void motorStop(bool hard);
double motorDeg();
long motorSteps();
void motorLoop(uint32_t now);    // move_done events
void motorStatus(JsonObject o);
