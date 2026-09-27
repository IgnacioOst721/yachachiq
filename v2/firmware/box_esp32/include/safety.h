// Door interlocks (reed switches) and the host heartbeat watchdog.
#pragma once
#include <stdint.h>

#define DOOR_FRONT 0
#define DOOR_SHUTTER 1
static const char* const DOOR_NAMES[2] = {"front", "shutter"};

void safetyBegin();                 // reed pins + interrupt
void safetyLoop(uint32_t now);      // debounce, door events, heartbeat
bool doorClosed(int door);          // debounced: LOW for >= 50 ms
bool doorsClosed();                 // both doors closed
bool anyDoorRawOpen();              // instantaneous: any reed HIGH (open or broken wire)
uint32_t takeIsrCutMask();          // bits 0 (UV) / 1 (HALOGEN) cut by the ISR since last call

void hbTouch(uint32_t now);         // every valid request
bool hbLost();
bool hbArmed();
uint32_t hbAgeMs(uint32_t now);
