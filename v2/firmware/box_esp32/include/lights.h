// Light / heater / fan channels with on-time limits, cool-downs and the
// UV + halogen door interlock (see PROTOCOL.md).
#pragma once
#include <ArduinoJson.h>
#include <stdint.h>

#define NUM_CHANNELS 12
#define CH_UV 8
#define CH_HALOGEN 9
#define CH_COB 10
#define CH_FAN 11

void lightsBegin();
int lightsFind(const char* name);                  // -1 if unknown
const char* lightsName(int ch);
// Returns nullptr on success, else an error code; fills offInMs / waitMs.
const char* lightsSet(int ch, float level, uint32_t maxMs, uint32_t now, uint32_t* offInMs, uint32_t* waitMs);
void lightsLoop(uint32_t now);                     // automatic switch-off at max on-time
// Turns everything off. reason: nullptr for the all_off command (no events),
// else a light_off event per channel that was on. Fills `cut` if given.
void lightsAllOff(const char* reason, JsonArray* cut = nullptr);
void lightsInterlock(bool doorOpen, uint32_t isrMask, uint32_t now);
void lightsHoldFan(bool hold);                     // fan forced off while weighing
bool lightsAnyOn();
void lightsStatus(JsonObject o, uint32_t now);
