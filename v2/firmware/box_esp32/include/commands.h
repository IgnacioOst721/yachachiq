// Parses one request line and sends its reply (scale replies are deferred).
#pragma once
#include <stdint.h>

void commandsHandle(const char* line, uint32_t now);
const char* commandsResetReason();
