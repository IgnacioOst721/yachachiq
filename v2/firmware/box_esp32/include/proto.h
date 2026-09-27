// JSON-lines transport over the USB serial port (see PROTOCOL.md).
// All output happens from the Arduino loop task, so no locking is needed.
#pragma once
#include <ArduinoJson.h>

// Send a finished document as one line.
void protoSend(JsonDocument& doc);
uint32_t protoDropped();  // lines dropped because the TX buffer was full
// {"id":id,"ok":false,"error":code,"msg":msg}
void protoError(long id, const char* code, const char* msg = nullptr);
// {"event":name, ...fields}; fill `doc` after calling protoEventDoc().
JsonDocument protoEventDoc(const char* name);
// Read available bytes; returns true and fills `line` when a full line arrived.
bool protoPoll(char* line, size_t cap, bool* tooLong);
