#include "proto.h"

#include <Arduino.h>

static uint32_t s_dropped = 0;

// Never blocks the safety loop: if the host is not reading (TX buffer full) the
// line is dropped and counted (status "tx_dropped"). The DevKitC's USB-UART
// bridge normally drains the UART even with the port closed.
void protoSend(JsonDocument& doc) {
  size_t need = measureJson(doc) + 1;
  if ((size_t)Serial.availableForWrite() < need) {
    s_dropped++;
    return;
  }
  serializeJson(doc, Serial);
  Serial.write('\n');
}

uint32_t protoDropped() { return s_dropped; }

void protoError(long id, const char* code, const char* msg) {
  JsonDocument d;
  d["id"] = id;
  d["ok"] = false;
  d["error"] = code;
  if (msg) d["msg"] = msg;
  protoSend(d);
}

JsonDocument protoEventDoc(const char* name) {
  JsonDocument d;
  d["event"] = name;
  return d;
}

static size_t s_len = 0;
static bool s_overflow = false;
static char s_buf[520];

bool protoPoll(char* line, size_t cap, bool* tooLong) {
  *tooLong = false;
  while (Serial.available() > 0) {
    int c = Serial.read();
    if (c < 0) break;
    if (c == '\r') continue;
    if (c == '\n') {
      if (s_overflow) {
        s_overflow = false;
        s_len = 0;
        *tooLong = true;
        return true;
      }
      if (s_len == 0) continue;  // empty line
      size_t n = s_len < cap - 1 ? s_len : cap - 1;
      memcpy(line, s_buf, n);
      line[n] = 0;
      s_len = 0;
      return true;
    }
    if (s_len >= 512) {
      s_overflow = true;
      continue;
    }
    s_buf[s_len++] = (char)c;
  }
  return false;
}
