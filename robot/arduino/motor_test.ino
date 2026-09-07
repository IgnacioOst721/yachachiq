// Yachachiq pen plotter - standalone stepper test (NO GRBL)
//
// Flash this to test the motors directly, with the CNC Shield on the Arduino.
// It enables all three drivers and spins each axis one at a time, forward then
// back, forever. Use it to confirm a motor turns and to debug coil wiring,
// driver Vref, and the dead Z axis without GRBL in the way.
//
// CNC Shield V3 pin map (same as GRBL):
//   X: step D2, dir D5    Y: step D3, dir D6    Z: step D4, dir D7
//   Enable: D8 (active LOW)

const int EN = 8;                 // driver enable, LOW = on
const int X_STEP = 2, X_DIR = 5;
const int Y_STEP = 3, Y_DIR = 6;
const int Z_STEP = 4, Z_DIR = 7;

const int STEPS = 3200;           // pulses per test (~1 rev at 1/16 microstepping)
const int PULSE_US = 400;         // half-period of a step pulse; larger = slower = more torque

void setup() {
  pinMode(EN, OUTPUT);
  digitalWrite(EN, LOW);          // enable all drivers
  const int outs[] = {X_STEP, X_DIR, Y_STEP, Y_DIR, Z_STEP, Z_DIR};
  for (unsigned i = 0; i < sizeof(outs) / sizeof(outs[0]); i++) {
    pinMode(outs[i], OUTPUT);
  }
}

void spin(int stepPin, int dirPin, bool dir) {
  digitalWrite(dirPin, dir);
  for (int i = 0; i < STEPS; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(PULSE_US);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(PULSE_US);
  }
}

void loop() {
  spin(X_STEP, X_DIR, HIGH);  delay(500);
  spin(Y_STEP, Y_DIR, HIGH);  delay(500);
  spin(Z_STEP, Z_DIR, HIGH);  delay(500);
  spin(X_STEP, X_DIR, LOW);   delay(500);
  spin(Y_STEP, Y_DIR, LOW);   delay(500);
  spin(Z_STEP, Z_DIR, LOW);   delay(2000);
}
