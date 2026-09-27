// Turntable motion. 200 steps x 16 microsteps x 14 (280T/20T) = 44800 steps/rev.
// TMC2209 register access verified against TMCStepper 0.7.3 sources:
// begin() = pdn_disable + mstep_reg_select; rms_current(mA, hold_multiplier);
// en_spreadCycle(false) = stealthChop; test_connection() == 0 means OK;
// version() == 0x21 for a TMC2209.
#include "motor.h"

#include <Arduino.h>
#include <FastAccelStepper.h>
#include <TMCStepper.h>
#include <math.h>

#include "boxconfig.h"
#include "pins.h"
#include "proto.h"

#define R_SENSE 0.11f      // BIGTREETECH TMC2209 V1.3 sense resistors (110 mOhm)
#define TMC_ADDRESS 0      // MS1 = MS2 = GND (internal pull-downs)

static TMC2209Stepper driver(&Serial2, R_SENSE, TMC_ADDRESS);
static FastAccelStepperEngine engine;
static FastAccelStepper* stepper = nullptr;
static bool s_tmcOk = false;
static uint8_t s_tmcVersion = 0;
static bool s_enabled = false;
static bool s_moving = false;
static bool s_stopped = false;
static long s_ref = -1;
static double s_targetDeg = 0;

static double stepsPerDeg() { return cfg.stepsPerRev() / 360.0; }

void motorBegin() {
  Serial2.begin(115200, SERIAL_8N1, PIN_TMC_RX, PIN_TMC_TX);
  driver.begin();
  driver.toff(4);
  driver.blank_time(24);
  driver.I_scale_analog(false);  // current from UART, not from the VREF pot
  driver.rms_current(cfg.run_ma, cfg.hold_pct / 100.0f);
  driver.microsteps(cfg.microsteps);
  driver.en_spreadCycle(false);  // stealthChop: quiet, low vibration
  driver.pwm_autoscale(true);
  driver.pwm_autograd(true);
  driver.iholddelay(8);
  driver.TPOWERDOWN(20);         // ~0.4 s after the last step -> hold current
  uint8_t conn = driver.test_connection();
  s_tmcVersion = driver.version();
  s_tmcOk = conn == 0 && s_tmcVersion == 0x21 && driver.microsteps() == cfg.microsteps;
  if (!s_tmcOk) {
    JsonDocument d = protoEventDoc("fault");
    d["what"] = "tmc_uart";
    d["msg"] = "TMC2209 not answering on UART (check PDN_UART wiring, 1k resistor, VM power)";
    protoSend(d);
  }

#ifdef YQ_NO_STEPPER
  // Emulator build (env:qemu): QEMU does not emulate the MCPWM/PCNT peripherals
  // FastAccelStepper uses, so motion commands answer "tmc_uart" there.
  return;
#endif
  engine.init();
  stepper = engine.stepperConnectToPin(PIN_STEP);
  if (!stepper) {
    JsonDocument d = protoEventDoc("fault");
    d["what"] = "stepper";
    d["msg"] = "FastAccelStepper could not attach to the STEP pin";
    protoSend(d);
    return;
  }
  stepper->setDirectionPin(PIN_DIR);
  stepper->setEnablePin(PIN_EN, true);  // ENN is active LOW
  stepper->setAutoEnable(false);
  stepper->disableOutputs();
  motorApplySpeed();
}

bool motorTmcOk() { return s_tmcOk; }
uint8_t motorTmcVersion() { return s_tmcVersion; }

void motorEnable(bool on) {
  if (!stepper) return;
  if (!on && stepper->isRunning()) {
    stepper->forceStop();
    s_stopped = true;
  }
  if (on) stepper->enableOutputs();
  else stepper->disableOutputs();
  s_enabled = on;
}

bool motorEnabled() { return s_enabled; }
bool motorMoving() { return stepper && stepper->isRunning(); }

void motorApplySpeed() {
  if (!stepper) return;
  uint32_t hz = (uint32_t)(cfg.dps * stepsPerDeg() + 0.5);
  int32_t acc = (int32_t)(cfg.accel * stepsPerDeg() + 0.5);
  stepper->setSpeedInHz(hz < 1 ? 1 : hz);
  stepper->setAcceleration(acc < 1 ? 1 : acc);
}

static uint32_t etaFor(long steps) {
  double d = fabs((double)steps);
  double v = cfg.dps * stepsPerDeg();
  double a = cfg.accel * stepsPerDeg();
  double t = (d < v * v / a) ? 2.0 * sqrt(d / a) : d / v + v / a;
  return (uint32_t)(t * 1000.0 + 0.5);
}

static const char* startMove(double deg, long ref, uint32_t* etaMs, double* targetDeg) {
  if (!stepper) return "tmc_uart";
  if (!s_tmcOk && !cfg.allow_no_uart) return "tmc_uart";
  if (!isfinite(deg) || fabs(deg) > 36000.0) return "bad_param";
  if (!s_enabled) {
    stepper->enableOutputs();
    s_enabled = true;
    delay(5);  // TMC2209 needs a moment after ENN before the first step
  }
  long target = lround(deg * stepsPerDeg());
  long delta = target - stepper->getCurrentPosition();
  MoveResultCode rc = stepper->moveTo(target);
  if (!moveIsOk(rc)) return "bad_param";
  s_targetDeg = deg;
  s_moving = true;
  s_stopped = false;
  s_ref = ref;
  *etaMs = etaFor(delta);
  *targetDeg = deg;
  return nullptr;
}

const char* motorRotateTo(double deg, bool wrap, long ref, uint32_t* etaMs, double* targetDeg) {
  if (wrap) {
    double cur = motorDeg();
    double d = fmod(deg - cur, 360.0);
    if (d > 180.0) d -= 360.0;
    if (d < -180.0) d += 360.0;
    deg = cur + d;
  }
  return startMove(deg, ref, etaMs, targetDeg);
}

const char* motorRotateBy(double deg, long ref, uint32_t* etaMs, double* targetDeg) {
  double base = motorMoving() ? s_targetDeg : motorDeg();
  return startMove(base + deg, ref, etaMs, targetDeg);
}

const char* motorZero() {
  if (!stepper) return "tmc_uart";
  if (stepper->isRunning()) return "moving";
  stepper->setCurrentPosition(0);
  s_targetDeg = 0;
  return nullptr;
}

void motorStop(bool hard) {
  if (!stepper || !stepper->isRunning()) return;
  if (hard) stepper->forceStop();
  else stepper->stopMove();
  s_stopped = true;
}

long motorSteps() { return stepper ? stepper->getCurrentPosition() : 0; }
double motorDeg() { return motorSteps() / stepsPerDeg(); }

void motorLoop(uint32_t now) {
  (void)now;
  if (s_moving && stepper && !stepper->isRunning()) {
    s_moving = false;
    if (s_stopped) s_targetDeg = motorDeg();
    JsonDocument d = protoEventDoc("move_done");
    d["ref"] = s_ref;
    d["deg"] = motorDeg();
    d["steps"] = motorSteps();
    d["stopped"] = s_stopped;
    protoSend(d);
  }
}

void motorStatus(JsonObject o) {
  o["enabled"] = s_enabled;
  o["moving"] = motorMoving();
  o["steps"] = motorSteps();
  o["deg"] = motorDeg();
  o["target_deg"] = motorMoving() ? s_targetDeg : motorDeg();
  o["dps"] = cfg.dps;
  o["accel"] = cfg.accel;
  o["tmc_ok"] = s_tmcOk;
  o["steps_per_rev"] = cfg.stepsPerRev();
}
