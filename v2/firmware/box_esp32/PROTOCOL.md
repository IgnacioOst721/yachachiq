# Yachachiq analysis box: ESP32 serial protocol (v1)

Transport: USB serial (CP210x/CH340 on the DevKitC), **115200 baud 8N1**, one JSON
object per line, `\n` terminated, at most 512 bytes per line, UTF-8.
Implemented by `src/*.cpp` (firmware), `yq/box/device.py` (host driver) and
`yq/box/sim.py` (simulator, same protocol).

## Messages

Host -> ESP32 (request):

    {"id": 17, "cmd": "rotate_by", "deg": 10}

ESP32 -> host (reply, exactly one per request, same `id`):

    {"id": 17, "ok": true, "target_deg": 10.0, "eta_ms": 940}
    {"id": 18, "ok": false, "error": "interlock", "msg": "door open"}

ESP32 -> host (event, no `id`, may arrive at any time, also between the
bytes of two replies' lines, never inside a line):

    {"event": "door", "door": "front", "closed": false}

Every valid request (any `cmd`) also counts as a host heartbeat.
Most replies come immediately. `scale_tare`, `scale_cal` and `weigh` reply
only when the measurement ends (up to `timeout_ms`, default 8000 ms); the
host must use a longer timeout for them. Motion commands reply immediately
and later emit a `move_done` event carrying the request id in `ref`.

## Commands

| cmd | params | reply fields |
|---|---|---|
| `ping` | – | `pong`, `uptime_ms` |
| `info` | – | `fw`, `version`, `proto`, `board`, `tmc` {`ok`,`version`}, `hx711`, `steps_per_rev`, `channels`, `reset_reason` |
| `hb` | – | `uptime_ms` (heartbeat only) |
| `status` | – | see *Status* below |
| `motor` | `enable` bool | `enabled` |
| `rotate_to` | `deg` float (absolute turntable angle, unwrapped); optional `wrap` bool (go the short way to `deg` mod 360) | `target_deg`, `eta_ms` |
| `rotate_by` | `deg` float (relative, + = counter-clockwise seen from above) | `target_deg`, `eta_ms` |
| `speed` | optional `dps` (deg/s of the turntable), `accel` (deg/s²) | `dps`, `accel` (after clamping to the limits) |
| `stop` | optional `hard` bool (no deceleration) | `deg` |
| `zero` | – (only when not moving) | `deg` = 0 |
| `scale_tare` | optional `n` (samples, default 20), `timeout_ms` | `offset`, `sigma_raw`, `stable` |
| `scale_cal` | `grams` (known mass on the platter, after a tare), optional `n`, `timeout_ms` | `factor` (counts per gram, saved in NVS), `raw_mean` |
| `weigh` | optional `n` (window, default 20), `stable_g` (default 0.5), `timeout_ms` | `grams`, `sigma_g`, `samples` (grams, the window), `stable`, `tare_raw`, `tare_g`, `factor`, `raw_mean` |
| `light` | `ch`, `level` 0..1, optional `max_ms` | `ch`, `level`, `off_in_ms` |
| `all_off` | – | `cut` (channels that were on) |
| `estop` | – | all lights off, motor hard stop and disabled |
| `config_get` | – | `config` object (below) |
| `config_set` | any subset of the config keys; optional `save` bool (persist in NVS) | `config` (after clamping) |
| `reboot` | – | replies, then restarts after 100 ms |

### Channels (`ch`)

`rake1` … `rake8` (3 W white LEDs, raking light for RTI, PWM), `uv` (365 nm LED,
on/off only), `halogen` (MR16 12 V 35 W heat source, on/off only), `cob`
(uniform COB strip, PWM), `fan` (40 mm 12 V fan, PWM).

* `level` 0 turns the channel off. For `uv` and `halogen` any level > 0 means fully
  on (they are plain GPIO outputs so the door interrupt can cut them directly).
* PWM is 1 kHz, 10 bit. **Capture with level 1.0** (no PWM ripple, no banding).
* Only one `rakeN` can be on: turning one on turns the others off
  (event `light_off` reason `exclusive`).
* On-time: the channel switches itself off after `min(max_ms, max_on_ms[kind])`
  (event `light_off`, reason `max_on`). `max_ms` omitted = the kind limit.
* Cool-down: after being on for `t` ms a channel refuses to turn on again for
  `cool_factor[kind] * t` ms (error `cooldown`, field `wait_ms`).
* `uv` and `halogen` refuse to turn on unless **both** doors are closed (error
  `interlock`) and are cut by the door interrupt within microseconds of a door
  opening (the main loop re-checks every iteration as a second path).

### Status reply

    {"id":3,"ok":true,"uptime_ms":123456,"hb_age_ms":210,"hb_lost":false,
     "doors":{"front":true,"shutter":true},"doors_closed":true,
     "motor":{"enabled":true,"moving":false,"steps":4480,"deg":36.0,"target_deg":36.0,
              "dps":30.0,"accel":45.0,"tmc_ok":true},
     "lights":{"rake1":{"level":0.0,"on_ms":0,"left_ms":0,"cool_ms":0}, ...},
     "scale":{"busy":false,"tared":true,"calibrated":true,"factor":-412.3,"offset":81234,"hx711":true},
     "faults":[]}

`on_ms` = time on so far, `left_ms` = time until the automatic switch-off,
`cool_ms` = cool-down still remaining.

### Config keys (`config_get` / `config_set`)

| key | default | hard limits (firmware refuses to go beyond) |
|---|---|---|
| `max_on_ms` {`rake`,`uv`,`halogen`,`cob`,`fan`} | 20000, 30000, 45000, 900000, 0 (0 = unlimited, fan only) | rake ≤ 30000, uv ≤ 60000, halogen ≤ 60000, cob ≤ 1800000 |
| `cool_factor` {`rake`,`uv`,`halogen`,`cob`,`fan`} | 2.0, 2.0, 3.0, 0.5, 0 | rake ≥ 1, uv ≥ 1, halogen ≥ 2 |
| `hb_timeout_ms` | 3000 | 500 … 10000 |
| `dps` / `accel` | 30 / 45 | ≤ `max_dps` / ≤ `max_accel` |
| `max_dps` / `max_accel` | 90 / 180 | ≤ 120 / ≤ 360 |
| `motor_steps`, `microsteps`, `ratio` | 200, 16, 14.0 | microsteps ∈ {8,16,32,64}; applied on next boot if saved |
| `run_ma` / `hold_pct` | 1000 / 30 | 300 … 1400 mA / 0 … 60 % |
| `door_stops_motor` | true | when true, opening a door stops the platter and `rotate_*` is refused (`interlock`) while a door is open |
| `allow_no_uart` | false | when false, motion is refused if the TMC2209 does not answer (microsteps would be wrong) |

`steps_per_rev = motor_steps * microsteps * ratio` (default 200·16·14 = 44 800).

## Events

| event | fields | when |
|---|---|---|
| `boot` | `fw`, `version`, `reset_reason` | once after reset (all outputs already off) |
| `door` | `door` (`front`/`shutter`), `closed` | reed changed (open = immediately, closed = after 50 ms stable LOW). After boot both doors start as "open", so a closed door is announced once ~50 ms after `boot` |
| `interlock` | `cut` [channels] | a door opened while `uv`/`halogen` were on |
| `light_off` | `ch`, `reason` (`max_on`, `interlock`, `exclusive`, `watchdog`) | automatic switch-off |
| `move_done` | `ref` (request id), `deg`, `steps`, `stopped` (true if cut short) | motion finished |
| `watchdog` | `reason` = `heartbeat` | no request for `hb_timeout_ms`: all lights and fan off, motor stopped and disabled |
| `fault` | `what`, `msg` | e.g. `tmc_uart`, `hx711` |

## Errors (`error` field)

`bad_json`, `line_too_long`, `unknown_cmd`, `bad_param`, `interlock`, `cooldown`,
`moving` (command needs the motor stopped), `tmc_uart`, `scale_busy`,
`not_tared`, `not_calibrated`, `scale_timeout`, `hx711_missing`.

## Safety (firmware, independent of the host)

1. Every output is driven LOW in a C++ global constructor (before `setup()`) and
   again in `initVariant()`; during reset, bootloader and flashing the pins are
   high-impedance, so **hardware pull-downs (10 kΩ) on the UV and HALOGEN MOSFET
   inputs and a 10 kΩ pull-up on the TMC2209 EN pin are mandatory** (see
   `docs/box.md`).
2. UV/halogen interlock: reed inputs have external 10 kΩ pull-ups; closed door =
   magnet = LOW. Open door **or broken wire** reads HIGH = open. A GPIO
   interrupt cuts UV and halogen by writing the GPIO clear register directly.
3. Per-channel maximum on-time and cool-down (table above).
4. Host heartbeat: armed by the first request after boot; no request for 3 s =
   everything off + motor disabled (`watchdog` event). The next request re-arms it.
5. Hardware task watchdog (ESP-IDF TWDT, 5 s, panic = reset) on the Arduino loop.
