# Arduino firmware — GRBL

The Arduino UNO on Yachachiq runs **GRBL**, the open-source CNC firmware that
receives G-code over USB serial and drives the stepper motors and pen servo.

## How to flash it

1. Download the GRBL source: https://github.com/gnea/grbl
   (early prototypes used https://github.com/TGit-Tech/GRBL-28byj-48 for the
   28BYJ-48 steppers; the final machine uses standard GRBL with NEMA 17
   steppers on DRV8825 drivers and a CNC shield).
2. Copy the `grbl` folder into your Arduino `libraries` folder
   (Arduino IDE → Sketch → Include Library → Add .ZIP Library also works).
3. Open [`grblUpload/grblUpload.ino`](grblUpload/grblUpload.ino) in the
   Arduino IDE.
4. Tools → Board → **Arduino Uno**, select your serial port.
5. Click **Upload**.

The sketch itself is just `#include <grbl.h>` — all the real firmware lives in
the library. Machine settings (steps/mm, max travel, servo pen angles) are
configured over serial with `$` commands after flashing.

GRBL is Copyright (c) Sungeun K. Jeon, MIT license — it is not our code; we
include only the standard upload sketch here for completeness.
