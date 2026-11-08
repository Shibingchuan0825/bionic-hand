#!/usr/bin/env bash
# Compile-check the bionic_hand sketch with the real AVR toolchain.
#
# DRIVER_MODE 0 (Servo library) is compiled against the real Arduino Servo
# library.  DRIVER_MODE 1 (Adafruit_PWMServoDriver) is compiled against a
# minimal declaration stub, because that library is not installed in this
# environment; the stub only checks that our own code type-checks.
set -u

ARDUINO15="$HOME/AppData/Local/Arduino15"
AVR_GCC="$ARDUINO15/packages/arduino/tools/avr-gcc/7.3.0-atmel3.6.1-arduino7/bin"
CORE="$ARDUINO15/packages/arduino/hardware/avr/1.8.6/cores/arduino"
VARIANT="$ARDUINO15/packages/arduino/hardware/avr/1.8.6/variants/standard"
CORE_LIBS="$ARDUINO15/packages/arduino/hardware/avr/1.8.6/libraries"
SERVO_LIB="$ARDUINO15/libraries/Servo/src"

SKETCH="${1:-/d/hand/arduino/bionic_hand/bionic_hand.ino}"
MODE="${2:-0}"
OUTDIR="${3:-/tmp/bh_build}"

mkdir -p "$OUTDIR"
WORK="$OUTDIR/sketch_$MODE.cpp"
# The sketch ships with DRIVER_MODE 1; rewrite that single line so the other
# branch can be compiled too, exactly as an editor would.
sed "s/^#define DRIVER_MODE [0-9]\+/#define DRIVER_MODE $MODE/" "$SKETCH" > "$WORK"

INCLUDES="-I$CORE -I$VARIANT -I$CORE_LIBS/Wire/src -I$CORE_LIBS/SPI/src -I$SERVO_LIB -I$SERVO_LIB/avr"
STUB=""
if [ "$MODE" = "1" ]; then
  STUB="$OUTDIR/stub"
  mkdir -p "$STUB"
  cat > "$STUB/Adafruit_PWMServoDriver.h" <<'EOF'
// Compile-check stub.  Declares the subset of the Adafruit PCA9685 API used by
// this sketch so our own code can be type-checked without the real library.
#ifndef ADAFRUIT_PWMSERVODRIVER_STUB_H
#define ADAFRUIT_PWMSERVODRIVER_STUB_H
#include <stdint.h>
class Adafruit_PWMServoDriver {
 public:
  explicit Adafruit_PWMServoDriver(uint8_t addr = 0x40);
  void begin();
  void setPWMFreq(float freq);
  void setPWM(uint8_t num, uint16_t on, uint16_t off);
 private:
  uint8_t _addr;
};
#endif
EOF
  INCLUDES="$INCLUDES -I$STUB"
fi

echo "=== Compiling DRIVER_MODE=$MODE (atmega328p) ==="
"$AVR_GCC/avr-g++" \
  -c "$WORK" \
  -o "$OUTDIR/sketch_$MODE.o" \
  -mmcu=atmega328p \
  -DF_CPU=16000000L \
  -DARDUINO=10819 \
  -DARDUINO_AVR_UNO \
  -DARDUINO_ARCH_AVR \
  $INCLUDES \
  -include Arduino.h \
  -Os -w -std=gnu++11 -fpermissive -fno-exceptions \
  -ffunction-sections -fdata-sections -fno-threadsafe-statics \
  -Wall 2>&1

rc=$?
if [ $rc -eq 0 ]; then
  echo "[OK] DRIVER_MODE=$MODE compiled with no errors"
else
  echo "[FAIL] DRIVER_MODE=$MODE exit=$rc"
fi
exit $rc
