#!/usr/bin/env bash
# Full firmware build for the Arduino Uno: compiles the sketch, the Arduino
# core, and the required bundled libraries, then links a flashable .hex.
#
# DRIVER_MODE=0 links against the real Servo library.
# DRIVER_MODE=1 links against a declaration stub, because
# Adafruit_PWMServoDriver is not installed in this environment.  The stub
# verifies that our code compiles and links; it is not the real driver.
set -u

ARDUINO15="$HOME/AppData/Local/Arduino15"
AVR="$ARDUINO15/packages/arduino/tools/avr-gcc/7.3.0-atmel3.6.1-arduino7/bin"
HW="$ARDUINO15/packages/arduino/hardware/avr/1.8.6"
CORE="$HW/cores/arduino"
VARIANT="$HW/variants/standard"
CORELIB="$HW/libraries"
SERVO_LIB="$ARDUINO15/libraries/Servo/src"

SKETCH="${1:-/d/hand/arduino/bionic_hand/bionic_hand.ino}"
MODE="${2:-0}"
OUT="${3:-/c/TEMP/bh_build}"

MCU_FLAGS="-mmcu=atmega328p -DF_CPU=16000000L -DARDUINO=10819 -DARDUINO_AVR_UNO -DARDUINO_ARCH_AVR"
CFLAGS="-Os -w -ffunction-sections -fdata-sections"
CXXFLAGS="$CFLAGS -std=gnu++11 -fpermissive -fno-exceptions -fno-threadsafe-statics"

rm -rf "$OUT"; mkdir -p "$OUT/obj"

INC="-I$CORE -I$VARIANT -I$CORELIB/Wire/src -I$CORELIB/SPI/src"
if [ "$MODE" = "0" ]; then
  INC="$INC -I$SERVO_LIB -I$SERVO_LIB/avr"
else
  mkdir -p "$OUT/stub"
  cat > "$OUT/stub/Adafruit_PWMServoDriver.h" <<'EOF'
// Compile/link stub for the Adafruit PCA9685 API used by this sketch.
// NOT the real library: it only proves our own code builds.
#ifndef ADAFRUIT_PWMSERVODRIVER_STUB_H
#define ADAFRUIT_PWMSERVODRIVER_STUB_H
#include <stdint.h>
class Adafruit_PWMServoDriver {
 public:
  explicit Adafruit_PWMServoDriver(uint8_t addr = 0x40) : _addr(addr) {}
  void begin() {}
  void setPWMFreq(float) {}
  void setPWM(uint8_t, uint16_t, uint16_t) {}
 private:
  uint8_t _addr;
};
#endif
EOF
  INC="$INC -I$OUT/stub"
fi

WORK="$OUT/bionic_hand.cpp"
sed "s/^#define DRIVER_MODE [0-9]\+/#define DRIVER_MODE $MODE/" "$SKETCH" > "$WORK"

echo "=== Building DRIVER_MODE=$MODE ==="

# --- sketch ---
"$AVR/avr-g++" -c "$WORK" -o "$OUT/obj/sketch.o" $MCU_FLAGS $INC \
  -include Arduino.h $CXXFLAGS || exit 1

# --- Arduino core ---
for src in "$CORE"/*.c; do
  [ -e "$src" ] || continue
  "$AVR/avr-gcc" -c "$src" -o "$OUT/obj/$(basename "$src" .c).o" $MCU_FLAGS $INC $CFLAGS || exit 1
done
for src in "$CORE"/*.cpp; do
  [ -e "$src" ] || continue
  "$AVR/avr-g++" -c "$src" -o "$OUT/obj/$(basename "$src" .cpp).o" $MCU_FLAGS $INC $CXXFLAGS || exit 1
done
# The standard Uno variant is header-only (pins_arduino.h); compile a .c only
# if a board package actually ships one.
if [ -f "$VARIANT/pins_arduino.c" ]; then
  "$AVR/avr-gcc" -c "$VARIANT/pins_arduino.c" -o "$OUT/obj/pins_arduino.o" \
    $MCU_FLAGS $INC $CFLAGS || exit 1
fi

# --- libraries ---
if [ "$MODE" = "0" ]; then
  # The modern Servo library splits per-architecture sources under src/avr.
  "$AVR/avr-g++" -c "$SERVO_LIB/avr/Servo.cpp" -o "$OUT/obj/Servo.o" \
    $MCU_FLAGS $INC -I"$SERVO_LIB" $CXXFLAGS || exit 1
else
  for src in "$CORELIB/Wire/src"/*.cpp; do
    [ -e "$src" ] || continue
    "$AVR/avr-g++" -c "$src" -o "$OUT/obj/$(basename "$src" .cpp).o" \
      $MCU_FLAGS $INC $CXXFLAGS || exit 1
  done
  for src in "$CORELIB/Wire/src/utility"/*.c; do
    [ -e "$src" ] || continue
    "$AVR/avr-gcc" -c "$src" -o "$OUT/obj/$(basename "$src" .c).o" \
      $MCU_FLAGS $INC $CFLAGS || exit 1
  done
fi

# --- link ---
"$AVR/avr-gcc" -o "$OUT/firmware.elf" "$OUT"/obj/*.o $MCU_FLAGS \
  -Wl,--gc-sections -mmcu=atmega328p -lm || exit 1

"$AVR/avr-objcopy" -O ihex -R .eeprom "$OUT/firmware.elf" "$OUT/firmware.hex" || exit 1

SIZE=$("$AVR/avr-size" -A "$OUT/firmware.elf" | awk '/\.text|\.data/{s+=$2} END{print s}')
echo "[OK] DRIVER_MODE=$MODE built: $OUT/firmware.hex (${SIZE} bytes flash)"
