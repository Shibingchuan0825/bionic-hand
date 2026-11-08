#!/usr/bin/env bash
# Flash the compiled firmware onto a real Arduino Uno over USB.
#
# Uses the avrdude that ships with the Arduino IDE package, so the Arduino IDE
# itself is not required.  Run tools/build_firmware.sh first to produce the hex.
#
# Usage:
#   bash tools/flash_firmware.sh COM3            # flash DRIVER_MODE 0 (default)
#   bash tools/flash_firmware.sh COM3 1          # flash a DRIVER_MODE 1 build
#   bash tools/flash_firmware.sh COM3 0 /c/TEMP/bh_m0
set -u

PORT="${1:-}"
MODE="${2:-0}"
BUILD="${3:-/c/TEMP/bh_m$MODE}"
HEX="$BUILD/firmware.hex"

ARDUINO15="$HOME/AppData/Local/Arduino15"
AVRDUDE="$ARDUINO15/packages/arduino/tools/avrdude/6.3.0-arduino17/bin/avrdude.exe"
AVRDUDE_CONF="$ARDUINO15/packages/arduino/tools/avrdude/6.3.0-arduino17/etc/avrdude.conf"

if [ -z "$PORT" ]; then
  echo "用法: bash tools/flash_firmware.sh <COM口> [模式] [构建目录]"
  echo
  echo "先看看有哪些串口:"
  python tools/first_contact.py --list
  exit 1
fi

if [ ! -f "$HEX" ]; then
  echo "[FAIL] 找不到固件: $HEX"
  echo "先编译:  bash tools/build_firmware.sh arduino/bionic_hand/bionic_hand.ino $MODE $BUILD"
  exit 1
fi

echo "=== 烧录 ==="
echo "  端口: $PORT"
echo "  固件: $HEX  ($(wc -c < "$HEX") 字节)"
echo "  模式: DRIVER_MODE=$MODE"
echo

# avrdude.exe is a native Windows binary, so it cannot open MSYS-style paths
# like /c/TEMP/... — convert both paths to Windows form first.
if command -v cygpath >/dev/null 2>&1; then
  HEX_WIN="$(cygpath -w "$HEX")"
  CONF_WIN="$(cygpath -w "$AVRDUDE_CONF")"
else
  HEX_WIN="$HEX"
  CONF_WIN="$AVRDUDE_CONF"
fi

# -carduino: the Uno's bootloader protocol (auto-reset via DTR).
# -b 115200: the Uno bootloader's baud rate, NOT the sketch's SERIAL_BAUD.
"$AVRDUDE" -C "$CONF_WIN" \
  -p atmega328p -c arduino -P "$PORT" -b 115200 \
  -D -U "flash:w:$HEX_WIN:i" || exit 1

echo
echo "[OK] 烧录完成。下一步:"
echo "  python tools/first_contact.py --port $PORT          # 只读层: PING/PONG"
echo "  python tools/first_contact.py --port $PORT --move   # 加舵机测试"
