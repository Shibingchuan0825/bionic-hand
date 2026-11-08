#!/usr/bin/env python
"""Ask a board's bootloader to identify itself, continuously.

avrdude opens the port, tries ~10 times within a second, then gives up.  On a
clone whose auto-reset (DTR) line is not wired, that whole window can pass
without the bootloader ever running, and the failure looks identical to a dead
board.  This script instead keeps knocking for the whole duration, so pressing
RESET by hand lands inside the window.

A live bootloader answers the sync request with the two bytes 0x14 0x10.

Usage:
    python tools/bootloader_probe.py COM5 45
"""

import sys
import time

try:
    import serial
except ImportError:
    sys.exit("pyserial is not installed:  pip install pyserial")

SYNC = bytes([0x30, 0x20])   # STK_GET_SYNC + CRC_EOP


def probe(port, baud, seconds):
    try:
        s = serial.Serial(port, baud, timeout=0.05)
    except Exception as e:
        print(f"[{baud}] 打不开端口: {e}")
        return False
    try:
        # Release DTR/RTS so the board is NOT held in reset; the user presses
        # the button instead.
        s.dtr = False
        s.rts = False
        deadline = time.time() + seconds
        hits = 0
        while time.time() < deadline:
            s.write(SYNC)
            data = s.read(8)
            if data:
                hits += 1
                print(f"[{baud}] 收到 {data.hex()}", flush=True)
                if data[:2] == b"\x14\x10":
                    print(f"[{baud}] *** bootloader 活着 ***", flush=True)
                    return True
        if hits == 0:
            print(f"[{baud}] {seconds:.0f} 秒内没有任何响应", flush=True)
        return False
    finally:
        s.close()


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "COM5"
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0

    print(f"探测 {port}，每个波特率 {seconds:.0f} 秒。", flush=True)
    print("现在按板子上的 RESET 按钮，每个波特率期间都按两三次。", flush=True)
    for baud in (115200, 57600):
        if probe(port, baud, seconds):
            return 0
    print()
    print("两个波特率都没有回应 —— bootloader 不存在或芯片没在运行。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
