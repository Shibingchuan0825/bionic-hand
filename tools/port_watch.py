#!/usr/bin/env python
"""Watch COM ports appear and disappear.

The point is to answer one question that software alone cannot: *is the port
I have been trying to flash actually the board in front of me?*  Run this,
unplug the board, plug it back in, and see whether a port disappears.

Usage:
    python tools/port_watch.py            # watch for 30 s
    python tools/port_watch.py 60         # watch for 60 s
"""

import sys
import time

try:
    from serial.tools import list_ports
except ImportError:
    sys.exit("pyserial is not installed:  pip install pyserial")


def snapshot():
    return {p.device: p for p in list_ports.comports()}


def describe(port):
    bits = [port.description or "?"]
    if port.vid is not None and port.pid is not None:
        bits.append(f"VID_{port.vid:04X}&PID_{port.pid:04X}")
    if port.serial_number:
        bits.append(port.serial_number)
    return "  ".join(bits)


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0

    before = snapshot()
    print("当前串口:")
    if not before:
        print("  (没有)")
    for dev, p in sorted(before.items()):
        print(f"  {dev}  {describe(p)}")
    print()
    print(f"现在请拔掉 Arduino 板子，等 2 秒再插回去。监听 {seconds:.0f} 秒……")
    print()

    started = time.time()
    known = dict(before)
    seen_change = False
    while time.time() - started < seconds:
        time.sleep(0.4)
        now = snapshot()
        for dev in sorted(set(known) - set(now)):
            print(f"[拔掉] {dev}  {describe(known[dev])}", flush=True)
            seen_change = True
        for dev in sorted(set(now) - set(known)):
            print(f"[插上] {dev}  {describe(now[dev])}", flush=True)
            seen_change = True
        known = now

    print()
    if seen_change:
        print("结论: 串口随插拔变化 —— 这个口就是你要烧的板子。")
    else:
        print("结论: 插拔期间端口没有任何变化。")
        print("      要么板子没插、要么插的是另一个设备、要么这根线只能供电不能通信。")
        print("      换一根确认能传数据的 USB 线（有些线只有电源两根芯）再试。")


if __name__ == "__main__":
    main()
