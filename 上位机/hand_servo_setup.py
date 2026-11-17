"""Single-device HLS commissioning. Never writes torque-enable or goal position.

Reference: https://ftservo.github.io/ftservo-wiki/reference/parameter/memory-hls/
Identity/baud changes take effect before a normal write/readback round trip can
finish. Send a unicast write, then verify using the new communication endpoint.
"""
import time
import tkinter as tk
from tkinter import messagebox, ttk

from hand_control import HLS_BAUD_RATES, HlsBus, integer
from hand_dashboard import BG, PANEL, MUTED, CYAN

LOCK = 55
CONFIG_BYTES = frozenset((5, 6, 7, 8, 30, 33, LOCK))
RX_TIMEOUT = -6  # FTServo SDK COMM_RX_TIMEOUT


def read_servo_settings(bus, sid):
    integer(sid, 0, 253, "舵机主 ID")
    info = bus.probe(sid)
    code = info.get("baud_code")
    if code is None:
        code = bus.read_int(sid, 6, 1)
    if code not in range(len(HLS_BAUD_RATES)):
        raise RuntimeError("舵机波特率档位不属于 HLS 内存表")
    return {**info, "baud_code": code, "baud_rate": HLS_BAUD_RATES[code],
            "eeprom_locked": bus.read_int(sid, LOCK, 1) == 1}


def responds(bus, sid):
    data, result, error = bus.packet.readTxRx(sid, 5, 1)
    if result == RX_TIMEOUT:
        return False
    bus.check(result, error, f"检查主 ID {sid}")
    if len(data) != 1 or data[0] != sid:
        raise RuntimeError(f"ID {sid} 的识别应答异常；请检查是否重复 ID 或接线干扰")
    return True


def scan_servo_ids(bus, *, stop_event=None, notice=None):
    found = []
    for sid in range(254):
        if stop_event is not None and stop_event.is_set():
            break
        if responds(bus, sid):
            found.append(read_servo_settings(bus, sid))
        if notice is not None and sid % 32 == 0:
            notice(f"只读扫描至 ID {sid}，已发现 {len(found)} 台")
    return found


def _send_byte(bus, sid, address, value, settle):
    if address not in CONFIG_BYTES:
        raise ValueError("新舵机配置不允许写入该地址")
    integer(sid, 0, 253, "舵机主 ID")  # never a broadcast
    integer(value, 0, 255, "参数值")
    result = bus.packet.writeTxOnly(sid, address, 1, [value])
    bus.check(result, 0, f"ID {sid} 写配置 {address}")
    bus.port.ser.flush()
    settle(.02)


def _write_byte(bus, sid, address, value, settle):
    _send_byte(bus, sid, address, value, settle)
    if bus.read_int(sid, address, 1) != value:
        raise RuntimeError(f"ID {sid} 配置地址 {address} 回读不符")


def _host_baud(bus, baud):
    if bus.port.getBaudRate() != baud:
        if not bus.port.setBaudRate(baud):
            raise RuntimeError(f"串口无法切换到 {baud}")
        bus.port.ser.write_timeout = .2


def _relock(bus, endpoints, settle):
    checked = set()
    for sid, baud in endpoints:
        if (sid, baud) in checked:
            continue
        checked.add((sid, baud))
        try:
            bus.port.is_using = False
            _host_baud(bus, baud)
            if bus.read_int(sid, 5, 1) != sid:
                continue
            _write_byte(bus, sid, LOCK, 1, settle)
            return sid, baud
        except Exception:
            continue
    return None


def configure_servo(bus, current_id, new_id, new_baud, *, isolated=False,
                    initialize=True, expected=None, stop_event=None, settle=time.sleep):
    """Apply a single-servo EEPROM transaction and relock even after partial failure."""
    integer(current_id, 0, 253, "当前主 ID")
    integer(new_id, 0, 253, "新主 ID")
    if new_baud not in HLS_BAUD_RATES:
        raise ValueError("请选择受支持的 HLS 波特率")
    if isolated is not True:
        raise ValueError("修改前必须确认总线上只接待配置的这一台舵机")

    def cancellation():
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("配置已取消")

    cancellation()
    before = read_servo_settings(bus, current_id)
    HlsBus.check_health(before)
    if (before["primary_id"] != current_id or before["model_number"] != 2570
            or before.get("firmware_major") != 3 or before.get("endian") != 0):
        raise ValueError("识别结果不符合本项目 HL-3606-C001 / HLS 舵机，请勿写入")
    if before["torque_enabled"] or abs(before["speed_rpm"]) > 1:
        raise ValueError("舵机需要静止且扭矩关闭；请先停止运动再配置")
    old_baud = bus.port.getBaudRate()
    if before["baud_rate"] != old_baud:
        raise ValueError("读取的波特率档位与当前连接不符")
    if expected is not None:
        keys = ("primary_id", "model_number", "firmware_major", "firmware_minor", "baud_rate")
        if any(before.get(key) != expected.get(key) for key in keys):
            raise ValueError("设备与上次读取结果不符，请重新读取后配置")
    if new_id != current_id and responds(bus, new_id):
        raise ValueError(f"目标 ID {new_id} 已有设备应答；请只连接待配置舵机并选择未占用的 ID")
    cancellation()
    sid, baud = current_id, old_baud
    failure, unlocked, endpoint = None, False, None
    try:
        unlocked = True  # a lost reply must not skip relocking
        _write_byte(bus, sid, LOCK, 0, settle)
        if initialize:
            for address, value, previous in ((8, 1, before["reply_level"]),
                                             (30, 1, before["resolution"]), (33, 0, before["mode"])):
                cancellation()
                if previous != value:
                    _write_byte(bus, sid, address, value, settle)
        cancellation()
        if new_id != sid:
            old_id, sid = sid, new_id
            _send_byte(bus, old_id, 5, new_id, settle)
            if bus.read_int(sid, 5, 1) != new_id:
                raise RuntimeError("新 ID 回读不符")
        cancellation()
        if bus.read_int(sid, 7, 1) != new_id:
            _write_byte(bus, sid, 7, new_id, settle)
        cancellation()
        if new_baud != baud:
            baud = new_baud
            _send_byte(bus, sid, 6, HLS_BAUD_RATES.index(new_baud), settle)
            _host_baud(bus, new_baud)
            if bus.read_int(sid, 6, 1) != HLS_BAUD_RATES.index(new_baud):
                raise RuntimeError("新波特率回读不符")
        cancellation()
    except BaseException as exc:
        failure = exc
    finally:
        if unlocked:
            endpoint = _relock(bus, ((sid, baud), (new_id, new_baud), (new_id, old_baud),
                                     (current_id, old_baud), (current_id, new_baud)), settle)
    if failure is not None or endpoint != (new_id, new_baud):
        recovery = (f"已在 ID {endpoint[0]} / {endpoint[1]} 回锁。请按这个地址重新读取。"
                    if endpoint is not None else
                    f"未确认回锁；请仅连接此舵机，检查 ID {current_id} / {new_id}、波特率 {old_baud} / {new_baud}，重新读取。")
        raise RuntimeError(f"配置未完整完成：{failure or '新地址验证失败'}。{recovery}") from failure
    if stop_event is not None and stop_event.is_set():
        raise RuntimeError(f"配置期间收到取消；已在 ID {new_id} / {new_baud} 回锁，请按新地址读取核对。")
    try:
        after = read_servo_settings(bus, new_id)
    except Exception as exc:
        raise RuntimeError(f"写入后已在 ID {new_id} / {new_baud} 回锁，但最终读取失败：{exc}。请按新地址重新读取。") from exc
    if (after["primary_id"] != new_id or after["secondary_id"] != new_id
            or after["baud_rate"] != new_baud or not after["eeprom_locked"] or after["torque_enabled"] != 0):
        raise RuntimeError(f"ID {new_id} / {new_baud} 最终参数未通过完整回读，请重新读取后再接回多舵机总线")
    if initialize and (after["mode"], after["resolution"], after["reply_level"]) != (0, 1, 1):
        raise RuntimeError("位置模式初始化未通过回读")
    return {"before": before, "after": after}


class ServoSetupDialog:
    def __init__(self, app):
        self.app, self.observed, self.read_endpoint = app, None, None
        self.timer = None
        self.window = tk.Toplevel(app.root)
        self.window.title("新舵机配置 · ID 与波特率")
        self.window.configure(bg=BG)
        self.window.geometry("760x650")
        self.window.minsize(700, 620)
        self.window.transient(app.root)
        self.window.grab_set()
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Escape>", lambda _: self.close())
        footer = tk.Frame(self.window, bg=PANEL, padx=16, pady=10)
        footer.pack(side="bottom", fill="x")
        body = tk.Frame(self.window, bg=PANEL, padx=16, pady=14)
        body.pack(fill="both", expand=True)
        app.label(body, "新舵机配置", size=15, bold=True).pack(anchor="w")
        warning = app.label(body, "改 ID 前，只在 Hub 上连接待配置的这一台舵机，先断开其他舵机。\n"
                            "新舵机常用默认 ID 1；重复 ID 无法靠扫描可靠区分。本窗口不发送转动指令。", color=MUTED, size=9)
        warning.configure(wraplength=710, justify="left")
        warning.pack(fill="x", pady=(5, 10))
        self.port = tk.StringVar(value=app.port.get())
        self.current_id = tk.StringVar(value="1")
        self.current_baud = tk.StringVar(value=app.baud.get())
        self.new_id = tk.StringVar(value="2")
        self.new_baud = tk.StringVar(value="1000000")
        self.initialize = tk.BooleanVar(value=True)
        self.isolated = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="先读取当前参数；不知道 ID 时可按当前波特率扫描。")
        self.editors = []
        form = tk.Frame(body, bg=PANEL)
        form.pack(fill="x")
        for row, specs in enumerate(((("串口", self.port, None), ("当前 ID", self.current_id, None),
                                      ("当前波特率", self.current_baud, HLS_BAUD_RATES)),
                                     (("新 ID", self.new_id, None), ("新波特率", self.new_baud, HLS_BAUD_RATES)))):
            for column, (label, variable, values) in enumerate(specs):
                app.label(form, label, color=MUTED, size=9).grid(row=row*2, column=column, sticky="w", pady=(5, 2))
                widget = (ttk.Combobox(form, textvariable=variable, values=values, state="readonly", width=16)
                          if values else ttk.Entry(form, textvariable=variable, width=14))
                widget.grid(row=row*2+1, column=column, sticky="ew", padx=(0, 10), pady=(0, 4))
                self.editors.append((widget, "readonly" if values else "normal"))
                form.columnconfigure(column, weight=1)
        buttons = tk.Frame(body, bg=PANEL)
        buttons.pack(fill="x", pady=8)
        self.read_button = ttk.Button(buttons, text="读取当前参数", command=self.read)
        self.read_button.pack(side="left")
        self.scan_button = ttk.Button(buttons, text="扫描 ID（只读）", command=self.scan)
        self.scan_button.pack(side="left", padx=8)
        self.cancel_button = ttk.Button(buttons, text="停止当前操作", command=app.stop)
        self.cancel_button.pack(side="right")
        app.label(body, "同一总线的舵机需使用相同波特率。整手可按计划使用 ID 1–20。", color=MUTED, size=9).pack(anchor="w")
        self.details = tk.Text(body, height=6, wrap="word", bg=BG, fg=CYAN, relief="flat", font=("Microsoft YaHei UI", 10), padx=10, pady=7)
        self.details.pack(fill="both", expand=True, pady=8)
        self.show("尚未读取设备。读取和扫描不要求舵机已经绑定关节。")
        self.init_check = ttk.Checkbutton(body, text="同时设置位置模式 0、分辨率 1、应答级别 1", variable=self.initialize)
        self.init_check.pack(anchor="w", pady=(2, 4))
        self.isolation_check = ttk.Checkbutton(body, text="我确认总线上只接待配置的这一台舵机", variable=self.isolated)
        self.isolation_check.pack(anchor="w", pady=(0, 5))
        status = app.label(body, var=self.status, color=MUTED, size=9)
        status.configure(wraplength=710, justify="left")
        status.pack(fill="x", pady=(0, 7))
        ttk.Button(footer, text="关闭", command=self.close).pack(side="right")
        self.apply_button = ttk.Button(footer, text="写入 ID 与配置…", command=self.apply, style="Accent.TButton")
        self.apply_button.pack(side="right", padx=8)
        self.refresh()

    def show(self, text):
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", text)
        self.details.configure(state="disabled")

    def connection(self):
        port = self.port.get().strip()
        if not port:
            raise ValueError("请填写 FE-URT-2 对应串口")
        baud = int(self.current_baud.get())
        if baud not in HLS_BAUD_RATES:
            raise ValueError("请选择当前波特率")
        sid = integer(int(self.current_id.get()), 0, 253, "当前 ID")
        return port, baud, sid

    def summary(self, info):
        return (f"主 ID：{info['primary_id']}    副 ID：{info['secondary_id']}\n"
                f"波特率：{info['baud_rate']}    型号编号：{info['model_number']}\n"
                f"模式：{info['mode']}    分辨率：{info['resolution']}    应答级别：{info['reply_level']}\n"
                f"固件：{info['firmware_major']}.{info['firmware_minor']}    EEPROM：{'已回锁' if info['eeprom_locked'] else '未回锁'}\n"
                f"扭矩：{'关闭' if info['torque_enabled'] == 0 else '开启'}    电压：{info['voltage_v']} V")

    def read(self):
        if self.app.busy:
            return
        try:
            port, baud, sid = self.connection()
        except ValueError as exc:
            messagebox.showerror("读取设置", str(exc), parent=self.window)
            return
        self.observed, self.read_endpoint = None, None

        def done(info):
            self.observed, self.read_endpoint = info, (port, baud, sid)
            self.show(self.summary(info))
            self.status.set("参数已读取。填写新 ID 后，确认只连接这一台，再写入。")

        self.app.start_job(f"只读识别 ID {sid} 的设备配置。", lambda bus: read_servo_settings(bus, sid),
                           on_success=done, connection=(port, baud))

    def scan(self):
        if self.app.busy:
            return
        try:
            port, baud, _ = self.connection()
        except ValueError as exc:
            messagebox.showerror("扫描设置", str(exc), parent=self.window)
            return
        self.observed, self.read_endpoint = None, None

        def done(found):
            if len(found) == 1:
                info = found[0]
                self.current_id.set(str(info["primary_id"]))
                self.observed, self.read_endpoint = info, (port, baud, info["primary_id"])
                self.show(self.summary(info))
                self.status.set("扫描发现一组有效应答；仍须确认物理上只连接这一台后写入。")
            else:
                self.show("扫描结果：" + (", ".join(f"ID {info['primary_id']}" for info in found) or "未发现设备")
                          + "\n改 ID 前请只连接一台，并用正确的当前 ID 重新读取。")

        self.app.start_job(f"按 {baud} 只读扫描 ID 0–253，可按停止取消。",
            lambda bus: scan_servo_ids(bus, stop_event=self.app.cancel,
                notice=lambda text: self.app.events.put(("notice", text))),
            on_success=done, connection=(port, baud))

    def apply(self):
        if self.app.busy:
            return
        try:
            port, baud, sid = self.connection()
            new_id = integer(int(self.new_id.get()), 0, 253, "新 ID")
            new_baud = int(self.new_baud.get())
            if new_baud not in HLS_BAUD_RATES:
                raise ValueError("请选择新波特率")
            if not self.isolated.get():
                raise ValueError("请先断开其他舵机，并勾选只连接这一台的确认项")
            if self.observed is None or self.read_endpoint != (port, baud, sid):
                raise ValueError("请先按当前串口、ID 和波特率读取一次参数")
        except ValueError as exc:
            messagebox.showinfo("新舵机配置", str(exc), parent=self.window)
            return
        if not messagebox.askokcancel("写入这一台舵机的配置",
                f"主 ID：{sid} → {new_id}\n波特率：{baud} → {new_baud}\n"
                "副 ID 将同步为新主 ID。参数写入掉电保存区域，回读后重新锁定。\n"
                "确认 Hub 总线上只有这一台待配置舵机。", parent=self.window):
            return
        expected = self.observed.copy()
        initialize = self.initialize.get()
        self.observed, self.read_endpoint = None, None

        def done(result):
            info = result["after"]
            self.current_id.set(str(new_id))
            self.current_baud.set(str(new_baud))
            self.app.port.set(port)
            self.app.baud.set(str(new_baud))
            self.observed, self.read_endpoint = info, (port, new_baud, new_id)
            self.show(self.summary(info))
            self.status.set(f"配置已回读并回锁：ID {new_id}。重新上电后可再次读取核对；接着在「关节绑定」添加此 ID。")
            self.app.log("配置", f"新舵机配置完成：ID {sid} → {new_id}，波特率 {baud} → {new_baud}；未发送运动指令。")

        self.app.start_job("正在配置单台舵机并回读、回锁；不发送运动指令。",
            lambda bus: configure_servo(bus, sid, new_id, new_baud, isolated=True,
                initialize=initialize, expected=expected, stop_event=self.app.cancel),
            on_success=done, connection=(port, baud))

    def refresh(self):
        busy = self.app.busy
        for widget, state in self.editors:
            widget.configure(state="disabled" if busy else state)
        for widget in (self.read_button, self.scan_button, self.init_check, self.isolation_check):
            widget.configure(state="disabled" if busy else "normal")
        self.cancel_button.configure(state="normal" if busy else "disabled")
        self.apply_button.configure(state="normal" if not busy and self.observed is not None and self.isolated.get() else "disabled")
        self.timer = self.window.after(100, self.refresh)

    def close(self):
        if self.app.busy:
            self.status.set("请等待回读和回锁完成后关闭。可用「停止当前操作」取消，程序会尝试回锁。")
            return "break"
        if self.timer is not None:
            self.window.after_cancel(self.timer)
        self.window.destroy()
        return "break"
