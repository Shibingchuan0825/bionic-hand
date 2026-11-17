"""直驱灵巧手桌面控制台。"""
from collections import deque
import csv
import json
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from hand_control import (
    HOST, ROOT, HandConfig, calibrate_bindings, initial_angles, integer, jog,
    pose_timeout_message, pose_timeout_seconds, serial_bus, serve,
)
from hand_dashboard import (DashboardView, BG, PANEL, INPUT, LINE, TEXT, MUTED,
                            CYAN, BLUE, ORANGE, FONT, MONO, FINGERS, AXES)
from hand_motion import PRESETS, preset_pose
from hand_bindings import BindingEditor
from hand_servo_setup import ServoSetupDialog


def set_style(root):
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", font=FONT, background=PANEL, foreground=TEXT,
                    bordercolor=LINE, lightcolor=LINE, darkcolor=LINE)
    style.configure("TButton", padding=(12, 7), background=INPUT, font=("Microsoft YaHei UI", 9), borderwidth=1)
    style.configure("Small.TButton", padding=(7, 4))
    style.configure("Quiet.TButton", padding=(9, 5), background=PANEL)
    style.configure("Nudge.TButton", padding=(2, 0), font=("Consolas", 9))
    style.map("TButton", background=[("active", LINE)], foreground=[("disabled", "#7790a8")])
    style.configure("Accent.TButton", background=CYAN, foreground=BG, font=("Microsoft YaHei UI", 9, "bold"))
    style.map("Accent.TButton", background=[("disabled", INPUT), ("active", "#a1efe4")],
              foreground=[("disabled", "#7790a8")])
    style.configure("Stop.TButton", background="#71323f", foreground="#ffe8ec", font=("Microsoft YaHei UI", 9, "bold"))
    style.map("Stop.TButton", background=[("disabled", INPUT), ("active", "#964354")],
              foreground=[("disabled", "#7790a8")])
    style.configure("TEntry", fieldbackground=INPUT, insertcolor=TEXT, padding=5)
    style.configure("TCombobox", fieldbackground=INPUT, background=INPUT, padding=5,
                    arrowcolor=TEXT)
    style.map("TCombobox", fieldbackground=[("readonly", INPUT)], foreground=[("readonly", TEXT)])
    style.configure("Horizontal.TScale", troughcolor=INPUT, background=CYAN)
    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure("TNotebook.Tab", background=BG, foreground=MUTED, padding=(13, 8))
    style.map("TNotebook.Tab", background=[("selected", PANEL)], foreground=[("selected", CYAN)])
    style.configure("TCheckbutton", background=PANEL, foreground=MUTED, indicatorbackground=INPUT)
    style.map("TCheckbutton", background=[("active", PANEL)])
    style.configure("Treeview", background=PANEL, fieldbackground=PANEL, rowheight=28,
                    foreground=TEXT, borderwidth=0)
    style.configure("Treeview.Heading", background=INPUT, foreground=MUTED, padding=(7, 7), relief="flat")
    style.map("Treeview", background=[("selected", LINE)], foreground=[("selected", TEXT)])
    root.option_add("*TCombobox*Listbox.background", INPUT)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)


class Console(DashboardView):
    def __init__(self, root, *, parent=None):
        self.root = root
        self.content_parent = parent if parent is not None else root
        self.embedded = parent is not None
        self.closed = False
        if not self.embedded:
            self.root.title("控制工作台")
            self.root.configure(bg=BG)
            width, height = min(1400, root.winfo_screenwidth() - 32), min(850, root.winfo_screenheight() - 65)
            self.root.geometry(f"{width}x{height}+12+12")
            self.root.minsize(1120, 740)
        set_style(root)

        self.profile = tk.StringVar(value="整手 · 20 台")
        self.port = tk.StringVar()
        self.baud = tk.StringVar(value="1000000")
        self.finger = tk.StringVar()
        self.caption = tk.StringVar()
        self.message = tk.StringVar()
        self.state = tk.StringVar()
        self.stream_info = tk.StringVar(value="等待联动")
        self.connection_info = tk.StringVar(value="等待实机反馈")
        self.motion_title = tk.StringVar(value="等待实机操作")
        self.motion_note = tk.StringVar(value="调节角度和选择预设只更新目标；点击「发送当前姿态」后才执行。")
        self.chart_motor = tk.StringVar()
        self.mapping_group = tk.StringVar()
        self.mapping_inputs = tk.StringVar()
        self.mapping_formula = tk.StringVar()
        self.binding_status = tk.StringVar()
        self.fullscreen = False
        self.frame_count = 0
        self.frame_times = deque(maxlen=60)
        self.last_feedback = 0
        self.local_source = "手动预览"
        self.logs = deque(maxlen=500)
        self.profiles = {
            "食指 · 4 台": ROOT / "index.json", "对捏 · 8 台": ROOT / "pinch.json",
            "整手 · 20 台": ROOT / "full_hand.json",
        }
        self.busy = False
        self.job_success = None
        self.closing = False
        self.cancel = threading.Event()
        self.events = queue.Queue()
        self.config = None
        self.angles = {}
        self.readings = {}
        self.targets = {}
        self.zero_readings = {}
        self.samples = deque(maxlen=20000)
        self.controls = []

        self.build()
        self.load_profile()
        self.log("系统", "实机控制工作台已启动；等待选择串口与操作指令。")
        if not self.embedded:
            self.root.protocol("WM_DELETE_WINDOW", self.close)
            self.root.bind("<F11>", lambda _: self.toggle_fullscreen())
            self.root.bind("<Escape>", self.escape)
        self.poll_after = self.root.after(50, self.poll)

    def label(self, parent, text="", *, color=TEXT, size=10, bold=False, var=None):
        return tk.Label(parent, text=text, textvariable=var, bg=parent["bg"], fg=color,
                        font=("Microsoft YaHei UI", size, "bold" if bold else "normal"), anchor="w")


    def load_profile(self, path=None):
        if self.busy:
            return
        name = Path(path).name if path else self.profile.get()
        try:
            source = path or self.profiles[name]
            config = HandConfig.from_data(source.to_data()) if isinstance(source, HandConfig) else HandConfig(source)
        except (OSError, TypeError, ValueError, KeyError, AttributeError) as exc:
            if self.config is None:
                raise
            self.profile.set(self.loaded_profile)
            messagebox.showerror("配置无法打开", str(exc), parent=self.root)
            return
        if path:
            self.profiles[name] = Path(path)
            self.profile_box.configure(values=tuple(self.profiles))
            self.profile.set(name)
        self.loaded_profile = name
        self.config = config
        self.groups = {}
        for joint in config.joints:
            group = joint.partition("_")[0]
            self.groups.setdefault(FINGERS.get(group, group), []).append(joint)
        self.finger_box.configure(values=tuple(self.groups))
        self.finger.set(next(iter(self.groups)))
        self.angles = initial_angles(config)
        self.samples.clear()
        self.frame_count = 0
        self.frame_times.clear()
        self.chart_motor_box.configure(values=[f"ID {sid}" for sid in config.motors])
        self.chart_motor.set(f"ID {next(iter(config.motors))}" if config.motors else "")
        self.mapping_labels = {
            FINGERS.get(pair.name.partition("_")[0], pair.name) + " · " + pair.name.rpartition("_")[2].upper(): pair.name
            for pair in config.differential_pairs.values()
        }
        self.mapping_box.configure(values=tuple(self.mapping_labels))
        self.mapping_group.set(next(iter(self.mapping_labels), ""))
        self.rebuild_sliders()
        self.reset_feedback()
        self.log("配置", f"加载 {self.profile.get()}：{len(config.joints)} 个关节，{len(config.motors)} 台舵机；"
                 + ("配置标记为已标定。" if config.calibrated else "模板尚未标定。"))

    def open_profile(self):
        path = filedialog.askopenfilename(parent=self.root, filetypes=[("关节配置", "*.json")])
        if path:
            self.load_profile(path)

    def open_servo_setup(self):
        if self.busy:
            return
        dialog = getattr(self, "servo_setup_dialog", None)
        if dialog is not None and dialog.window.winfo_exists():
            dialog.window.lift()
            return
        self.servo_setup_dialog = ServoSetupDialog(self)

    def edit_bindings(self, joint=None):
        if not self.busy:
            self.binding_editor = BindingEditor(self, joint)

    def apply_binding_data(self, data, *, path=None, preserve_angles=False):
        if self.busy:
            return
        config = HandConfig.from_data(data)
        angles = self.angles.copy() if preserve_angles else None
        name = Path(path).name if path else "自定义关节绑定"
        self.profiles[name] = Path(path) if path else config
        self.profile_box.configure(values=tuple(self.profiles))
        self.profile.set(name)
        self.load_profile()
        if angles is not None and angles.keys() == config.joints.keys():
            self.display_angles(angles)
            self.refresh()
        self.message.set("绑定已应用；未绑定的关节仅预览，不发送。" +
                         ("已采集零位，可发送当前姿态。" if config.calibrated else "下一步：采集绑定零位。"))

    def save_bindings(self):
        if self.busy:
            return
        path = filedialog.asksaveasfilename(parent=self.root, initialfile="joint_bindings.json",
                                          defaultextension=".json", filetypes=[("关节绑定配置", "*.json")])
        if path:
            try:
                data = self.config.to_data()
                Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                self.apply_binding_data(data, path=path, preserve_angles=True)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.operation_error("保存绑定失败", exc)
                return
            self.log("配置", "保存关节绑定与零位至 " + str(path))

    def capture_binding_zero(self):
        if self.busy or not self.config.motors:
            return
        config = self.config
        ids = ", ".join(str(sid) for sid in config.motors)
        if not messagebox.askokcancel("采集绑定零位（只读）",
                f"只读取已绑定的 ID：{ids}。\n"
                "把机构放在各关节 0° 的姿态，保持静止且扭矩关闭。\n"
                "当前位置将作为舵机零位；软限位相对零位平移，方向和传动比例保持绑定设置。\n"
                "请确认绑定中填写的方向、比例和软限位适合实物。此步骤不发送转动指令。", parent=self.root):
            return

        def read(bus):
            readings = {}
            for sid in config.motors:
                if self.cancel.is_set():
                    return None
                readings[sid] = bus.probe(sid)
                self.report(readings[sid])
            return calibrate_bindings(config, readings).to_data(), readings

        def apply(result):
            data, readings = result
            self.apply_binding_data(data)
            self.readings = readings
            self.zero_readings = readings.copy()
            self.last_feedback = time.monotonic()
            self.refresh_table()
            self.update_metrics()

        self.start_job("正在只读采集已绑定舵机的零位。", read, on_success=apply)

    def reset_feedback(self):
        self.readings = {}
        self.zero_readings = {}
        self.samples.clear()
        self.frame_times.clear()
        self.frame_count = 0
        self.last_feedback = 0
        self.refresh()
        self.message.set("选择串口，配置并绑定舵机后采集零位；目标预览需点击发送才会执行。")
        self.draw_chart()

    def rebuild_sliders(self):
        for widget in self.sliders.winfo_children():
            widget.destroy()
        self.slider_vars = {}
        self.slider_widgets = []
        for index, joint in enumerate(self.groups[self.finger.get()]):
            low, high = self.config.joints[joint]
            row = tk.Frame(self.sliders, bg=PANEL)
            if getattr(self, "wide_joint_debug", False):
                self.sliders.columnconfigure(index % 2, weight=1, uniform="manual_joint")
                row.grid(row=index // 2, column=index % 2, sticky="ew", padx=(0, 12), pady=4)
            else:
                row.pack(fill="x", expand=True, pady=(2, 2))
            title = tk.Frame(row, bg=PANEL)
            title.pack(fill="x")
            axis = joint.partition("_")[2]
            self.label(title, AXES.get(axis, joint), size=10).pack(side="left")
            value = tk.DoubleVar(value=self.angles[joint])
            value_label = tk.Label(title, width=7, bg=PANEL, fg=CYAN, font=MONO, anchor="e",
                                   text=f"{value.get():.1f}°")
            value_label.pack(side="right")
            value_label.bind("<Double-1>", lambda _, key=joint: self.edit_joint(key))

            ids = [str(sid) for sid, motor in self.config.motors.items() if joint in motor.weights]
            binding_text = ("ID " + "/".join(ids) if len(ids) <= 2 else f"{len(ids)} 台") if ids else "仅预览"
            binding = self.label(title, binding_text, color=BLUE if ids else MUTED, size=8)
            binding.pack(side="left", padx=(8, 0))
            binding.configure(cursor="hand2")
            binding.bind("<Button-1>", lambda _, key=joint: self.edit_bindings(key))

            def change(raw, key=joint, label=value_label):
                self.angles[key] = round(float(raw), 1)
                label.configure(text=f"{self.angles[key]:.1f}°")
                self.local_source = "手动预览"
                self.refresh()

            slider_row = tk.Frame(row, bg=PANEL)
            slider_row.pack(fill="x")
            minus = ttk.Button(slider_row, text="−", width=2, style="Nudge.TButton", command=lambda key=joint: self.nudge_joint(key, -.5))
            minus.pack(side="left")
            scale = ttk.Scale(slider_row, from_=low, to=high, variable=value, command=change)
            scale.pack(side="left", fill="x", expand=True, padx=6)
            plus = ttk.Button(slider_row, text="+", width=2, style="Nudge.TButton", command=lambda key=joint: self.nudge_joint(key, .5))
            plus.pack(side="right")
            self.slider_vars[joint] = (value, value_label)
            self.slider_widgets.extend((minus, scale, plus))
        self.draw()
        group_key = self.groups[self.finger.get()][0].partition("_")[0]
        for label, name in self.mapping_labels.items():
            if name.startswith(group_key + "_"):
                self.mapping_group.set(label)
                break
        self.refresh_mapping()

    def refresh_controls(self):
        locked = self.busy
        for widget, state in self.controls:
            widget.configure(state="disabled" if locked else state)
        for widget in self.slider_widgets:
            widget.configure(state="disabled" if locked else "normal")
        for widget in (self.port_entry, self.baud_entry, self.servo_setup_button, self.port_refresh):
            widget.configure(state="disabled" if locked else "normal")
        self.move_button.configure(state="normal" if not locked and self.targets and self.config.calibrated else "disabled")
        self.jog_button.configure(state="disabled" if locked else "normal")
        self.binding_button.configure(state="disabled" if locked else "normal")
        self.binding_save_button.configure(state="disabled" if locked else "normal")
        self.binding_zero_button.configure(state="normal" if not locked and self.config.motors else "disabled")
        bound = {joint for motor in self.config.motors.values() for joint in motor.weights}
        self.binding_status.set(f"{len(bound)} 个关节已绑定 · {len(self.config.motors)} 台舵机\n"
            + (f"其余 {len(self.config.joints)-len(bound)} 个关节不发送" if len(bound) < len(self.config.joints) else "仅发送绑定表中的舵机")
            + (" · 需采集零位" if not self.config.calibrated and bound else ""))
        for button in self.preset_buttons:
            button.configure(state="disabled" if locked else "normal")
        for button in self.mapping_buttons:
            button.configure(state="disabled" if locked or not self.mapping_group.get() else "normal")
        self.stop_button.configure(state="normal" if locked else "disabled")
        self.stream_button.configure(state="normal" if self.config.motors and self.config.calibrated and not locked else "disabled")
        self.home_button.configure(state="disabled" if locked else "normal")
        self.export_button.configure(state="normal" if self.samples and not locked else "disabled")
        self.state.set("正在停止" if self.busy and self.cancel.is_set() else "实机 · 操作中" if self.busy else "实机 · 待操作")
        self.badge.configure(fg=ORANGE if self.cancel.is_set() and self.busy else CYAN if self.busy else MUTED)
        self.update_metrics()

    def refresh(self):
        try:
            self.targets = self.config.targets(self.angles)
        except ValueError as exc:
            self.targets = {}
            self.caption.set(str(exc))
        else:
            self.caption.set("已采集零位 · 点击下方「发送当前姿态」" if self.config.calibrated
                             else "发送前：设置绑定 → 采集绑定零位")
        self.draw()
        self.refresh_table()
        self.refresh_controls()
        self.update_metrics()

    def refresh_table(self):
        for index, (sid, motor) in enumerate(self.config.motors.items()):
            read = self.readings.get(sid, {})
            group = next(iter(motor.weights)).partition("_")[0]
            position = read.get("position_tick")
            target = self.targets.get(sid)
            error = f"{(target-position)*360/4096:+.2f}" if target is not None and position is not None else "—"
            value = (sid, FINGERS.get(group, group), self.targets.get(sid, "—"),
                     read.get("position_tick", "—"), read.get("speed_rpm", "—"),
                     read.get("motor_current_ma", "—"), read.get("voltage_v", "—"),
                     read.get("temperature_c", "—"),
                     ("正常" if read.get("status") == 0 else f"0x{read['status']:02X}" if "status" in read else "—"))
            value = value[:4] + (error,) + value[4:]
            tags = ("odd" if index % 2 else "even",) + (("fault",) if read.get("status", 0) else ())
            if self.table.exists(str(sid)):
                self.table.item(str(sid), values=value, tags=tags)
            else:
                self.table.insert("", "end", iid=str(sid), values=value, tags=tags)
        for row in self.table.get_children():
            if int(row) not in self.config.motors:
                self.table.delete(row)
        self.refresh_mapping()

    def refresh_mapping(self):
        if self.config is None:
            return
        name = self.mapping_labels.get(self.mapping_group.get())
        pair = self.config.differential_pairs.get(name)
        self.mapping_table.delete(*self.mapping_table.get_children())
        if pair is None:
            self.mapping_inputs.set("当前配置没有 MCP / CMC 双舵机差动组")
            self.mapping_formula.set("需要配置两个关节自由度及对应两台舵机的权重。")
            return
        flex, side = pair.flex_joint, pair.side_joint
        side_label = "旋转" if side.endswith("roll") else "侧摆"
        self.mapping_inputs.set(f"输入目标：屈曲 {self.angles[flex]:+.1f}°  /  {side_label} {self.angles[side]:+.1f}°")
        formulas = []
        for sid in pair.motor_ids:
            weights = self.config.motors[sid].weights
            formulas.append(f"ID {sid} = {weights[flex]:g}×屈曲 {weights[side]:+g}×{side_label}")
        self.mapping_formula.set("； ".join(formulas) + "  ·  舵机转角相对各自零位")
        try:
            plan = self.config.describe_mapping(self.angles)
        except ValueError as exc:
            self.mapping_inputs.set("目标无效：" + str(exc))
            return
        group_valid = all(detail["within_limits"] for detail in plan.values())
        for sid in pair.motor_ids:
            motor, detail = self.config.motors[sid], plan[sid]
            status = "整帧有效" if group_valid else "越界·整帧拒绝"
            values = (sid, f"{detail['contributions_deg'][flex]:+.2f}", f"{detail['contributions_deg'][side]:+.2f}",
                      f"{detail['motor_deg']:+.2f}", motor.zero_tick, detail["target_tick"],
                      f"{motor.min_tick}–{motor.max_tick}", status)
            self.mapping_table.insert("", "end", iid=str(sid), values=values, tags=() if group_valid else ("fault",))

    def select_mapping_group(self, _event=None):
        name = self.mapping_labels.get(self.mapping_group.get(), "")
        finger = FINGERS.get(name.partition("_")[0])
        if not self.busy and finger in self.groups and finger != self.finger.get():
            self.finger.set(finger)
            self.rebuild_sliders()
        self.refresh_mapping()

    def preview_differential(self, kind):
        if self.busy:
            return
        name = self.mapping_labels.get(self.mapping_group.get())
        pair = self.config.differential_pairs.get(name)
        if pair is None:
            return
        self.select_mapping_group()
        values = {"flex": (10, 0), "side": (0, 5), "combined": (10, 5), "zero": (0, 0)}
        flex, side = values[kind]
        angles = self.angles.copy()
        angles.update({pair.flex_joint: flex, pair.side_joint: side})
        try:
            self.config.targets(angles)
        except ValueError as exc:
            self.operation_error("差动预览超出当前配置", exc)
            return
        self.display_angles(angles)
        self.local_source = "差动预览 / " + name
        self.refresh()
        self.message.set("已预览双舵机联合映射；点击「发送当前姿态」后才执行。")
        self.log("操作", f"{name} 差动预览：屈曲 {flex}°，第二自由度 {side}°；未触发实机运动。")


    def display_angles(self, angles):
        self.angles = angles.copy()
        for name, (variable, label) in self.slider_vars.items():
            variable.set(angles[name])
            label.configure(text=f"{angles[name]:.1f}°")
        self.draw()

    def home(self):
        try:
            angles = dict.fromkeys(self.config.joints, 0)
            self.config.targets(angles)
        except ValueError as exc:
            messagebox.showerror("零位超出配置范围", str(exc), parent=self.root)
            return
        self.display_angles(angles)
        self.refresh()
        self.message.set("已预览标定零位。点击「发送当前姿态」确认回零。")


    def log(self, level, message):
        stamp = time.strftime("%H:%M:%S")
        item = (stamp, level, str(message))
        self.logs.append(item)
        row = self.log_table.insert("", "end", values=item, tags=(level,))
        children = self.log_table.get_children()
        if len(children) > 500:
            self.log_table.delete(children[0])
        self.log_table.see(row)

    def save_log(self):
        path = filedialog.asksaveasfilename(parent=self.root, initialfile="hand_console.log", defaultextension=".log", filetypes=[("运行日志", "*.log")])
        if path:
            try:
                Path(path).write_text("\n".join(f"{stamp} [{level}] {message}" for stamp, level, message in self.logs) + "\n", encoding="utf-8")
            except OSError as exc:
                self.operation_error("保存日志失败", exc)

    def operation_error(self, title, error):
        raw = str(error)
        if "FileNotFoundError" in raw or "could not open port" in raw:
            hint = "所选串口不存在或无法打开。连接 FE-URT-2 后点「刷新」，选择实际 COM 口。"
        elif "no status packet" in raw.lower():
            hint = "未收到舵机状态包。检查舵机外部供电、共地、TTL 接口、ID 与波特率；USB 串口存在不代表舵机已连接。"
        elif "PermissionError" in raw or "Access is denied" in raw:
            hint = "串口可能被其他程序占用。关闭飞特调试助手或其他串口程序后重试。"
        elif "8765" in raw or "10048" in raw or "10013" in raw:
            hint = "TCP 监听未能启动。检查 8765 端口是否被另一个控制台占用，以及本机网络权限。"
        else:
            hint = raw
        self.message.set(hint.replace("\n", " ")[:110])
        self.log("错误", title + "：" + raw)
        messagebox.showerror(title, hint + ("\n\n原始信息：\n" + raw if hint != raw else ""), parent=self.root)

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def escape(self, _event=None):
        self.stop()
        if self.fullscreen:
            self.toggle_fullscreen()
        return "break"


    def nudge_joint(self, joint, amount):
        if self.busy:
            return
        low, high = self.config.joints[joint]
        self.angles[joint] = round(max(low, min(high, self.angles[joint] + amount)), 2)
        self.display_angles(self.angles)
        self.local_source = "手动预览"
        self.refresh()

    def edit_joint(self, joint):
        if self.busy:
            return
        low, high = self.config.joints[joint]
        angle = simpledialog.askfloat("输入关节目标", f"{joint}\n范围 {low}° 至 {high}°；仅修改目标预览。", initialvalue=self.angles[joint], minvalue=low, maxvalue=high, parent=self.root)
        if angle is not None:
            self.angles[joint] = angle
            self.display_angles(self.angles)
            self.refresh()

    def apply_preset(self, key):
        if self.busy:
            return
        try:
            pose = preset_pose(self.config, key)
        except ValueError as exc:
            self.operation_error("预设超出当前配置", exc)
            return
        label, note = PRESETS[key]
        self.display_angles(pose)
        self.local_source = "预设 / " + label
        self.motion_title.set("姿态预览 · " + label)
        self.motion_note.set(note + "；点击「发送当前姿态」后才执行。")
        self.refresh()
        self.log("操作", "预览动作「" + label + "」，未向实机发送。")

    def save_pose(self):
        if self.busy:
            return
        try:
            self.config.targets(self.angles)
        except ValueError as exc:
            self.operation_error("当前目标无效", exc)
            return
        path = filedialog.asksaveasfilename(parent=self.root, initialfile="hand_pose.json", defaultextension=".json", filetypes=[("姿态参数", "*.json")])
        if path:
            try:
                Path(path).write_text(json.dumps({"format": "rapid-hand-pose/v1", "joints_deg": self.angles}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except OSError as exc:
                self.operation_error("保存姿态失败", exc)
                return
            self.log("操作", "已保存姿态 " + str(path))
            self.message.set("姿态参数已保存。")

    def load_pose(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(parent=self.root, filetypes=[("姿态参数", "*.json")])
        if path:
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
                if data.get("format") != "rapid-hand-pose/v1":
                    raise ValueError("请选择保存的姿态 JSON，配置文件需从左侧打开。")
                pose = data["joints_deg"]
                self.config.targets(pose)
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                self.operation_error("无法载入姿态", exc)
                return
            self.display_angles(pose)
            self.local_source = "载入姿态"
            self.refresh()
            self.log("操作", "已载入姿态预览 " + str(path))
            self.message.set("姿态已载入预览。点击「发送当前姿态」后才执行。")


    def select_chart_motor(self, _event=None):
        selected = self.table.selection()
        if selected:
            self.chart_motor.set("ID " + selected[0])
        self.draw_chart()


    def list_ports(self, _event=None):
        try:
            from serial.tools.list_ports import comports
        except ImportError:
            self.connection_info.set("串口库未安装，请按 README 安装依赖")
            return
        ports = list(comports())
        self.port_entry.configure(values=[port.device for port in ports])
        if _event is None:
            self.log("设备", "可用串口：" + ("；".join(f"{port.device} {port.description}" for port in ports) or "未检测到"))
            self.message.set("检测到 " + str(len(ports)) + " 个串口；请选择 FE-URT-2 对应端口，串口列表不表示舵机在线。")

    def show_details(self, event):
        row = self.table.identify_row(event.y)
        if not row:
            return
        info = self.readings.get(int(row), {})
        fields = {"model_number": "型号编号", "mode": "运行模式", "resolution": "分辨率系数",
                  "min_tick": "舵机最小编码", "max_tick": "舵机最大编码",
                  "secondary_id": "副 ID", "reply_level": "应答级别", "torque_enabled": "扭矩开关"}
        details = "\n".join(f"{label}  {info[key]}" for key, label in fields.items() if key in info)
        messagebox.showinfo(f"ID {row} 识别详情", details or "完成只读检查后可查看型号、模式与舵机限位。",
                            parent=self.root)

    def serial_settings(self):
        port = self.port.get().strip()
        if not port:
            raise ValueError("请填写 FE-URT-2 的串口，例如 COM5。")
        return port, int(self.baud.get())

    def start_job(self, label, task, *, on_success=None, connection=None):
        if self.busy:
            return
        try:
            port, baud = connection if connection is not None else self.serial_settings()
        except ValueError as exc:
            messagebox.showerror("串口设置", str(exc), parent=self.root)
            return
        self.busy = True
        self.job_success = on_success
        self.cancel.clear()
        self.readings.clear()
        self.zero_readings.clear()
        self.samples.clear()
        self.frame_count = 0
        self.frame_times.clear()
        self.refresh_table()
        self.message.set(label)
        self.motion_title.set("实机操作")
        self.motion_note.set(label)
        self.log("操作", label)
        self.refresh_controls()

        def worker():
            try:
                with serial_bus(port, baud) as bus:
                    result = task(bus)
            except Exception as exc:
                self.events.put(("error", str(exc)))
            else:
                self.events.put(("done", (self.cancel.is_set(), result)))

        self.worker = threading.Thread(target=worker, name="hand-worker")
        self.worker.start()

    def report(self, info):
        self.events.put(("reading", info))


    def start_stream(self):
        if self.busy:
            return
        config = self.config
        if not config.motors or not config.calibrated:
            return
        try:
            integer(config.model_number, 1, 65535, "model_number")
            self.serial_settings()
        except ValueError as exc:
            messagebox.showerror("视觉联动配置", str(exc), parent=self.root)
            return
        if not messagebox.askokcancel("确认实机遥操作",
            "收到有效视觉帧后将驱动已绑定舵机，首帧应接近当前姿态。\n"
            "确认已验证标定与限位，台架可随时断开舵机电源。", parent=self.root):
            return

        def stream(bus):
            return serve(config, bus, 8765, stop_event=self.cancel, nearby_start=True,
                on_frame=lambda angles, reply: self.events.put(("frame", (angles, reply, time.time()))),
                notice=lambda message: self.events.put(("notice", message)))

        self.stream_info.set("准备联动")
        self.draw_chart()
        self.start_job(f"等待外部视觉连接 {HOST}:8765。", stream)

    def record_frame(self, payload):
        angles, reply, timestamp = payload
        self.display_angles(angles)
        self.targets = reply["target_tick"]
        self.frame_count = reply["frame"]
        self.frame_times.append(reply.get("feedback_monotonic_s", time.monotonic()))
        if reply["feedback_fresh"]:
            self.readings = {info["id"]: info for info in reply["servo_status"]}
            self.last_feedback = time.monotonic()
            for sid, info in self.readings.items():
                self.samples.append({
                    "time_unix_s": timestamp, "mode": "实机", "source": getattr(self, "vision_source", "外部视觉 TCP"),
                    "frame": reply["frame"], "id": sid, "target_tick": self.targets[sid],
                    **{key: info.get(key, "") for key in (
                        "position_tick", "speed_rpm", "motor_current_ma", "voltage_v", "temperature_c", "status")},
                    "joints_deg": json.dumps(angles, ensure_ascii=False),
                })
        self.stream_info.set(f"第 {reply['frame']} 帧 · {reply['process_ms']:.1f} ms")
        self.motion_title.set("视觉联动 · 正在接收关节参数")
        self.motion_note.set(getattr(self, "vision_source", "外部视觉 TCP") + "；目标数据经过完整帧与行程检查")
        self.update_metrics()
        self.draw_chart()

    def export(self):
        if not self.samples:
            return
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".csv",
                                            initialfile="hand_session.csv",
                                            filetypes=[("实验记录", "*.csv")])
        if path:
            try:
                with open(path, "w", encoding="utf-8-sig", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=list(self.samples[0]))
                    writer.writeheader()
                    writer.writerows(self.samples)
            except OSError as exc:
                messagebox.showerror("导出失败", str(exc), parent=self.root)
                return
            self.message.set(f"已导出 {len(self.samples)} 行反馈记录。")
            self.log("数据", f"导出 {len(self.samples)} 行记录至 {path}")

    def run_jog(self):
        selected = self.table.selection()
        if not selected:
            messagebox.showinfo("单台点动", "请先在舵机反馈表中选中一个 ID。", parent=self.root)
            return
        sid = int(selected[0])
        if not messagebox.askokcancel("确认空载点动", f"ID {sid} 将从当前位置正向转动约 3°。\n"
                                      "确认台架只接这一台舵机，且输出轴已脱离齿轮负载。",
                                      parent=self.root):
            return
        self.start_job(f"正在准备 ID {sid} 的空载点动。",
                       lambda bus: jog(bus, sid, 3, report=self.report, stop_event=self.cancel))

    def run_pose(self):
        if self.busy:
            return
        config = self.config
        try:
            targets = config.targets(self.angles)
            if not targets:
                raise ValueError("尚未绑定任何舵机。请先打开「关节绑定…」。")
            if not config.calibrated:
                raise ValueError("请先设置关节绑定、确认方向与软限位，再点击「采集绑定零位」。")
            model = integer(config.model_number, 1, 65535, "model_number")
        except ValueError as exc:
            messagebox.showinfo("发送当前姿态", str(exc), parent=self.root)
            return
        ids = ", ".join(str(sid) for sid in targets)
        if not messagebox.askokcancel("发送当前姿态到绑定舵机",
            f"仅向已绑定的 ID：{ids} 发送当前姿态。\n"
            "未绑定关节不发送；到位、停止或异常后关闭扭矩。\n"
            "确认方向、比例与软限位适合实物，关节周围有活动空间。", parent=self.root):
            return

        def move(bus):
            positions = bus.prepare(config.motors, model, stop_event=self.cancel)
            if self.cancel.is_set():
                return
            bus.move(targets, stop_event=self.cancel)
            timeout_s = pose_timeout_seconds(config.motors, positions, targets)
            deadline = time.monotonic() + timeout_s
            readings = {}
            read_status = getattr(bus, "motion_status", bus.status)

            def timeout():
                self.events.put(("notice", "到位诊断：" + json.dumps({"initial": positions,
                    "targets": targets, "timeout_s": timeout_s, "last_feedback": readings}, ensure_ascii=False)))
                return RuntimeError(pose_timeout_message(positions, targets, readings, timeout_s))

            settled = 0
            while time.monotonic() < deadline and not self.cancel.is_set():
                arrived = True
                for sid in targets:
                    if self.cancel.is_set():
                        return
                    if time.monotonic() >= deadline:
                        raise timeout()
                    info = read_status(sid)
                    readings[sid] = info
                    self.report(info)
                    bus.check_health(info)
                    integer(info["position_tick"], *bus.limits[sid], f"ID {sid} 反馈位置")
                    if "torque_enabled" in info and info["torque_enabled"] != 1:
                        raise RuntimeError(pose_timeout_message(positions, targets, readings, timeout_s,
                            reason="运行中扭矩已关闭，不自动重新使能"))
                    if "goal_tick" in info and info["goal_tick"] != targets[sid]:
                        raise RuntimeError(pose_timeout_message(positions, targets, readings, timeout_s,
                            reason="指令目标未保持"))
                    arrived = arrived and abs(info["position_tick"] - targets[sid]) <= 6 and abs(info.get("speed_rpm", 0)) <= 1
                settled = settled + 1 if arrived else 0
                if settled >= 3:
                    return
                self.cancel.wait(.05)
            if not self.cancel.is_set():
                raise timeout()

        self.start_job("正在发送当前姿态，仅控制已绑定舵机。", move)

    def poll(self):
        self.poll_after = None
        changed = False
        while True:
            try:
                kind, result = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "reading":
                self.readings[result["id"]] = result
                self.last_feedback = time.monotonic()
                self.frame_count += 1
                self.frame_times.append(self.last_feedback)
                sid = result["id"]
                self.samples.append({"time_unix_s": time.time(), "mode": "实机", "source": "实机读取",
                                     "frame": self.frame_count, "id": sid, "target_tick": self.targets.get(sid, result["position_tick"]),
                                     **{key: result.get(key, "") for key in ("position_tick", "speed_rpm", "motor_current_ma", "voltage_v", "temperature_c", "status")},
                                     "joints_deg": json.dumps(self.angles, ensure_ascii=False)})
                changed = True
            elif kind == "frame":
                self.record_frame(result)
                changed = True
            elif kind == "notice":
                self.message.set(result)
                self.log("联动", result)
            else:
                self.busy = False
                success = self.job_success
                self.job_success = None
                self.refresh_controls()
                if kind == "error":
                    self.operation_error("本次操作未完成", result)
                    self.motion_title.set("本次操作未完成 · 查看运行日志")
                else:
                    stopped, zero_readings = result
                    if not stopped and success is not None:
                        try:
                            success(zero_readings)
                        except Exception as exc:
                            self.operation_error("应用操作结果失败", exc)
                            self.motion_title.set("应用操作结果失败 · 查看运行日志")
                            continue
                    elif not stopped and zero_readings is not None:
                        self.zero_readings = zero_readings
                    self.message.set("操作已停止。" if stopped else "操作完成 · 反馈栏保留最后一轮读数。")
                    self.motion_title.set("操作已停止" if stopped else "操作完成 · 可查看曲线与导出数据")
                    self.log("结果", self.message.get())
        if changed:
            self.refresh_table()
        if self.closing and not self.busy:
            self.destroy_root()
        else:
            self.poll_after = self.root.after(50, self.poll)

    def stop(self):
        if self.busy:
            self.cancel.set()
            self.message.set("正在结束操作并完成卸载或配置回锁。")
        self.refresh_controls()

    def close(self):
        if self.closed:
            return
        self.stop()
        if self.busy:
            self.closing = True
            self.stop()
        else:
            self.destroy_root()

    def destroy_root(self):
        if self.closed:
            return
        dialog = getattr(self, "servo_setup_dialog", None)
        if dialog is not None and dialog.window.winfo_exists():
            dialog.close()
        editor = getattr(self, "binding_editor", None)
        if editor is not None and editor.window.winfo_exists():
            editor.window.destroy()
        for name in ("poll_after",):
            handle = getattr(self, name, None)
            if handle is not None:
                self.root.after_cancel(handle)
                setattr(self, name, None)
        self.closed = True
        if self.embedded:
            self.container.destroy()
        else:
            self.root.destroy()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="实机控制工作台")
    parser.parse_args()
    root = tk.Tk()
    Console(root)
    root.mainloop()


if __name__ == "__main__":
    main()

