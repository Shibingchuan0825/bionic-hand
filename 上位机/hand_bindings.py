"""Editable, explicit joint-to-servo bindings. Opening this dialog never uses serial."""
from copy import deepcopy
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from hand_control import HandConfig, TICKS_PER_DEGREE
from hand_dashboard import BG, PANEL, MUTED, CYAN, FINGERS, AXES


def joint_label(name):
    finger, _, axis = name.partition("_")
    return FINGERS.get(finger, finger) + " / " + AXES.get(axis, axis)


class BindingEditor:
    def __init__(self, app, joint=None):
        self.app = app
        self.data = app.config.to_data()
        self.data["allow_unbound_joints"] = True
        self.window = tk.Toplevel(app.root)
        self.window.title("关节与舵机绑定")
        self.window.configure(bg=BG)
        self.window.geometry("940x650")
        self.window.minsize(830, 590)
        self.window.transient(app.root)
        self.window.grab_set()
        body = tk.Frame(self.window, bg=PANEL, padx=14, pady=12)
        body.pack(fill="both", expand=True)
        app.label(body, "关节与舵机绑定", size=15, bold=True).pack(anchor="w")
        note = app.label(body, "单台测试：先清空模板绑定，再选择关节和舵机 ID 添加。未出现在表中的关节不发送。\n"
                         "同一关节可绑定多台舵机；同一 ID 的多行贡献会相加，支持 MCP / CMC 差动。", color=MUTED, size=9)
        note.pack(fill="x", pady=(5, 8))
        toolbar = tk.Frame(body, bg=PANEL)
        toolbar.pack(fill="x", pady=(0, 6))
        ttk.Button(toolbar, text="清空全部绑定", command=self.clear).pack(side="left")
        ttk.Button(toolbar, text="删除选中绑定", command=self.remove).pack(side="left", padx=6)
        self.summary = tk.StringVar()
        app.label(toolbar, var=self.summary, color=CYAN, size=9).pack(side="right")
        self.table = app.make_table(body, (
            ("joint", "关节", 170), ("id", "舵机 ID", 65), ("weight", "比例 / 方向", 95),
            ("zero", "零位编码", 85), ("limits", "舵机软限位 °", 145),
            ("current", "限流 mA", 85), ("speed", "限速 RPM", 85)), height=7)
        self.table.bind("<<TreeviewSelect>>", self.select)
        form = tk.Frame(body, bg=PANEL)
        form.pack(fill="x", pady=(10, 3))
        self.joints = {joint_label(name): name for name in self.data["joint_limits_deg"]}
        self.fields = {key: tk.StringVar(value=value) for key, value in (
            ("joint", joint_label(joint or next(iter(app.config.joints)))), ("id", "1"),
            ("weight", "1"), ("zero", "2048"), ("low", "-5"), ("high", "5"),
            ("current", "250"), ("speed", "5"))}
        for column, (key, text) in enumerate((("joint", "关节"), ("id", "舵机 ID"), ("weight", "比例 / 方向"))):
            app.label(form, text, color=MUTED, size=9).grid(row=0, column=column, sticky="w", padx=(0, 8))
            if key == "joint":
                widget = ttk.Combobox(form, textvariable=self.fields[key], values=tuple(self.joints), state="readonly", width=26)
            else:
                widget = ttk.Entry(form, textvariable=self.fields[key], width=12)
            widget.grid(row=1, column=column, sticky="ew", padx=(0, 8), pady=(2, 7))
            form.columnconfigure(column, weight=1)
        settings = tk.Frame(body, bg=PANEL)
        settings.pack(fill="x", pady=(0, 5))
        for column, (key, text) in enumerate((("zero", "零位编码"), ("low", "软下限 °"), ("high", "软上限 °"),
                                             ("current", "限流 mA"), ("speed", "限速 RPM"))):
            app.label(settings, text, color=MUTED, size=9).grid(row=0, column=column, sticky="w", padx=(0, 8))
            ttk.Entry(settings, textvariable=self.fields[key], width=11).grid(row=1, column=column, sticky="ew", padx=(0, 8), pady=3)
            settings.columnconfigure(column, weight=1)
        app.label(body, "舵机转角 = 关节角度 × 比例；负比例反向。软限位相对舵机零位，新增默认 ±5°。\n"
                  "多圈零位可为负数或超过 4095；采集时核验模式，软限位仍相对零位生效。\n"
                  "同一 ID 共用零位、限流与限速。修改绑定后需重新采集绑定零位；不会自动发送动作。",
                  color=MUTED, size=9).pack(anchor="w", pady=(2, 7))
        buttons = tk.Frame(body, bg=PANEL)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="添加 / 更新这条绑定", command=self.upsert).pack(side="left")
        ttk.Button(buttons, text="应用绑定", command=self.apply, style="Accent.TButton").pack(side="right")
        ttk.Button(buttons, text="保存并应用…", command=self.save).pack(side="right", padx=8)
        ttk.Button(buttons, text="取消", command=self.window.destroy).pack(side="right")
        self.render()

    def render(self):
        self.table.delete(*self.table.get_children())
        bound = set()
        for motor in self.data["motors"]:
            for name, weight in motor["weights"].items():
                bound.add(name)
                low = (motor["min_tick"] - motor["zero_tick"]) / TICKS_PER_DEGREE
                high = (motor["max_tick"] - motor["zero_tick"]) / TICKS_PER_DEGREE
                self.table.insert("", "end", iid=f"{motor['id']}:{name}", values=(
                    joint_label(name), motor["id"], f"{weight:+g}", motor["zero_tick"],
                    f"{low:+.2f} ~ {high:+.2f}", motor["current_ma"], motor["speed_rpm"]))
        self.summary.set(f"{len(bound)} 个关节 / {len(self.data['motors'])} 台舵机")

    def select(self, _event=None):
        selection = self.table.selection()
        if not selection:
            return
        sid, name = selection[0].split(":", 1)
        motor = next(m for m in self.data["motors"] if m["id"] == int(sid))
        values = {"joint": joint_label(name), "id": sid, "weight": motor["weights"][name],
                  "zero": motor["zero_tick"], "current": motor["current_ma"], "speed": motor["speed_rpm"],
                  "low": (motor["min_tick"] - motor["zero_tick"]) / TICKS_PER_DEGREE,
                  "high": (motor["max_tick"] - motor["zero_tick"]) / TICKS_PER_DEGREE}
        for key, value in values.items():
            self.fields[key].set(f"{value:.8g}" if isinstance(value, float) else str(value))

    def commit(self, candidate):
        candidate["calibrated"] = False
        candidate["allow_unbound_joints"] = True
        HandConfig.from_data(candidate)
        self.data = candidate
        self.render()

    def clear(self):
        candidate = deepcopy(self.data)
        candidate["motors"] = []
        self.commit(candidate)

    def remove(self):
        selection = self.table.selection()
        if not selection:
            return
        sid, name = selection[0].split(":", 1)
        candidate = deepcopy(self.data)
        for motor in candidate["motors"]:
            if motor["id"] == int(sid):
                motor["weights"].pop(name)
        candidate["motors"] = [m for m in candidate["motors"] if m["weights"]]
        try:
            self.commit(candidate)
        except ValueError as exc:
            messagebox.showerror("绑定无效", str(exc), parent=self.window)

    def upsert(self):
        try:
            values = {key: var.get().strip() for key, var in self.fields.items()}
            sid, zero = int(values["id"]), int(values["zero"])
            weight = float(values["weight"])
            low, high = float(values["low"]), float(values["high"])
            candidate = deepcopy(self.data)
            motor = next((m for m in candidate["motors"] if m["id"] == sid), None)
            if motor is None:
                motor = {"id": sid, "weights": {}, "allow_multiturn": False}
                candidate["motors"].append(motor)
            motor.update(zero_tick=zero, min_tick=zero + round(low * TICKS_PER_DEGREE),
                         max_tick=zero + round(high * TICKS_PER_DEGREE),
                         current_ma=float(values["current"]), speed_rpm=float(values["speed"]))
            motor["weights"][self.joints[values["joint"]]] = weight
            self.commit(candidate)
        except (ValueError, KeyError, OverflowError) as exc:
            messagebox.showerror("无法添加绑定", str(exc), parent=self.window)

    def apply(self):
        try:
            self.app.apply_binding_data(self.data)
        except (ValueError, KeyError, TypeError) as exc:
            messagebox.showerror("绑定无效", str(exc), parent=self.window)
            return
        self.window.destroy()

    def save(self):
        path = filedialog.asksaveasfilename(parent=self.window, initialfile="joint_bindings.json",
                                          defaultextension=".json", filetypes=[("关节绑定配置", "*.json")])
        if not path:
            return
        try:
            config = HandConfig.from_data(self.data)
            Path(path).write_text(json.dumps(config.to_data(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.app.apply_binding_data(config.to_data(), path=path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            messagebox.showerror("保存绑定失败", str(exc), parent=self.window)
            return
        self.window.destroy()
