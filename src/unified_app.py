"""单窗口视觉识别 + HLS 舵机控制，进程内自动转发关节角度。"""
import argparse
import logging
import os
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
import yaml

from .control_bridge import CONTROL_DIR, SUPPORTED_JOINTS, VisionControlBridge, extract_joint_features
from .hand_tracking.mediapipe_tracker import HandTracker
from .hand_mapper.joint_mapper import JointMapper
from .hand_mapper.gestures import GestureRecognizer, finger_state
from .hand_mapper.orientation import OrientationDetector
from .hand_mapper.motion_gestures import MotionRecognizer
from .ui.display import HandDisplay
from hand_dashboard import BG, PANEL, INPUT, TEXT, MUTED, CYAN, AXES, FINGERS
from hand_panel import EmbeddedControlPanel

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


class UnifiedDisplay(HandDisplay):
    """复用原视觉绘图，不显示旧的模拟串口提示或虚构舵机反馈。"""

    def _build_status_bar(self, total_width):
        bar = np.full((40, total_width, 3), (28, 28, 28), dtype=np.uint8)
        message = f"Vision {self._fps:.1f} FPS | " + ("Hand detected" if self._landmarks is not None else "No hand detected")
        cv2.putText(bar, message, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (100, 225, 190), 1)
        cv2.putText(bar, "Motor status: control panel", (max(330, total_width - 310), 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (185, 185, 185), 1)
        return bar

    def _build_dashboard_panel(self):
        angles = self._angles.to_dict() if self._angles else {}
        return np.vstack([self._build_gesture_strip(), self._virtual_hand.render(angles)])


class VisionWorker:
    """唯一摄像头/推理线程；不创建窗口、不启动旧的 PWM/Arduino 串口。"""

    def __init__(self, config, bridge):
        self.config, self.bridge = config, bridge
        self.cancel = threading.Event()
        self.thread = None
        self._lock = threading.Lock()
        self._sequence = 0
        self._snapshot = {"sequence": 0, "image": None, "message": "等待启动摄像头"}

    def snapshot(self):
        with self._lock:
            return self._snapshot.copy()

    def _publish(self, message, image=None):
        with self._lock:
            self._sequence += 1
            self._snapshot = {"sequence": self._sequence, "image": image, "message": message}

    def start(self, camera_index):
        if self.is_running:
            raise RuntimeError("摄像头正在运行或关闭中")
        self.cancel.clear()
        self.thread = threading.Thread(target=self._run, args=(camera_index,), name="vision-inference")
        self.thread.start()

    @property
    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    def stop(self):
        self.cancel.set()

    def _run(self, camera_index):
        camera = tracker = None
        failed = False
        try:
            self._publish("正在加载视觉模型，请稍候…")
            mp = self.config.get("mediapipe", {})
            tracker = HandTracker(static_image_mode=mp.get("static_image_mode", False),
                max_num_hands=1, min_detection_confidence=mp.get("min_detection_confidence", .7),
                min_tracking_confidence=mp.get("min_tracking_confidence", .5))
            tracker.start()
            if self.cancel.is_set():
                return
            camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
            if not camera.isOpened() and os.name == "nt":
                camera.release()
                camera = cv2.VideoCapture(camera_index)
            if not camera.isOpened():
                raise RuntimeError(f"无法打开摄像头 {camera_index}，请关闭占用摄像头的程序后重试")
            cam = self.config.get("camera", {})
            width, height = int(cam.get("width", 640)), int(cam.get("height", 480))
            fps = max(1, min(60, int(cam.get("fps", 30))))
            for prop, value in ((cv2.CAP_PROP_FRAME_WIDTH, width), (cv2.CAP_PROP_FRAME_HEIGHT, height),
                                (cv2.CAP_PROP_FPS, fps), (cv2.CAP_PROP_BUFFERSIZE, 1)):
                camera.set(prop, value)
            smooth = self.config.get("servos", {}).get("smoothing", {})
            mapper = JointMapper(smoothing_enabled=smooth.get("enabled", True),
                smoothing_alpha=smooth.get("alpha", .3), max_step=smooth.get("max_step", 5),
                forced_handedness=self.config.get("orientation", {}).get("forced_handedness") or None)
            ocfg = self.config.get("orientation", {})
            gesture, motion = GestureRecognizer(), MotionRecognizer()
            orientation = OrientationDetector(forced_handedness=ocfg.get("forced_handedness") or None,
                invert_front_back=ocfg.get("invert_front_back", False))
            display = UnifiedDisplay()
            previous_hand = None
            last_render = 0
            while not self.cancel.is_set():
                started = time.monotonic()
                ok, frame = camera.read()
                captured_at = time.monotonic()
                if not ok:
                    self.bridge.tracking_lost("摄像头读取失败，关节跟随已停止")
                    self._publish("摄像头读取失败，等待恢复或切换设备")
                    self.cancel.wait(.05)
                    continue
                frame = cv2.resize(frame, (width, height))
                if cam.get("flip_horizontal", False):
                    frame = cv2.flip(frame, 1)
                landmarks = tracker.process(frame)
                if landmarks is not None and landmarks.handedness != previous_hand:
                    mapper.reset_smoothing()
                    previous_hand = landmarks.handedness
                angles = mapper.compute_angles(landmarks) if landmarks is not None else None
                recognized = gesture.update(mapper.instant_angles if angles is not None else None, landmarks)
                facing = orientation.update(landmarks)
                moving = motion.update(landmarks)
                try:
                    features = extract_joint_features(angles, landmarks)
                    self.bridge.publish(features, landmarks.handedness if landmarks is not None else "", captured_at)
                except (ValueError, TypeError, KeyError) as exc:
                    self.bridge.tracking_lost("无效视觉帧：" + str(exc))
                if started - last_render >= .1:
                    states = {name: finger_state(angles, name)
                              for name in ("thumb", "index", "middle", "ring", "pinky")} if angles else {}
                    composite = display.update(frame, landmarks, angles, fps=tracker.fps,
                        gesture_name=recognized.label or None, finger_states=states)
                    hand = landmarks.handedness if landmarks is not None else "未检测到手"
                    message = (f"{hand} · {tracker.fps:.1f} FPS · 手势 {recognized.label or '—'}"
                               f" · 朝向 {facing.facing} · 运动 {moving.name}")
                    self._publish(message, composite)
                    last_render = started
                self.cancel.wait(max(0, 1 / fps - (time.monotonic() - started)))
        except Exception as exc:
            failed = True
            LOGGER.exception("视觉线程结束")
            self.bridge.tracking_lost("视觉异常：" + str(exc))
            self._publish("视觉异常：" + str(exc))
        finally:
            try:
                if camera is not None:
                    camera.release()
            finally:
                try:
                    if tracker is not None:
                        tracker.stop()
                finally:
                    self.bridge.tracking_lost("视觉识别已停止")
                    if not failed:
                        self._publish("视觉识别已停止（保留最后画面）", self.snapshot()["image"])


class UnifiedApplication:
    def __init__(self, root, config, *, bindings=None, port="", baud=1000000, auto_start=True,
                 home_path=CONTROL_DIR.parent / "default_home.json"):
        self.root, self.closing = root, False
        self.after_id = None
        self.image = None
        self.last_image_sequence = -1
        self.bound_config = None
        self.root.title("视觉识别与灵巧手控制")
        self.root.configure(bg=BG)
        width, height = min(1400, root.winfo_screenwidth() - 32), min(850, root.winfo_screenheight() - 65)
        self.root.geometry(f"{width}x{height}+12+12")
        self.root.minsize(1120, 740)
        if os.name == "nt" and auto_start:
            self.root.state("zoomed")
        header = tk.Frame(root, bg=BG, padx=16, pady=12)
        header.pack(fill="x")
        brand = tk.Frame(header, bg=BG)
        brand.pack(side="left")
        tk.Label(brand, text="灵巧手控制台", bg=BG, fg=TEXT,
                 font=("Microsoft YaHei UI", 16, "bold")).pack(anchor="w")
        tk.Label(brand, text="视觉识别  /  关节控制  /  实机反馈", bg=BG, fg=MUTED,
                 font=("Microsoft YaHei UI", 8)).pack(anchor="w", pady=(3, 0))
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)
        self.workspace_page = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(self.workspace_page, text="视觉与舵机控制")
        self.control_page = self.workspace_page
        self.panel = EmbeddedControlPanel(self.control_page, config=bindings, home_path=home_path,
                                          integrated_workspace=True)
        self.vision_page = self.panel.vision_host
        self.panel.port.set(port)
        self.panel.baud.set(str(baud))
        self.panel.vision_source = "内置摄像头识别"
        self.panel.stream_button.configure(command=self.start_follow)
        self.emergency_button = ttk.Button(header, text="■  急停  Esc", style="Stop.TButton",
                                          command=self.panel.emergency_stop)
        self.emergency_button.pack(side="right")
        self.return_button = ttk.Button(header, text="停止并归位", command=self.panel.stop)
        self.return_button.pack(side="right", padx=(6, 12))
        self.control_nav = ttk.Button(header, text="手动调试", style="Quiet.TButton",
                     command=lambda: self.panel.tabs.select(self.panel.manual_view))
        self.control_nav.pack(side="right", padx=3)
        self.vision_nav = ttk.Button(header, text="监视曲线", style="Quiet.TButton",
                     command=lambda: self.panel.tabs.select(1))
        self.vision_nav.pack(side="right", padx=3)
        self.run_state = tk.StringVar(value="仅预览")
        self.run_badge = tk.Label(header, textvariable=self.run_state, bg=PANEL, fg=CYAN,
                                 font=("Microsoft YaHei UI", 9, "bold"), padx=12, pady=6)
        self.run_badge.pack(side="right", padx=12)
        self.bridge = VisionControlBridge()
        self.worker = VisionWorker(config, self.bridge)
        self.camera_index = tk.StringVar(value=str(config.get("camera", {}).get("index", 0)))
        self.gain = tk.StringVar(value="1.0")
        self.vision_status = tk.StringVar(value="等待识别")
        self.bridge_status = tk.StringVar()
        self.home_status = tk.StringVar(value=self.panel.home_activity)
        self.next_step = tk.StringVar(value="先启动识别，再采集视觉零姿态")
        self.settings_open = False
        self.build_vision_page()
        self.sync_config()
        self.workspace_page.bind("<Configure>", self.resize_sections, add="+")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", lambda event: self.panel.escape(event))
        self.root.bind("<F11>", lambda event: self.panel.toggle_fullscreen())
        self.after_id = root.after(50, self.poll)
        if auto_start:
            self.start_after = root.after(250, self.start_camera)
        else:
            self.start_after = None

    def build_vision_page(self):
        page = self.panel.vision_host
        toolbar = tk.Frame(page, bg=PANEL)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        tk.Label(toolbar, text="视觉识别", bg=PANEL, fg=TEXT,
                 font=("Microsoft YaHei UI", 11, "bold")).pack(side="left", padx=(0, 10))
        self.camera_entry = ttk.Entry(toolbar, textvariable=self.camera_index, width=4)
        self.camera_entry.pack(side="left", padx=(0, 5))
        self.camera_start = ttk.Button(toolbar, text="启动识别", command=self.start_camera, style="Small.TButton")
        self.camera_start.pack(side="left", padx=3)
        ttk.Button(toolbar, text="停止识别", command=self.stop_camera, style="Small.TButton").pack(side="left", padx=3)
        self.video_card = tk.Frame(page, bg=BG)
        self.video_card.grid(row=1, column=0, sticky="nsew")
        self.video_label = tk.Label(self.video_card, bg=BG, fg=MUTED,
            text="等待摄像头画面\n点击「启动识别」开始预览", font=("Microsoft YaHei UI", 11), justify="center")
        self.video_label.pack(fill="both", expand=True)
        self.vision_status_label = tk.Label(page, textvariable=self.vision_status, bg=PANEL,
            fg=MUTED, anchor="w", justify="left", font=("Microsoft YaHei UI", 8))
        self.vision_status_label.grid(row=2, column=0, sticky="ew", pady=(5, 7))
        self.overview_card = self.panel.overview_frame
        self.overview_canvas = self.panel.canvas

        side = self.panel.integration_controls
        self.panel.stream_button.pack_forget()
        side.columnconfigure(0, weight=1)
        title = tk.Frame(side, bg=PANEL)
        title.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        tk.Label(title, text="视觉与归位", bg=PANEL, fg=TEXT,
            font=("Microsoft YaHei UI", 10, "bold")).pack(side="left")
        self.settings_button = ttk.Button(title, text="设置 ▾", width=6,
            style="Quiet.TButton", command=self.toggle_settings)
        self.settings_button.pack(side="right")
        self.zero_button = ttk.Button(side, text="采集视觉零姿态", command=self.capture_neutral)
        self.zero_button.grid(row=1, column=0, sticky="ew", pady=3)
        home_row = tk.Frame(side, bg=PANEL)
        home_row.grid(row=2, column=0, sticky="ew", pady=3)
        self.home_button = ttk.Button(home_row, text="设为默认归位姿态", style="Small.TButton", command=self.capture_home)
        self.home_button.pack(side="left", fill="x", expand=True)
        self.clear_home_button = ttk.Button(home_row, text="清除", width=4, style="Small.TButton", command=self.clear_home)
        self.clear_home_button.pack(side="right", padx=(4, 0))
        self.settings_frame = tk.Frame(side, bg=PANEL)
        self.settings_frame.columnconfigure(1, weight=1)
        self.settings_frame.grid(row=3, column=0, sticky="ew", pady=3)
        tk.Label(self.settings_frame, text="比例", bg=PANEL, fg=MUTED).grid(row=0, column=0)
        self.gain_entry = ttk.Entry(self.settings_frame, textvariable=self.gain, width=5)
        self.gain_entry.grid(row=0, column=1, sticky="ew", padx=5)
        self.gain_button = ttk.Button(self.settings_frame, text="应用", width=4, style="Small.TButton", command=self.apply_gain)
        self.gain_button.grid(row=0, column=2)
        self.settings_frame.grid_remove()
        self.open_button = self.panel.config_button
        self.follow_button = self.panel.stream_button
        self.follow_button.configure(text="开始跟随", style="Accent.TButton")
        self.follow_button.grid(row=4, column=0, sticky="ew", pady=(5, 3))
        tk.Label(side, textvariable=self.next_step, bg=PANEL, fg=CYAN, wraplength=190,
            justify="left", anchor="w", font=("Microsoft YaHei UI", 8)).grid(row=5, column=0, sticky="ew", pady=(3, 0))
        tk.Label(side, textvariable=self.home_status, bg=PANEL, fg=MUTED, wraplength=190,
            justify="left", anchor="w", font=("Microsoft YaHei UI", 8)).grid(row=6, column=0, sticky="ew", pady=(3, 0))
        self.vision_detail_tab = tk.Frame(self.panel.tabs, bg=PANEL, padx=10, pady=8)
        self.panel.tabs.add(self.vision_detail_tab, text="  识别角度详情  ")
        self.joint_table = self.panel.make_table(self.vision_detail_tab,
            (("joint", "关节", 220), ("target", "目标 °", 110), ("status", "状态", 140)), height=4)


    def toggle_settings(self):
        self.settings_open = not self.settings_open
        if self.settings_open:
            self.settings_frame.grid()
        else:
            self.settings_frame.grid_remove()
        self.settings_button.configure(text="收起 ▴" if self.settings_open else "设置 ▾")
        self.resize_sections()

    def sync_config(self):
        if self.panel.config is not self.bound_config:
            self.bound_config = self.panel.config
            self.bridge.configure(self.panel.config, self.panel.joint_input)
            self.joint_table.delete(*self.joint_table.get_children())
            for joint in self.panel.config.joints:
                finger, _, axis = joint.partition("_")
                self.joint_table.insert("", "end", iid=joint,
                    values=(FINGERS.get(finger, finger) + " " + AXES.get(axis, axis), "—", "等待"))

    def resize_sections(self, event=None):
        if self.panel.closed or not self.vision_status_label.winfo_exists():
            return
        self.vision_status_label.configure(wraplength=max(240, self.vision_page.winfo_width() - 24))

    def sync_joint_overview(self, state):
        # Preview is independent of transport acceptance. Even unbound joints
        # or a rejected motor target must remain visible without sending it.
        targets = state["targets"]
        if (self.closing or self.panel.closed or targets is None
                or targets == self.panel.angles):
            return
        self.panel.display_angles(targets)
        if not self.panel.busy:
            self.panel.local_source = "视觉目标预览"
            self.panel.refresh()

    def start_camera(self):
        self.start_after = None
        if self.closing or self.worker.is_running:
            return
        try:
            index = int(self.camera_index.get())
            if index < 0:
                raise ValueError("摄像头编号不能为负数")
            self.worker.start(index)
        except ValueError as exc:
            messagebox.showerror("摄像头设置", str(exc), parent=self.root)

    def stop_camera(self):
        self.panel.stop()
        self.worker.stop()
        self.bridge.tracking_lost("视觉识别已停止")

    def capture_neutral(self):
        if self.panel.busy:
            return
        self.sync_config()
        try:
            self.bridge.capture_neutral()
        except ValueError as exc:
            messagebox.showinfo("采集视觉零姿态", str(exc), parent=self.root)

    def apply_gain(self):
        if self.panel.busy:
            return
        try:
            self.bridge.set_gain(float(self.gain.get()))
        except ValueError as exc:
            messagebox.showerror("视觉比例", str(exc), parent=self.root)

    def capture_home(self):
        if self.panel.busy or self.closing:
            return
        self.sync_config()
        state = self.bridge.status()
        try:
            stamp = state["captured_at"]
            if (not state["neutral_ready"] or state["targets"] is None or state["features"] is None
                    or stamp is None or time.monotonic() - stamp > self.panel.joint_input.max_age):
                raise ValueError("请先采集视觉零姿态，再把手指伸直并保持在镜头前，等待有效识别")
            if state["unsupported"]:
                raise ValueError("部分绑定尚无视觉参数，请先调整绑定")
            self.panel.set_home_pose(state["targets"], visual_features=state["features"])
        except (ValueError, OSError, KeyError, TypeError) as exc:
            messagebox.showinfo("设置默认归位姿态", str(exc), parent=self.root)
        self.home_status.set("默认归位已设置" if self.panel.home_pose is not None else "默认归位：未设置")

    def clear_home(self):
        try:
            self.panel.clear_home_pose()
        except OSError as exc:
            messagebox.showerror("清除归位设置", str(exc), parent=self.root)
        self.home_status.set("默认归位已设置" if self.panel.home_pose is not None else "默认归位：未设置")

    def start_follow(self):
        self.sync_config()
        state = self.bridge.status()
        if state["unsupported"]:
            messagebox.showinfo("部分关节暂无视觉参数",
                "下列绑定尚无可靠视觉参数，无法开始跟随：\n" + "\n".join(state["unsupported"])
                + "\n请在关节绑定中只保留当前可识别的屈曲关节。", parent=self.root)
        elif not state["neutral_ready"]:
            messagebox.showinfo("开始跟随", "请先启动识别并采集视觉零姿态。", parent=self.root)
        elif self.worker.is_running and state["features"] is not None:
            self.panel.start_stream()

    def poll(self):
        self.after_id = None
        # Camera/settings now belong to the same workbench container. Once its
        # close handler removes it, wait for inference without touching widgets.
        if self.panel.closed:
            if self.closing and not self.worker.is_running:
                self.root.destroy()
            elif self.closing:
                self.after_id = self.root.after(50, self.poll)
            return
        self.sync_config()
        snapshot = self.worker.snapshot()
        self.vision_status.set(snapshot["message"])
        if snapshot["image"] is None:
            empty = snapshot["message"]
            if not self.worker.is_running and not self.closing:
                empty += "\n点击上方「启动识别」查看摄像头画面"
            self.video_label.configure(image="", text=empty)
            self.image = None
        if snapshot["sequence"] != self.last_image_sequence and snapshot["image"] is not None:
            self.last_image_sequence = snapshot["sequence"]
            image = Image.fromarray(cv2.cvtColor(snapshot["image"], cv2.COLOR_BGR2RGB))
            width = max(320, self.video_label.winfo_width() - 8)
            height = max(240, self.video_label.winfo_height() - 8)
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            self.image = ImageTk.PhotoImage(image, master=self.root)
            self.video_label.configure(image=self.image, text="")
        state = self.bridge.status()
        self.sync_joint_overview(state)
        self.bridge_status.set(state["message"])
        if self.panel.home_failed:
            self.home_status.set("归位异常，请查看运行日志")
        elif self.panel.home_reached and not self.panel.busy:
            self.home_status.set("已归位并卸载")
        else:
            self.home_status.set("默认归位已设置" if self.panel.home_pose is not None else "默认归位：未设置")
        for joint in self.panel.config.joints:
            targets = state["targets"]
            value = f"{targets[joint]:.1f}" if targets is not None and joint in SUPPORTED_JOINTS else "—"
            note = "无视觉参数" if joint not in SUPPORTED_JOINTS else "限幅" if joint in state["limited"] else "已接入" if targets else "等待"
            current = self.joint_table.item(joint, "values")
            self.joint_table.item(joint, values=(current[0], value, note))
        idle = not self.panel.busy and not self.closing
        for widget in (self.open_button, self.zero_button, self.gain_entry, self.gain_button,
                       self.home_button, self.clear_home_button):
            widget.configure(state="normal" if idle else "disabled")
        ready = (idle and self.worker.is_running and state["features"] is not None
                 and state["neutral_ready"] and not state["unsupported"]
                 and self.panel.config.calibrated and self.panel.config.motors)
        self.follow_button.configure(state="normal" if ready else "disabled")
        if self.closing:
            label, hint = "正在关闭", "等待归位与卸载完成"
        elif self.panel.busy:
            homing = "归位" in self.panel.motion_title.get()
            label, hint = ("自动归位", "正在返回默认姿态，Esc 可立即中止") if homing else ("舵机运行中", "可停止并归位，或按 Esc 急停")
        elif not self.worker.is_running:
            label, hint = "仅预览", "先点击「启动识别」"
        elif not self.panel.config.calibrated:
            label, hint = "待标定", "先在下方绑定舵机并采集绑定零位"
        elif not state["neutral_ready"]:
            label, hint = "识别预览", "下一步：采集视觉零姿态"
        elif state["unsupported"]:
            label, hint = "识别预览", "部分绑定暂无视觉角度，请检查关节绑定"
        elif state["features"] is None:
            label, hint = "等待人手", "把手放到镜头前，等待有效识别"
        elif self.panel.joint_input.snapshot() is None:
            label, hint = "检查目标", state["message"]
        else:
            label = "准备就绪"
            hint = "可开始跟随 · 尚未设置默认归位" if self.panel.home_pose is None else "可开始跟随 · 已启用默认归位"
        self.run_state.set(label)
        self.next_step.set(hint)
        self.camera_entry.configure(state="disabled" if self.worker.is_running else "normal")
        self.camera_start.configure(state="disabled" if self.worker.is_running or self.closing else "normal")
        if self.closing and self.panel.closed and not self.worker.is_running:
            self.root.destroy()
        else:
            self.after_id = self.root.after(50, self.poll)

    def close(self):
        if self.closing:
            return
        self.closing = True
        if self.start_after is not None:
            self.root.after_cancel(self.start_after)
            self.start_after = None
        self.stop_camera()
        self.panel.close()


def main():
    parser = argparse.ArgumentParser(description="视觉识别与灵巧手控制统一程序")
    parser.add_argument("--camera", type=int)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "hand_config.yaml", help="视觉 YAML 配置")
    parser.add_argument("--bindings", type=Path, help="已标定的舵机绑定 JSON")
    parser.add_argument("--port", default="", help="FE-URT-2 串口，也可启动后选择")
    parser.add_argument("--baud", type=int, default=1000000)
    args = parser.parse_args()
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "unified_hand.log", encoding="utf-8")])
    with args.config.open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    if args.camera is not None:
        config.setdefault("camera", {})["index"] = args.camera
    bindings = args.bindings
    local_bindings = CONTROL_DIR.parent / "joint_bindings.json"
    if bindings is None and local_bindings.is_file():
        bindings = local_bindings
    root = tk.Tk()
    app = UnifiedApplication(root, config, bindings=bindings, port=args.port, baud=args.baud)
    try:
        root.mainloop()
    finally:
        app.worker.stop()
        app.panel.cancel.set()
        for thread in (app.worker.thread, getattr(app.panel, "worker", None)):
            if thread is not None:
                thread.join()
        try:
            root.destroy()
        except tk.TclError:
            pass


if __name__ == "__main__":
    main()
