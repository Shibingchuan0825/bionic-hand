"""把现有控制工作台放进宿主 Tk 界面，接收识别线程传入的关节参数。"""
import time
import threading
from tkinter import messagebox

from hand_control import HandConfig, integer
from hand_gui import Console
from hand_integration import JointInput, stream_joint_input
from hand_home import HomePose


class EmbeddedControlPanel(Console):
    def __init__(self, parent, *, config=None, home_path=None, integrated_workspace=False):
        self.integrated_workspace = integrated_workspace
        self.joint_input = None
        self.home_path = home_path
        self.home_pose = None
        self.home_abort = threading.Event()
        self.home_activity = "未设置默认归位姿态"
        self.home_reached = self.home_failed = False
        self.preview_sequence = 0
        self.vision_source = "内嵌视觉参数"
        super().__init__(parent.winfo_toplevel(), parent=parent)
        self.stream_endpoint.configure(text="进程内输入 / 关节角度 °")
        self.stream_button.configure(text="开始关节跟随")
        self.stop_button.configure(text="停止并归位")
        self.stream_info.set("等待视觉端传入关节参数")
        self.container.bind("<Destroy>", self._container_destroyed, add="+")
        if config is not None:
            if isinstance(config, HandConfig):
                self.apply_binding_data(config.to_data())
            else:
                self.load_profile(config)

    def load_profile(self, path=None):
        if self.busy:
            return
        previous = self.config
        super().load_profile(path)
        if self.config is not previous:
            self.joint_input = JointInput(self.config)
            self.preview_sequence = 0
            self.home_pose = None
            self.home_activity = "未设置默认归位姿态"
            if self.home_path is not None:
                try:
                    self.home_pose = HomePose.load(self.config, self.home_path)
                except FileNotFoundError:
                    pass
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    self.home_activity = "归位不可用：" + str(exc)
                else:
                    self.home_activity = "默认归位已加载；跟随中丢手、停止或关闭后自动归位"

    def set_home_pose(self, angles, *, visual_features=None):
        if self.busy or self.closing:
            raise ValueError("请先结束当前舵机操作，再采集归位姿态")
        home = HomePose(self.config, angles, visual_features=visual_features)
        if self.home_path is not None:
            home.save(self.home_path)
        self.home_pose = home
        self.home_activity = "伸直归位姿态已保存；跟随中丢手、停止或关闭后自动归位"
        self.log("归位", self.home_activity)

    def clear_home_pose(self):
        if self.busy or self.closing:
            return
        if self.home_path is not None:
            from pathlib import Path
            Path(self.home_path).unlink(missing_ok=True)
        self.home_pose = None
        self.home_activity = "自动归位已取消；请重新采集伸直姿态以启用"

    def submit_joints(self, angles, *, captured_at=None):
        """识别线程只调用此接口；界面更新和串口发送由控制模块处理。"""
        if self.closed:
            raise RuntimeError("控制面板已关闭")
        return self.joint_input.submit_joints(angles, captured_at=captured_at)

    def tracking_lost(self, reason="视觉跟踪丢失"):
        self.joint_input.invalidate(reason)

    def start_stream(self):
        if self.busy:
            return
        config, source = self.config, self.joint_input
        try:
            if not config.motors or not config.calibrated:
                raise ValueError("请先完成关节绑定和零位标定")
            integer(config.model_number, 1, 65535, "model_number")
            self.serial_settings()
            source.begin()
            if self.home_pose is not None:
                self.home_pose.targets(config)
        except (ValueError, TimeoutError) as exc:
            messagebox.showinfo("开始关节跟随", str(exc), parent=self.root)
            return
        if not messagebox.askokcancel("确认实机关节跟随",
            "将根据视觉端传入的关节参数驱动已绑定舵机。\n"
            "首帧应接近当前位置。\n"
            + ("已启用默认归位：丢手、停止和正常关闭时先低速归位，再卸载。\n"
               if self.home_pose is not None else "尚未设置自动归位；结束后直接卸载。\n")
            + "Esc 急停、欠压、过热或通信故障直接卸载。\n"
            "确认标定和限位已验证，并可随时断开舵机电源。", parent=self.root):
            return
        try:
            epoch = source.begin()
        except TimeoutError as exc:
            messagebox.showinfo("等待有效视觉参数", str(exc), parent=self.root)
            return

        home = self.home_pose
        self.home_abort.clear()

        def notice(message):
            self.home_activity = message
            self.events.put(("notice", message))

        def stream(bus):
            self.home_reached = stream_joint_input(config, bus, source, stop_event=self.cancel, epoch=epoch,
                on_frame=lambda angles, reply: self.events.put(("frame", (angles, reply, time.time()))),
                home_pose=home, emergency=self.home_abort, notice=notice)

        self.stream_info.set("准备接收内嵌关节参数")
        self.start_job("正在跟随视觉端关节参数，仅控制已绑定舵机。", stream)

    def start_job(self, *args, **kwargs):
        if not self.busy:
            self.home_reached = self.home_failed = False
        return super().start_job(*args, **kwargs)

    def operation_error(self, title, error):
        self.home_failed = True
        if self.home_pose is not None:
            self.home_activity = "本轮异常，已停止归位；请查看错误提示和运行日志"
        super().operation_error(title, error)

    def record_frame(self, payload):
        super().record_frame(payload)
        if payload[1].get("execution_phase") == "homing":
            self.motion_title.set("正在自动归位")
            self.motion_note.set("返回已保存的伸直姿态；到位后关闭扭矩，Esc 可立即中止")

    def emergency_stop(self):
        self.home_abort.set()
        super().stop()

    def escape(self, _event=None):
        self.emergency_stop()
        if self.fullscreen:
            self.toggle_fullscreen()
        return "break"

    def poll(self):
        was_busy = self.busy
        if not self.closing and not self.busy and self.joint_input is not None:
            frame = self.joint_input.snapshot()
            if frame is not None and frame.sequence != self.preview_sequence:
                self.preview_sequence = frame.sequence
                self.display_angles(frame.angles)
                self.refresh()
                self.stream_info.set("已收到关节参数 · 仅预览，未发送")
            elif frame is None:
                self.stream_info.set("等待新鲜的关节参数 · 未发送")
        super().poll()
        if was_busy and not self.busy and not self.home_failed:
            if self.home_reached:
                self.home_activity = "已归位并卸载；再次点击开始才会跟随"
            elif self.home_abort.is_set() and not self.closed:
                self.home_activity = "已急停并卸载；未继续归位"

    def _container_destroyed(self, event):
        if event.widget is self.container:
            self.home_abort.set()
            self.cancel.set()
            self.joint_input.invalidate("宿主已关闭控制面板")
            if self.poll_after is not None:
                self.root.after_cancel(self.poll_after)
                self.poll_after = None
            self.closed = True
