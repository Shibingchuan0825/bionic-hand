"""实机控制台视图：关节目标、绑定、设备配置与实测反馈。"""
import math
import time
import tkinter as tk
from tkinter import ttk

BG = "#0c1422"
PANEL = "#142133"
INPUT = "#1c2d44"
LINE = "#2e435d"
TEXT = "#f0f5fb"
MUTED = "#a0b2c9"
CYAN = "#4ddac3"
BLUE = "#77adff"
ORANGE = "#ffb28f"
FONT = ("Microsoft YaHei UI", 10)
MONO = ("Consolas", 11)

FINGERS = {"index": "食指", "thumb": "拇指", "middle": "中指", "ring": "无名指", "little": "小指"}
AXES = {"mcp_flex": "MCP 屈曲", "mcp_abd": "MCP 侧摆", "pip_flex": "PIP 屈曲",
        "dip_flex": "DIP 屈曲", "cmc_flex": "CMC 屈曲", "cmc_roll": "CMC 旋转", "ip_flex": "IP 屈曲"}


class DashboardView:
    def build(self):
        integrated = getattr(self, "integrated_workspace", False)
        self.wide_joint_debug = integrated
        self.controls = []
        shell = self.container = tk.Frame(self.content_parent, bg=BG, padx=14, pady=12)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, minsize=222)
        shell.columnconfigure(1, weight=1)
        shell.columnconfigure(2, minsize=390 if integrated else 302)
        shell.rowconfigure(2, weight=4)
        shell.rowconfigure(3, weight=2, minsize=170)
        if integrated:
            shell.rowconfigure(2, weight=1, minsize=275)
            shell.rowconfigure(3, weight=0, minsize=195)

        header = tk.Frame(shell, bg=BG)
        header.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 12))
        if not self.embedded:
            mark = tk.Label(header, text="H", font=("Consolas", 20, "bold"), bg=CYAN, fg=BG, padx=10)
            mark.pack(side="left", padx=(0, 12))
        title = tk.Frame(header, bg=BG)
        title.pack(side="left")
        self.label(title, "02  关节与舵机" if self.embedded else "灵巧手控制台", size=14, bold=True).pack(anchor="w")
        self.label(title, "调节关节目标 · 绑定舵机 · 查看实测反馈", size=8, color=MUTED).pack(anchor="w", pady=(3, 0))
        self.stop_button = ttk.Button(header, text="■  停止" if self.embedded else "■  停止  Esc", command=self.stop, style="Stop.TButton")
        self.stop_button.pack(side="right", padx=(8, 0))
        ttk.Button(header, text="全屏" if self.embedded else "全屏  F11", command=self.toggle_fullscreen).pack(side="right", padx=(8, 0))
        self.badge = self.label(header, var=self.state, color=CYAN, bold=True)
        self.badge.pack(side="right", padx=12)
        if integrated:
            header.grid_remove()

        side_panel, side = self.scroll_panel(shell, width=214)
        self.configuration_panel = side_panel
        side_panel.grid(row=1, column=0, rowspan=3, sticky="nsew", padx=(0, 12))
        self.label(side, "配置与连接", bold=True, size=11).pack(anchor="w", pady=(0, 7))
        self.profile_box = ttk.Combobox(side, textvariable=self.profile, values=tuple(self.profiles), state="readonly", width=22)
        self.profile_box.pack(fill="x")
        self.profile_box.bind("<<ComboboxSelected>>", lambda _: self.load_profile())
        self.controls.append((self.profile_box, "readonly"))
        config_button = self.config_button = ttk.Button(side, text="打开标定配置…", command=self.open_profile)
        config_button.pack(fill="x", pady=(6, 0))
        self.controls.append((config_button, "normal"))
        ttk.Separator(side).pack(fill="x", pady=8)
        self.label(side, "设备连接", size=11, bold=True).pack(anchor="w")
        port_row = tk.Frame(side, bg=PANEL)
        port_row.pack(fill="x", pady=(7, 5))
        self.port_entry = ttk.Combobox(port_row, textvariable=self.port, width=11)
        self.port_entry.pack(side="left", fill="x", expand=True)
        self.port_entry.bind("<Button-1>", self.list_ports)
        self.port_refresh = ttk.Button(port_row, text="刷新", command=self.list_ports, width=4)
        self.port_refresh.pack(side="right", padx=(5, 0))
        baud_row = tk.Frame(side, bg=PANEL)
        baud_row.pack(fill="x")
        self.label(baud_row, "波特率", color=MUTED, size=9).pack(side="left")
        self.baud_entry = ttk.Entry(baud_row, textvariable=self.baud, width=12)
        self.baud_entry.pack(side="right")
        self.servo_setup_button = ttk.Button(side, text="新舵机配置 · ID / 波特率…", command=self.open_servo_setup)
        self.servo_setup_button.pack(fill="x", pady=(7, 0))
        self.connection_label = self.label(side, var=self.connection_info, color=MUTED, size=9)
        self.connection_label.configure(wraplength=180, justify="left")
        self.connection_label.pack(fill="x")
        ttk.Separator(side).pack(fill="x", pady=8)
        if integrated:
            self.integration_controls = tk.Frame(side_panel, bg=PANEL, padx=12, pady=10)
            self.integration_controls.grid(row=1, column=0, columnspan=2, sticky="ew")
        else:
            self.label(side, "视觉联动", size=11, bold=True).pack(anchor="w", pady=(0, 7))
        self.stream_button = ttk.Button(self.integration_controls if integrated else side, text="启动视觉联动", command=self.start_stream)
        self.stream_button.pack(fill="x")
        self.stream_endpoint = self.label(side, "127.0.0.1 : 8765  /  JSON", color=MUTED, size=9)
        self.stream_endpoint.pack(anchor="w", pady=(5, 0))
        self.stream_label = self.label(side, var=self.stream_info, color=CYAN, size=9)
        self.stream_label.configure(wraplength=180, justify="left")
        self.stream_label.pack(fill="x", pady=(4, 0))
        if integrated:
            self.stream_endpoint.pack_forget()
            self.stream_label.pack_forget()
        ttk.Separator(side).pack(fill="x", pady=8)
        for attr, text, action in (("home_button", "零位预览", self.home),
                                    ("export_button", "导出本次数据 CSV…", self.export)):
            button = ttk.Button(side, text=text, command=action)
            button.pack(fill="x", pady=(0, 5))
            setattr(self, attr, button)

        metrics = self.metrics_frame = tk.Frame(shell, bg=PANEL if integrated else BG)
        metrics.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(0, 6 if integrated else 10))
        self.metric_vars = {}
        for index, (key, title, hint) in enumerate((
            ("config", "关节 / 执行器", "当前加载的配置"),
            ("frame", "已处理帧", "本次实机操作"),
            ("rate", "数据刷新", "实测反馈刷新频率"),
            ("feedback", "反馈来源", "状态与通信结果分开显示"),
        )):
            metrics.columnconfigure(index, weight=1, uniform="metric")
            card = tk.Frame(metrics, bg=PANEL, padx=10 if integrated else 13, pady=4 if integrated else 8)
            card.grid(row=0, column=index, sticky="nsew", padx=(0 if integrated or index == 0 else 5, 0))
            var = tk.StringVar(value="—")
            self.metric_vars[key] = var
            if integrated:
                self.label(card, title, color=MUTED, size=8).pack(side="left")
                self.label(card, var=var, color=CYAN if index != 3 else BLUE,
                    size=10, bold=True).pack(side="left", padx=(8, 0))
            else:
                self.label(card, title, color=MUTED, size=9).pack(anchor="w")
                self.label(card, var=var, color=CYAN if index != 3 else BLUE,
                    size=15, bold=True).pack(anchor="w", pady=(2, 1))

        overview = self.overview_frame = tk.Frame(shell, bg=PANEL, padx=12, pady=10)
        overview.grid(row=2, column=2 if integrated else 1, sticky="nsew",
                      padx=(0, 0 if integrated else 10), pady=(0, 10))
        overview.columnconfigure(0, weight=1)
        overview.rowconfigure(1, weight=1)
        view_header = tk.Frame(overview, bg=PANEL)
        view_header.grid(row=0, column=0, sticky="ew")
        self.label(view_header, "整手关节总览", size=12, bold=True).pack(side="left")
        self.overview_note = self.label(view_header, "点击卡片选择手指 · 单位 °", color=MUTED, size=9)
        if not integrated:
            self.overview_note.pack(side="right")
        self.canvas = tk.Canvas(overview, bg=PANEL, highlightthickness=0, height=180)
        self.canvas.grid(row=1, column=0, sticky="nsew", pady=(8, 2))
        self.canvas.bind("<Configure>", lambda _: self.draw())
        self.canvas.bind("<Button-1>", self.select_finger_card)
        self.flow_label = self.label(overview, "关节目标 → 绑定映射 → FE-URT-2 → 实测反馈", color=MUTED, size=9)
        self.flow_label.grid(row=2, column=0, sticky="w", pady=(0, 6))
        if integrated:
            self.flow_label.configure(text="视觉映射后的关节目标 · 单位 °", font=("Microsoft YaHei UI", 8))
            self.vision_host = tk.Frame(shell, bg=PANEL, padx=10, pady=10)
            self.vision_host.grid(row=2, column=1, sticky="nsew", padx=(0, 10), pady=(0, 10))
            self.vision_host.columnconfigure(0, weight=1)
            self.vision_host.rowconfigure(1, weight=1)
        stage = tk.Frame(self.vision_host if integrated else overview, bg=INPUT, padx=10, pady=7)
        stage.grid(row=3, column=0, sticky="ew")
        self.label(stage, var=self.motion_title, color=CYAN, size=11, bold=True).pack(anchor="w")
        note = self.label(stage, var=self.motion_note, color=MUTED, size=9)
        note.configure(wraplength=620, justify="left")
        stage.bind("<Configure>", lambda event: note.configure(wraplength=max(160, event.width - 24)))
        note.pack(anchor="w", pady=(2, 0))

        if integrated:
            self.tabs = ttk.Notebook(shell, height=185)
            self.tabs.grid(row=3, column=1, columnspan=2, sticky="nsew")
            self.manual_view = tk.Frame(self.tabs, bg=PANEL)
        command_panel, command = self.scroll_panel(self.manual_view if integrated else shell, width=290)
        if integrated:
            command_panel.pack(fill="both", expand=True)
        else:
            command_panel.grid(row=2, column=2, sticky="nsew", pady=(0, 10))
        command_header = tk.Frame(command, bg=PANEL)
        command_header.pack(fill="x")
        self.label(command_header, "关节调试", size=12, bold=True).pack(side="left")
        self.finger_box = ttk.Combobox(command_header, textvariable=self.finger, width=8, state="readonly")
        self.finger_box.pack(side="right")
        self.finger_box.bind("<<ComboboxSelected>>", lambda _: self.rebuild_sliders())
        self.controls.append((self.finger_box, "readonly"))
        self.sliders = tk.Frame(command, bg=PANEL)
        self.sliders.pack(fill="both", expand=True, pady=(6, 5))
        preset_row = tk.Frame(command, bg=PANEL)
        preset_row.pack(fill="x", pady=(2, 7))
        self.preset_buttons = []
        for i, (key, title) in enumerate((("open", "张开"), ("half", "半握"), ("fist", "握拳"), ("pinch", "对捏"), ("spread", "侧摆"), ("point", "食指伸展"))):
            preset_row.columnconfigure(i % 3, weight=1)
            button = ttk.Button(preset_row, text=title, width=6, command=lambda k=key: self.apply_preset(k), style="Small.TButton")
            button.grid(row=i // 3, column=i % 3, sticky="ew", padx=2, pady=2)
            self.preset_buttons.append(button)
        pose_row = tk.Frame(command, bg=PANEL)
        pose_row.pack(fill="x", pady=(0, 6))
        for text, action in (("保存姿态", self.save_pose), ("载入姿态", self.load_pose)):
            button = ttk.Button(pose_row, text=text, command=action, style="Small.TButton")
            button.pack(side="left", fill="x", expand=True, padx=2)
            self.controls.append((button, "normal"))
        self.caption_label = self.label(command, var=self.caption, color=MUTED, size=9)
        self.caption_label.configure(wraplength=256, justify="left")
        self.caption_label.pack(fill="x")
        # Keep motion and binding actions outside the scrollable slider area.
        actions = tk.Frame(command_panel, bg=PANEL, padx=10, pady=7)
        if integrated:
            command_panel.columnconfigure(2, minsize=270)
            actions.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        else:
            actions.grid(row=1, column=0, columnspan=2, sticky="ew")
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        self.binding_button = ttk.Button(actions, text="关节绑定…", command=self.edit_bindings, style="Small.TButton")
        self.binding_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.binding_zero_button = ttk.Button(actions, text="采集绑定零位…", command=self.capture_binding_zero, style="Small.TButton")
        self.binding_zero_button.grid(row=0, column=1, sticky="ew", padx=(3, 0))
        self.binding_save_button = ttk.Button(actions, text="保存绑定配置…", command=self.save_bindings, style="Small.TButton")
        self.binding_save_button.grid(row=1, column=0, sticky="ew", padx=(0, 3), pady=4)
        self.jog_button = ttk.Button(actions, text="单台点动…", command=self.run_jog, style="Small.TButton")
        self.jog_button.grid(row=1, column=1, sticky="ew", padx=(3, 0), pady=4)
        self.move_button = ttk.Button(actions, text="发送当前姿态", command=self.run_pose, style="Accent.TButton")
        self.move_button.grid(row=2, column=0, columnspan=2, sticky="ew")
        summary = self.label(actions, var=self.binding_status, color=MUTED, size=8)
        summary.configure(wraplength=262, justify="left")
        summary.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.actions_frame = actions

        if not integrated:
            self.tabs = ttk.Notebook(shell)
            self.tabs.grid(row=3, column=1, columnspan=2, sticky="nsew")
        feedback = tk.Frame(self.tabs, bg=PANEL, padx=10, pady=8)
        chart_view = tk.Frame(self.tabs, bg=PANEL, padx=10, pady=8)
        log_view = tk.Frame(self.tabs, bg=PANEL, padx=10, pady=8)
        for frame, title in ((feedback, "  舵机监视  "), (chart_view, "  目标 / 反馈曲线  "),
                             (log_view, "  运行日志  ")):
            self.tabs.add(frame, text=title)
        self.tabs.bind("<<NotebookTabChanged>>", lambda _: self.draw_chart())
        self.feedback_note = self.label(feedback, "等待实机反馈 · 双击舵机行查看设备详情", color=MUTED, size=9)
        self.feedback_note.pack(anchor="w", pady=(0, 7))
        columns = (("id", "ID", 40), ("finger", "手指", 65), ("target", "目标编码", 85),
                   ("position", "反馈编码", 85), ("error", "偏差 °", 70), ("speed", "转速 RPM", 85),
                   ("current", "相电流 mA", 90), ("voltage", "电压 V", 70),
                   ("temperature", "温度 °C", 70), ("status", "状态", 80))
        self.table = self.make_table(feedback, columns, height=3)
        self.table.tag_configure("fault", foreground=ORANGE)
        self.table.bind("<Double-1>", self.show_details)
        self.table.bind("<<TreeviewSelect>>", self.select_chart_motor)
        chart_header = tk.Frame(chart_view, bg=PANEL)
        chart_header.pack(fill="x")
        self.label(chart_header, "选择监测电机", color=MUTED, size=9).pack(side="left")
        self.chart_motor_box = ttk.Combobox(chart_header, textvariable=self.chart_motor, width=8, state="readonly")
        self.chart_motor_box.pack(side="left", padx=(6, 12))
        self.chart_motor_box.bind("<<ComboboxSelected>>", lambda _: self.draw_chart())
        self.label(chart_header, "实线：反馈编码    虚线：目标编码    ·    横轴：秒", color=MUTED, size=9).pack(side="left")
        self.chart = tk.Canvas(chart_view, bg=PANEL, height=120, highlightthickness=0)
        self.chart.pack(fill="both", expand=True)
        self.chart.bind("<Configure>", lambda _: self.draw_chart())

        log_toolbar = tk.Frame(log_view, bg=PANEL)
        log_toolbar.pack(fill="x", pady=(0, 5))
        self.label(log_toolbar, "保留最近 500 条操作、连接和错误记录", color=MUTED, size=9).pack(side="left")
        ttk.Button(log_toolbar, text="保存日志…", command=self.save_log, style="Small.TButton").pack(side="right")
        self.log_table = self.make_table(log_view, (("time", "时间", 90), ("level", "类型", 70), ("message", "事件", 660)), height=3)
        self.log_table.tag_configure("错误", foreground=ORANGE)
        self.log_table.tag_configure("操作", foreground=CYAN)
        self.mapping_view = tk.Frame(self.tabs, bg=PANEL, padx=10, pady=8)
        self.tabs.add(self.mapping_view, text="  差动映射  ")
        mapping_header = tk.Frame(self.mapping_view, bg=PANEL)
        mapping_header.pack(fill="x", pady=(0, 4))
        self.mapping_box = ttk.Combobox(mapping_header, textvariable=self.mapping_group, state="readonly", width=14)
        self.mapping_box.pack(side="left", padx=(0, 10))
        self.mapping_box.bind("<<ComboboxSelected>>", self.select_mapping_group)
        self.mapping_buttons = []
        for kind, text in (("flex", "屈曲预览"), ("side", "侧摆/旋转预览"), ("combined", "组合预览"), ("zero", "双轴归零预览")):
            button = ttk.Button(mapping_header, text=text, command=lambda k=kind: self.preview_differential(k), style="Small.TButton")
            button.pack(side="left", padx=(0, 5))
            self.mapping_buttons.append(button)
        self.label(mapping_header, var=self.mapping_inputs, color=CYAN, size=9).pack(side="right")
        self.label(self.mapping_view, var=self.mapping_formula, color=MUTED, size=9).pack(anchor="w", pady=(0, 4))
        self.mapping_table = self.make_table(self.mapping_view, (
            ("id", "舵机 ID", 65), ("flex", "屈曲贡献 °", 105), ("side", "第二轴贡献 °", 110),
            ("angle", "舵机转角 °", 105), ("zero", "零位编码", 90), ("target", "目标编码", 90),
            ("limits", "编码范围", 115), ("status", "整帧校验", 140)), height=2)
        self.mapping_table.tag_configure("fault", foreground=ORANGE)
        if integrated:
            self.tabs.add(self.manual_view, text="  手动关节调试  ")

        footer = tk.Frame(shell, bg=BG)
        footer.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.status_label = self.label(footer, var=self.message, color=MUTED, size=9)
        self.status_label.pack(side="left", fill="x", expand=True)
        self.label(footer, "HL-3606-C001  ·  HLS TTL", color=MUTED, size=9).pack(side="right")

    @staticmethod
    def scroll_panel(parent, width):
        outer = tk.Frame(parent, bg=PANEL)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)
        canvas = tk.Canvas(outer, bg=PANEL, width=width, height=1, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        bar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        bar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=bar.set)
        content = tk.Frame(canvas, bg=PANEL, padx=12, pady=10)
        window = canvas.create_window(0, 0, window=content, anchor="nw")

        def fit(_event=None):
            canvas.itemconfigure(window, width=canvas.winfo_width())
            canvas.configure(scrollregion=canvas.bbox("all"))
        content.bind("<Configure>", fit)
        canvas.bind("<Configure>", fit)

        def scroll(event):
            if content.winfo_reqheight() > canvas.winfo_height():
                canvas.yview_scroll(-int(event.delta / 120), "units")
                return "break"
        # 仅处理本面板子控件的滚轮，避免干扰下方表格和下拉框。
        def bind_children():
            pending = [content]
            while pending:
                widget = pending.pop()
                widget.bind("<MouseWheel>", scroll, add="+")
                pending.extend(widget.winfo_children())
        parent.after_idle(bind_children)
        return outer, content

    @staticmethod
    def make_table(parent, columns, height):
        container = tk.Frame(parent, bg=PANEL)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)
        table = ttk.Treeview(container, columns=[item[0] for item in columns], show="headings", height=height, selectmode="browse")
        from tkinter import font
        heading_font = font.Font(root=parent, font=("Microsoft YaHei UI", 9))
        for key, title, width in columns:
            table.heading(key, text=title)
            minimum = min(width, max(35, heading_font.measure(title) + 18))
            table.column(key, width=width, minwidth=minimum, anchor="w" if key in ("message", "note", "name") else "center")
        scroll = ttk.Scrollbar(container, orient="vertical", command=table.yview)
        horizontal = ttk.Scrollbar(container, orient="horizontal", command=table.xview)
        table.configure(yscrollcommand=scroll.set, xscrollcommand=horizontal.set)
        scroll.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        table.grid(row=0, column=0, sticky="nsew")
        table.tag_configure("even", background=PANEL)
        table.tag_configure("odd", background="#192a40")
        return table

    def draw(self):
        self.card_hits = self.draw_joint_cards(self.canvas)

    def draw_joint_cards(self, canvas):
        """Render the overview in the layout chosen by its host."""
        if self.config is None:
            return []
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 100 or height < 70:
            return []
        if getattr(self, "integrated_workspace", False) and canvas is self.canvas:
            return self.draw_joint_rows(canvas, width, height)
        hits = []
        count = len(self.groups)
        gap = 8
        card_width = (width - (count - 1) * gap - 2) / count
        for column, (name, joints) in enumerate(self.groups.items()):
            x = 1 + column * (card_width + gap)
            right = x + card_width
            selected = name == self.finger.get()
            canvas.create_rectangle(x, 1, right, height - 1, fill=INPUT if selected else PANEL, outline=CYAN if selected else LINE, width=1)
            if selected:
                canvas.create_line(x + 1, 2, right - 1, 2, fill=CYAN, width=3)
            hits.append((x, right, name))
            canvas.create_text(x + 10, 16, text=name, anchor="w", fill=CYAN if selected else TEXT, font=("Microsoft YaHei UI", 11, "bold"))
            canvas.create_text(right - 8, 16, text=f"{len(joints)}轴", anchor="e", fill=MUTED, font=("Microsoft YaHei UI", 8))
            if height < 140:
                # 小投影窗口使用两行摘要，给播放控件保留完整高度。
                short = {"mcp_flex": "屈", "mcp_abd": "摆", "pip_flex": "P", "dip_flex": "D",
                         "cmc_flex": "屈", "cmc_roll": "旋", "ip_flex": "IP"}
                for index, joint in enumerate(joints):
                    cell_x = x + 8 + (index % 2) * (card_width - 12) / 2
                    cell_y = 35 + (index // 2) * 19
                    axis = joint.partition("_")[2]
                    canvas.create_text(cell_x, cell_y, text=f"{short.get(axis, axis)} {self.angles[joint]:.1f}", anchor="w", fill=TEXT, font=("Microsoft YaHei UI", 7))
                continue
            row_height = min(48, max(17, (height - 58) / max(1, len(joints))))
            for row, joint in enumerate(joints):
                y = 38 + row * row_height
                axis = joint.partition("_")[2]
                angle = self.angles[joint]
                low, high = self.config.joints[joint]
                compact = card_width < 145 and row_height >= 30
                narrow_axes = {"mcp_flex": "MCP屈", "mcp_abd": "MCP摆", "pip_flex": "PIP屈",
                               "dip_flex": "DIP屈", "cmc_flex": "CMC屈", "cmc_roll": "CMC旋", "ip_flex": "IP屈"}
                axis_text = narrow_axes.get(axis, axis) if card_width < 120 and not compact else AXES.get(axis, axis)
                canvas.create_text(x + 9, y, text=axis_text, anchor="w", fill=MUTED,
                    font=("Microsoft YaHei UI", 7 if card_width < 105 else 8 if card_width < 145 else 9), tags=(f"axis_label:{column}:{row}",))
                canvas.create_text(right - 9, y + (16 if compact else 0), text=f"{angle:.1f}", anchor="e", fill=TEXT,
                    font=("Consolas", 10 if card_width < 145 else 11, "bold"), tags=(f"axis_value:{column}:{row}",))
                bar_y = y + (26 if compact else 14)
                bar_left, bar_right = x + 9, right - 9
                if row_height >= 27:
                    canvas.create_line(bar_left, bar_y, bar_right, bar_y, fill=LINE, width=3)
                if high > low and row_height >= 27:
                    ratio = max(0, min(1, (angle - low) / (high - low)))
                    zero = max(0, min(1, -low / (high - low)))
                    canvas.create_line(bar_left + zero * (bar_right - bar_left), bar_y,
                                       bar_left + ratio * (bar_right - bar_left), bar_y, fill=CYAN if selected else BLUE, width=3)
            group_key = joints[0].partition("_")[0]
            ids = [str(sid) for sid, motor in self.config.motors.items() if any(joint.startswith(group_key + "_") for joint in motor.weights)]
            id_text = "ID " + "/".join(ids) if ids else "未绑定"
            if len(id_text) * 6 > card_width - 18 and ids:
                id_text = f"{len(ids)} 台舵机"
            canvas.create_text(x + 9, height - 13, text=id_text, anchor="w", fill=MUTED, font=("Consolas", 8))
        return hits

    def draw_joint_rows(self, canvas, width, height):
        """Five stacked finger cards, four readable joint values per finger."""
        count = len(self.groups)
        gap = 4
        row_height = (height - 2 - gap * (count - 1)) / count
        value_size = 12 if row_height >= 70 else 11 if row_height >= 50 else 10
        label_size = 9 if row_height >= 55 else 8
        finger_size = 11 if row_height >= 55 else 10
        hits = []
        name_width = 67
        cell_width = (width - name_width - 12) / 2
        for index, (name, joints) in enumerate(self.groups.items()):
            top = 1 + index * (row_height + gap)
            bottom = top + row_height
            selected = name == self.finger.get()
            canvas.create_rectangle(1, top, width - 1, bottom,
                fill=INPUT if selected else PANEL, outline=CYAN if selected else LINE)
            hits.append((1, width - 1, top, bottom, name))
            canvas.create_text(9, top + row_height * .30, text=name, anchor="w",
                fill=CYAN if selected else TEXT, font=("Microsoft YaHei UI", finger_size, "bold"))
            group = joints[0].partition("_")[0]
            ids = [str(sid) for sid, motor in self.config.motors.items()
                   if any(j.startswith(group + "_") for j in motor.weights)]
            id_text = "ID " + "/".join(ids) if ids else "未绑定"
            if len(id_text) > 10:
                id_text = f"{len(ids)} 台"
            canvas.create_text(9, top + row_height * .74, text=id_text, anchor="w",
                fill=MUTED, font=("Microsoft YaHei UI", 8 if row_height >= 55 else 7))
            for row, joint in enumerate(joints):
                left = name_width + (row % 2) * cell_width
                y = top + row_height * (.29 if row < 2 else .74)
                axis = joint.partition("_")[2]
                canvas.create_text(left, y, text=AXES.get(axis, axis), anchor="w", fill=MUTED,
                    font=("Microsoft YaHei UI", label_size), tags=(f"axis_label:{index}:{row}",))
                canvas.create_text(left + cell_width - 9, y, text=f"{self.angles[joint]:.1f}", anchor="e",
                    fill=TEXT, font=("Consolas", value_size, "bold"), tags=(f"axis_value:{index}:{row}",))
        return hits

    def select_finger_card(self, event):
        if self.busy:
            return
        hits = getattr(self, "card_hits", [])
        for hit in hits:
            if len(hit) == 5:
                left, right, top, bottom, name = hit
                inside = left <= event.x <= right and top <= event.y <= bottom
            else:
                left, right, name = hit
                inside = left <= event.x <= right
            if inside:
                self.finger.set(name)
                self.rebuild_sliders()
                return

    def draw_chart(self):
        if self.config is None or not self.chart.winfo_ismapped():
            return
        chart = self.chart
        chart.delete("all")
        width, height = chart.winfo_width(), chart.winfo_height()
        if width < 120 or height < 70:
            return
        try:
            sid = int(self.chart_motor.get().replace("ID ", ""))
        except ValueError:
            sid = next(iter(self.config.motors), None)
        history = [row for row in self.samples if row["id"] == sid and row["mode"] == "实机" and isinstance(row["position_tick"], (int, float))][-200:]
        if len(history) < 2:
            chart.create_text(width / 2, height / 2, text="发送姿态或启动视觉联动后显示实测曲线", fill=MUTED, font=FONT)
            return
        values = [row[key] for row in history for key in ("target_tick", "position_tick")]
        low, high = min(values) - 8, max(values) + 8
        left, top, right, bottom = 53, 12, width - 13, height - 24
        for fraction in (0, .5, 1):
            level = low + (high - low) * fraction
            y = bottom - fraction * (bottom - top)
            chart.create_line(left, y, right, y, fill=LINE)
            chart.create_text(left - 7, y, text=f"{level:.0f}", anchor="e", fill=MUTED, font=("Consolas", 9))
        first, last = history[0]["time_unix_s"], history[-1]["time_unix_s"]
        span = max(.001, last - first)
        for fraction in (0, .25, .5, .75, 1):
            x = left + fraction * (right - left)
            chart.create_text(x, bottom + 13, text=f"{span * fraction:.1f}s", fill=MUTED, font=("Consolas", 9))
        for key, color, dash in (("position_tick", CYAN, ()), ("target_tick", ORANGE, (4, 3))):
            points = []
            for row in history:
                points.extend((left + (row["time_unix_s"] - first) / span * (right - left),
                               bottom - (row[key] - low) / (high - low) * (bottom - top)))
            chart.create_line(*points, fill=color, width=2, dash=dash)

    def update_metrics(self):
        if self.config is None:
            return
        self.metric_vars["config"].set(f"{len(self.config.joints)}  /  {len(self.config.motors)}")
        self.metric_vars["frame"].set(f"{self.frame_count:05d}")
        now = time.monotonic()
        stamps = [stamp for stamp in self.frame_times if now - stamp < 3]
        span = stamps[-1] - stamps[0] if len(stamps) > 1 else 0
        rate = (len(stamps) - 1) / span if span > 0 else 0
        self.metric_vars["rate"].set(f"{rate:.1f} Hz")
        if self.readings:
            age = now - self.last_feedback
            self.feedback_note.configure(text=(f"实测反馈 · 最近更新 {age:.1f} 秒前 · 双击行查看详情" if self.busy
                else "已停止采集 · 下表保留最后一次接收读数，非当前实时状态"))
            self.metric_vars["feedback"].set("实测 / 已停止" if not self.busy else "实测 / 读取中")
            received = time.strftime("%H:%M:%S", time.localtime(time.time() - age))
            self.connection_info.set(f"收到 {len(self.readings)}/{len(self.config.motors)} 个 ID 的反馈\n"
                + (f"最近更新 {age:.0f} 秒前" if self.busy else f"采集已停止 · {received}"))
        else:
            self.feedback_note.configure(text="等待实机反馈 · 预览角度不会驱动舵机，双击行查看设备详情")
            self.metric_vars["feedback"].set("等待实测")
            self.connection_info.set("选择串口后配置舵机或采集绑定零位")
