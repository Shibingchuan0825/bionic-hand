# 🤖 Bionic Dexterous Hand Controller

基于 **MediaPipe** 手部追踪的实时仿生灵巧手控制系统。通过摄像头捕捉人手动作，将 21 个手部关键点映射到 12 个舵机角度，实时控制仿生机械手。

---

## 📋 目录

- [系统架构](#-系统架构)
- [功能特性](#-功能特性)
- [快速开始](#-快速开始)
- [硬件要求](#-硬件要求)
- [使用说明](#-使用说明)
- [配置说明](#-配置说明)
- [手势映射原理](#-手势映射原理)
- [通信协议](#-通信协议)
- [项目结构](#-项目结构)
- [运行测试](#-运行测试)
- [常见问题](#-常见问题)

---

## 🏗 系统架构

```
┌──────────┐     ┌──────────────────┐     ┌───────────────┐     ┌──────────┐
│  Camera  │────▶│  MediaPipe       │────▶│  Joint Angle  │────▶│  Servo   │
│  (Webcam)│     │  Hand Tracker    │     │  Mapper       │     │  Manager │
└──────────┘     │  (21 landmarks)  │     │  (landmark→°) │     │  (°→PWM) │
                 └──────────────────┘     └───────────────┘     └────┬─────┘
                                                                     │
                                                              Serial │
                                                              ┌──────▼─────┐
                                                              │  Arduino   │
                                                              │  (PCA9685) │
                                                              └──────┬─────┘
                                                                     │
                                                              ┌──────▼─────┐
                                                              │  12x Servo │
                                                              │  Motors    │
                                                              └────────────┘
```

| 模块 | 功能 | 文件 |
|------|------|------|
| **Hand Tracker** | MediaPipe 手部关键点检测 | `src/hand_tracking/` |
| **Joint Mapper** | 关键点→关节角度映射 | `src/hand_mapper/` |
| **Servo Manager** | 舵机角度管理与PWM转换 | `src/motor_control/servo_manager.py` |
| **Serial Controller** | PC与Arduino串口通信 | `src/motor_control/serial_controller.py` |
| **Hand Display** | 实时可视化界面 | `src/ui/` |
| **Arduino Firmware** | 舵机驱动与控制 | `arduino/bionic_hand/` |

---

## ✨ 功能特性

- **实时手部追踪** — 使用 Google MediaPipe 检测 21 个手部 3D 关键点，精度高、延迟低
- **12 自由度控制** — 拇指(3)、食指(2)、中指(2)、无名指(2)、小指(2)、手腕(1)
- **双模式运行** — 支持仿真模式（无需硬件）和实际硬件模式
- **安全保护** — 角度限位、平滑滤波、指令超时自动复位
- **实时可视化** — 摄像头画面+关键点标注+关节角度仪表盘
- **可配置** — YAML 配置文件，支持自定义舵机引脚、角度范围、手势预设等
- **跨平台** — Windows / Linux / macOS 均可运行

---

## 🚀 快速开始

### 1. 环境要求

- Python 3.9+
- 摄像头（内置或 USB 外接）
- （可选）Arduino + PCA9685 + 舵机

### 2. 安装依赖

```bash
# 克隆项目
cd d:/hand

# 创建虚拟环境（推荐）
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/Mac:
# source venv/bin/activate

# 安装 Python 依赖
pip install -r requirements.txt
```

### 3. 仿真模式运行（无需硬件）

```bash
# 默认仿真模式（使用默认摄像头）
python -m src.main

# 指定摄像头
python -m src.main --camera 0

# 查看可用摄像头
python -m src.main --list-cameras

# 纯控制台模式（不显示 GUI）
python -m src.main --no-display
```

### 4. 硬件模式运行

```bash
# 使用真实硬件
python -m src.main --real

# 指定配置文件
python -m src.main --config config/my_hand.yaml --real
```

### 5. 灵巧手硬件测试工具

在接入摄像头控制前，建议先使用独立测试工具验证串口、舵机方向、行程和预置手势。默认使用仿真串口，不会驱动真实舵机；接入实物后必须显式加 `--real`。

```bash
# 仿真模式：测试串口回显、列出 12 路舵机
python -m src.diagnostics.hand_tester ping
python -m src.diagnostics.hand_tester list

# 仿真模式：单舵机或全手扫角
python -m src.diagnostics.hand_tester sweep --servo index_mcp
python -m src.diagnostics.hand_tester sweep --all

# 真实硬件：先 ping，再做低速、小范围验证
python -m src.diagnostics.hand_tester --real ping
python -m src.diagnostics.hand_tester --real gestures --cycles 1
python -m src.diagnostics.hand_tester --real soak --iterations 30
```

真实硬件首次测试前，请确认外部 5--6 V 舵机电源、公共地、PCA9685 通道和机械限位；未完成单舵机测试前不要直接执行全行程扫角。

---

## 🔧 硬件要求

### 最小硬件配置（初始验证，不是最终采购单）

| 组件 | 型号建议 | 数量 | 用途 |
|------|----------|------|------|
| 舵机驱动板 | **PCA9685** (16通道PWM) | 1 | 生成舵机控制信号 |
| 舵机样品 | 通过扭矩/电流/尺寸测试后确定 | 2--3 | Gate A 单轴验证 |
| Arduino | Uno / Mega / Nano（先冻结一种） | 1 | 接收串口指令 |
| 外部电源 | 5--6V，电流按实测峰值和余量确定 | 1 | 为舵机供电（**勿用Arduino 5V引脚！**） |
| 物理急停/保险 | 可切断舵机 V+ | 1 | 电气安全 |
| 摄像头 | USB Webcam | 1 | 捕捉手部动作 |

当前软件基线支持 12 个逻辑通道；`wrist_pitch`、`wrist_yaw` 仅在硬件清单中预留，尚未加入固件或 Python 角度协议。项目资料中旧的“7自由度”方案与当前 12 通道软件并不等价，必须以实物轴表和测量记录为准。

### 接线图

```
Arduino (Uno)          PCA9685              12x Servo
┌──────────┐      ┌──────────────┐      ┌─────────────┐
│ SDA (A4) ├──────┤ SDA          │      │             │
│ SCL (A5) ├──────┤ SCL          │      │ PWM0  → Thumb CMC
│ 5V       ├──────┤ VCC          │      │ PWM1  → Thumb MCP
│ GND      ├──────┤ GND          │      │ PWM2  → Thumb IP
└──────────┘      │              │      │ PWM3  → Index MCP
                  │ V+  ←── 5-6V │      │ PWM4  → Index PIP
                  │ GND ←── GND  │      │ PWM5  → Middle MCP
                  └──────────────┘      │ ...      ...
                                        │ PWM10 → Pinky PIP
                                        │ PWM11 → Wrist rotation
                                        └─────────────┘
        External Power (5-6V, current sized from measurements)
        ┌──────────────────────┐
        │ V+ → Servo V+ (红)   │
        │ GND → Servo GND (棕) │
        └──────────────────────┘
```

> ⚠️ **重要提示**：舵机必须由外部电源供电，**不要**从 Arduino 5V 引脚取电！最终电源电流必须按实际舵机在同时动作时的测量峰值（并留出余量）确定；“3A+”只能作为轻载单轴验证的临时假设，不能作为 12--14 轴整机规格。

---

## 📖 使用说明

### 键盘快捷键

| 按键 | 功能 |
|------|------|
| `q` / `ESC` | 退出程序 |
| `p` | 暂停/恢复舵机控制 |
| `r` | 重置平滑滤波器 + 复位姿态 |
| `d` | 设置默认（张开手）姿态 |
| `h` | 显示帮助信息 |
| `s` | 打印当前舵机状态 |

### 运行流程

1. 启动程序后，将手放在摄像头前（手背或手心朝向摄像头）
2. MediaPipe 自动检测手部关键点（显示绿色骨架线）
3. 系统实时计算关节角度并显示在仪表盘上
4. 角度指令通过串口发送到 Arduino
5. Arduino 驱动舵机复现人手动作

---

## ⚙ 配置说明

配置文件位于 `config/hand_config.yaml`，所有参数均可调整：

```yaml
# 摄像头设置
camera:
  index: 0              # 摄像头设备编号
  width: 640            # 采集分辨率
  height: 480
  flip_horizontal: false # 水平镜像

# MediaPipe 设置
mediapipe:
  max_num_hands: 1          # 追踪手数量（建议1）
  min_detection_confidence: 0.7  # 检测置信度阈值
  # static_image_mode: false=VIDEO（连续追踪，实时摄像头）；true=IMAGE（逐帧独立检测）
  # Tasks API 只提供 hand_landmarker 一个模型，没有 Lite/Full 复杂度开关

# 舵机引脚分配
servos:
  pins:
    thumb_mcp:  2
    index_mcp:  5
    # ... 完整列表见配置文件

  angle_limits:
    min_angle: 0        # 最小安全角度
    max_angle: 180      # 最大安全角度

  smoothing:
    enabled: true
    alpha: 0.3           # EMA 平滑系数（越小越平滑但延迟越大）
    max_step: 5          # 每帧最大角度变化量

# 串口通信
serial:
  port: "COM3"           # Windows: COM3, Linux: /dev/ttyUSB0
  baudrate: 115200

# 仿真模式
simulation:
  enabled: true          # true=仿真, false=实际硬件
```

### 手势预设

配置文件中内置了 4 个手势预设，可在代码中调用：

| 手势 | 说明 |
|------|------|
| `open_hand` | 五指张开 |
| `fist` | 握拳 |
| `pinch` | 拇指+食指捏合 |
| `point` | 食指指向 |

---

## 🧠 手势映射原理

### MediaPipe 手部关键点

MediaPipe 检测 **21 个手部关键点**，每个点有 (x, y, z) 三维坐标：

```
     8                                  12
     |                                   |
     7                                  11
    /                                   /
    6                                  10
   /                                   /
  5──────────9──────────13──────────17─── 0 (手腕)
   \                                   \
    1                                  14
     \                                  \
      2                                 15
       \                                 \
        3                                 16
        |                                 |
        4                                 20
```

### 角度计算方法

关节角度使用**向量点积法**计算：

```
θ = arccos( (v1 · v2) / (|v1| × |v2|) )

其中：
  v1 = p_before - p_joint  (指向关节的向量)
  v2 = p_after  - p_joint  (离开关节的向量)

结果：180° = 完全伸直，0° = 完全弯曲
```

### 映射示例

以食指 MCP 关节为例：
- 使用关键点：手腕(0) → 食指MCP(5) → 食指PIP(6)
- 计算向量 (0→5) 和 (6→5) 之间的夹角
- 该角度直接作为舵机的目标角度

---

## 📡 通信协议

### PC → Arduino 指令格式

```
手指名:角度,手指名:角度,...\n
```

示例：
```
thumb_cmc:140,thumb_mcp:130,thumb_ip:120,index_mcp:160,index_pip:170,middle_mcp:155,middle_pip:165,ring_mcp:150,ring_pip:160,pinky_mcp:145,pinky_pip:155,wrist_rot:90\n
```

### 关节名称对照表

| 键名 | 含义 | 手指 |
|------|------|------|
| `thumb_cmc` | 拇指腕掌关节 | 拇指 |
| `thumb_mcp` | 拇指掌指关节 | 拇指 |
| `thumb_ip` | 拇指指间关节 | 拇指 |
| `index_mcp` | 食指掌指关节 | 食指 |
| `index_pip` | 食指近端指间关节 | 食指 |
| `middle_mcp` | 中指掌指关节 | 中指 |
| `middle_pip` | 中指近端指间关节 | 中指 |
| `ring_mcp` | 无名指掌指关节 | 无名指 |
| `ring_pip` | 无名指近端指间关节 | 无名指 |
| `pinky_mcp` | 小指掌指关节 | 小指 |
| `pinky_pip` | 小指近端指间关节 | 小指 |
| `wrist_rot` | 手腕旋转 | 手腕 |

### Arduino 响应格式

```
OK:N servos updated
```

---

## 📁 项目结构

```
d:/hand/
├── README.md                          # 项目文档
├── requirements.txt                   # Python 依赖
├── config/
│   ├── hand_config.yaml               # 应用配置、手势和串口设置
│   └── hardware_manifest.yaml         # 轴/DoF/PCA9685通道权威清单
├── src/
│   ├── __init__.py
│   ├── main.py                        # 主入口（CLI + 控制循环）
│   ├── hand_tracking/
│   │   ├── __init__.py
│   │   └── mediapipe_tracker.py       # MediaPipe 手部追踪
│   ├── hand_mapper/
│   │   ├── __init__.py
│   │   └── joint_mapper.py            # 关键点→角度映射
│   ├── motor_control/
│   │   ├── __init__.py
│   │   ├── servo_manager.py           # 舵机角度管理
│   │   └── serial_controller.py       # 串口通信
│   ├── ui/
│   │   ├── __init__.py
│   │   └── display.py                 # 可视化界面
│   └── diagnostics/
│       ├── __init__.py
│       ├── hand_tester.py             # 无摄像头的硬件/串口测试工具
│       └── hardware_validator.py      # 配置、固件和通道一致性检查
├── arduino/
│   └── bionic_hand/
│       └── bionic_hand.ino            # Arduino 固件
├── tests/
│   ├── __init__.py
│   ├── test_joint_mapper.py           # 角度映射测试
│   ├── test_servo_manager.py          # 舵机管理测试
│   ├── test_hand_tester.py            # 硬件测试工具仿真测试
│   ├── test_hardware_validator.py     # 硬件清单一致性测试
│   └── test_serial_controller.py      # PING/ACK协议测试
└── models/                            # MediaPipe 模型缓存
```

---

## 🧪 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行单个测试文件
pytest tests/test_joint_mapper.py -v
pytest tests/test_servo_manager.py -v

# 带覆盖率报告
pytest tests/ --cov=src --cov-report=html
```

---

### 硬件清单和首轮安全测试

硬件仍处于采购前验证阶段。先运行清单校验，再只用限流电源、物理急停和单舵机小范围动作；软件通道数不等于已经完成的机械自由度。

```bash
python -m src.diagnostics.hardware_validator
python -m src.diagnostics.hand_tester ping
python -m src.diagnostics.hand_tester sweep --servo index_mcp
```

`hardware_manifest.yaml` 的默认软限位是 80--100° 的台架测试窗口，不是已测得的机械极限。只有在确认舵机型号、机械限位、供电和防护后，才可由现场负责人显式使用 `--full-range`；不得把仿真通过或软件中的 12 路名称写成 12 路实物自由度。

采购按三关推进：Gate A 购买少量舵机样品、PCA9685、主控、限流电源和安全/测量件；Gate B 完成一个手指和一个腕轴样机；Gate C 通过电流、扭矩、机械限位和多轴测试后再购买剩余执行器。记录模板见 `docs/hardware/`。

---


### Q: "No hand detected" 一直显示
- 确保手在摄像头视野内，光照充足
- 调整 `min_detection_confidence` 降低至 0.5
- 尝试将手背朝向摄像头（MediaPipe 对手背的检测更稳定）

### Q: 舵机抖动/不稳定
- 增大 `smoothing.alpha`（如 0.5）增加平滑力度
- 减小 `smoothing.max_step` 限制每帧角度变化
- 检查外部电源是否足够，并记录同时动作时的电压、电流和温升；不能直接假定 12 个舵机只需要 3A

### Q: 串口连接失败
- 检查串口号：Windows 用 `COM3`，Linux 用 `/dev/ttyUSB0`，Mac 用 `/dev/cu.usbmodem*`
- 关闭 Arduino IDE 的串口监视器（端口只能被一个程序占用）
- 使用仿真模式先测试：`python -m src.main --sim`

### Q: 摄像头打不开
- 运行 `python -m src.main --list-cameras` 查看可用设备
- 关闭其他占用摄像头的应用（视频会议、浏览器等）

### Q: FPS 很低
- 若不需要跨帧追踪，设置 `static_image_mode: true`（逐帧独立检测，无追踪开销）
- 降低摄像头分辨率：`width: 320, height: 240`
- 关闭仪表盘显示：`python -m src.main --no-display`

---

## 📄 License

MIT License

---

## 🙏 致谢

- [Google MediaPipe](https://developers.google.com/mediapipe) - 手部追踪框架
- [OpenCV](https://opencv.org/) - 计算机视觉库
- [Adafruit PCA9685](https://www.adafruit.com/product/815) - PWM 驱动库
