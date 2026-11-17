"""手动姿态预设，仅计算目标；发送由实机控制台显式触发。"""

PRESETS = {
    "open": ("张开", "各关节回到配置零位"),
    "half": ("半握", "同时调整各指屈曲关节"),
    "fist": ("握拳姿态", "展示多关节协同；不代表抓取已验证"),
    "pinch": ("对捏姿态", "食指与拇指的目标参数示例"),
    "spread": ("侧摆展开", "展示 MCP 侧摆与差动目标分配"),
    "point": ("食指伸展", "食指伸展，其余手指屈曲"),
}


def preset_pose(config, key):
    if key not in PRESETS:
        raise ValueError("未知动作预设")
    pose = {}
    for joint, (low, high) in config.joints.items():
        finger, _, axis = joint.partition("_")
        side = axis.endswith(("abd", "roll"))
        ratio = {"open": 0, "half": .45, "fist": .82, "pinch": .68,
                 "spread": .12, "point": .72}[key]
        if key == "pinch" and finger not in ("index", "thumb"):
            ratio = .12
        if key == "point" and finger == "index":
            ratio = 0
        angle = max(low, min(high, max(0, high) * ratio)) if not side else max(low, min(high, 0))
        if key == "spread" and side:
            direction = {"index": 1, "thumb": 1, "middle": .1, "ring": -.5, "little": -1}.get(finger, 0)
            angle = .7 * direction * (max(0, high) if direction >= 0 else abs(min(0, low)))
            angle = max(low, min(high, angle))
        pose[joint] = round(angle, 3)
    config.targets(pose)
    return pose
