"""
成员F — MuJoCo 实时播放演示
用法: python scripts/demo_play.py
自动加载轨迹文件(如果存在)，否则生成模拟舞蹈
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import mujoco
import mujoco.viewer

TRAJECTORY_FILE = "outputs/trajectory.npy"

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
home_qpos = data.qpos.copy()

# 加载轨迹
if Path(TRAJECTORY_FILE).exists():
    offsets = np.load(TRAJECTORY_FILE)
    print(f"已加载轨迹: {TRAJECTORY_FILE} ({offsets.shape[0]} 帧)")
else:
    T, J = 600, model.nu
    offsets = np.zeros((T, J))
    for t in range(T):
        for j in range(J):
            offsets[t, j] = (
                0.08 * np.sin(2 * np.pi * t / 120 + j * 0.4)
                + 0.04 * np.sin(2 * np.pi * t / 60 + j * 0.7)
            )
    print(f"使用模拟舞蹈 ({T} 帧)")

T = offsets.shape[0]
J = min(offsets.shape[1], model.nu)
FREE_DOF = 7

print(f"模型: {model.nu} 关节, {T} 帧, 按 ESC 退出")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = -160
    viewer.cam.elevation = -20
    viewer.cam.distance = 2.0

    frame = 0
    while viewer.is_running():
        t = frame % T
        data.qpos[FREE_DOF : FREE_DOF + J] = home_qpos[FREE_DOF : FREE_DOF + J] + offsets[t, :J]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        viewer.sync()
        frame += 1
        if frame % 30 == 0:
            print(f"  播放中... {frame // 30}s")

print(f"播放结束")
