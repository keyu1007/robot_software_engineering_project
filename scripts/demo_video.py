"""
成员F — 视频导出演示
用法: python scripts/demo_video.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from common.motion_data import MotionData, BOOSTER_T1_JOINT_NAMES
from project1_dance.mujoco_player.player import MuJoCoPlayerImpl

TRAJECTORY_FILE = "outputs/trajectory.npy"

if Path(TRAJECTORY_FILE).exists():
    offsets = np.load(TRAJECTORY_FILE)
    print(f"已加载轨迹: {TRAJECTORY_FILE} ({offsets.shape[0]} 帧)")
else:
    T = 300
    offsets = np.zeros((T, 23))
    for t in range(T):
        for j in range(23):
            offsets[t, j] = (
                0.08 * np.sin(2 * np.pi * t / 120 + j * 0.4)
                + 0.04 * np.sin(2 * np.pi * t / 60 + j * 0.7)
            )
    print(f"使用模拟舞蹈")

T = offsets.shape[0]
motion = MotionData(
    joint_names=list(BOOSTER_T1_JOINT_NAMES),
    fps=30,
    num_frames=T,
    angles=offsets,
    timestamps=np.arange(T) / 30.0,
)

player = MuJoCoPlayerImpl(width=640, height=480)
player.load_model("scene.xml")

output = "outputs/dance_demo.mp4"
player.play(motion, output_path=output, render=True)
print(f"视频已导出: {output}")
