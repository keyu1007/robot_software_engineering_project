"""
动作重定向模块 — 成员D

将人体关键点位置转换为 Booster T1 机器人23个执行器角度。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from common.interfaces import Retargeting
from common.motion_data import MotionData, BOOSTER_T1_JOINT_NAMES

# MuJoCo 执行器顺序（与 t1.xml 一致）
ACTUATOR_NAMES = [
    "AAHead_yaw", "Head_pitch",
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Waist",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch",
    "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch",
    "Right_Ankle_Pitch", "Right_Ankle_Roll",
]

# t1.xml 关节限位（弧度）
JOINT_LIMITS = {
    "AAHead_yaw": (-1.57, 1.57), "Head_pitch": (-0.35, 1.22),
    "Left_Shoulder_Pitch": (-3.31, 1.22), "Left_Shoulder_Roll": (-1.74, 1.57),
    "Left_Elbow_Pitch": (-2.27, 2.27), "Left_Elbow_Yaw": (-2.44, 0.0),
    "Right_Shoulder_Pitch": (-3.31, 1.22), "Right_Shoulder_Roll": (-1.57, 1.74),
    "Right_Elbow_Pitch": (-2.27, 2.27), "Right_Elbow_Yaw": (0.0, 2.44),
    "Waist": (-1.57, 1.57),
    "Left_Hip_Pitch": (-1.8, 1.57), "Left_Hip_Roll": (-0.2, 1.57),
    "Left_Hip_Yaw": (-1.0, 1.0), "Left_Knee_Pitch": (0.0, 2.34),
    "Left_Ankle_Pitch": (-0.87, 0.35), "Left_Ankle_Roll": (-0.44, 0.44),
    "Right_Hip_Pitch": (-1.8, 1.57), "Right_Hip_Roll": (-1.57, 0.2),
    "Right_Hip_Yaw": (-1.0, 1.0), "Right_Knee_Pitch": (0.0, 2.34),
    "Right_Ankle_Pitch": (-0.87, 0.35), "Right_Ankle_Roll": (-0.44, 0.44),
}


class RetargetingImpl(Retargeting):

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "mapping.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.scale_factors = config.get("scale_factors", {"default": 0.8})

    def retarget(self, human_motion: MotionData, **kwargs) -> MotionData:
        if human_motion.positions is None:
            raise ValueError("human_motion.positions 不能为 None")

        positions = human_motion.positions  # (T, J, 3)
        T = positions.shape[0]
        scale = kwargs.get("scale", self.scale_factors.get("default", 0.8))

        name_to_idx = {name: i for i, name in enumerate(human_motion.joint_names)}

        # 第一遍：计算每帧原始角度
        raw_angles = np.zeros((T, 23), dtype=np.float32)
        for t in range(T):
            raw_angles[t] = self._compute_angles(positions[t], name_to_idx)

        # 中位数作为中性站立姿态
        neutral = np.median(raw_angles, axis=0)

        # 第二遍：分部位缩放（腿部幅度缩小，避免双脚离地）
        arm_scale = scale * 1.0      # 2-9 手臂
        body_scale = scale * 0.6     # 0-1, 10 头+躯干

        robot_angles = np.zeros((T, 23), dtype=np.float32)
        for t in range(T):
            delta = raw_angles[t] - neutral
            for j in range(23):
                if 11 <= j <= 22:
                    s = 0.0  # 腿部完全锁定，脚不离地
                elif 2 <= j <= 9:
                    s = arm_scale
                else:
                    s = body_scale
                robot_angles[t, j] = float(delta[j] * s)

        return MotionData(
            joint_names=list(ACTUATOR_NAMES),
            fps=human_motion.fps,
            num_frames=T,
            positions=None,
            angles=robot_angles,
            timestamps=human_motion.timestamps,
        )

    def _compute_angles(self, pos: np.ndarray, idx: Dict[str, int]) -> np.ndarray:
        """从一帧3D关键点计算23个执行器原始角度。"""
        angles = np.zeros(23)

        def p(name):
            return pos[idx[name]] if name in idx else np.zeros(3)

        # 中心点
        m_sh = (p("left_shoulder") + p("right_shoulder")) / 2
        m_hip = (p("left_hip") + p("right_hip")) / 2

        # ---- 头部 ----
        head = p("head") - m_sh
        angles[0] = np.arctan2(head[0], abs(head[2]) + 1e-3)
        angles[1] = np.arctan2(head[1], abs(head[2]) + 1e-3)

        # ---- 左臂 ----
        lu = p("left_elbow") - p("left_shoulder")
        lf = p("left_wrist") - p("left_elbow")
        angles[2] = np.arctan2(lu[1], abs(lu[2]) + 1e-3)   # Pitch
        angles[3] = np.arctan2(lu[0], abs(lu[2]) + 1e-3)   # Roll
        angles[4] = np.pi - self._angle_between(lu, lf)     # 肘: 直=0, 弯=正
        angles[5] = 0.0

        # ---- 右臂 ----
        ru = p("right_elbow") - p("right_shoulder")
        rf = p("right_wrist") - p("right_elbow")
        angles[6] = np.arctan2(ru[1], abs(ru[2]) + 1e-3)
        angles[7] = np.arctan2(ru[0], abs(ru[2]) + 1e-3)
        angles[8] = np.pi - self._angle_between(ru, rf)
        angles[9] = 0.0

        # ---- 躯干 ----
        torso = m_sh - m_hip
        angles[10] = np.arctan2(torso[0], abs(torso[2]) + 1e-3)

        # ---- 左腿 ----
        lt = p("left_knee") - p("left_hip")
        ls = p("left_ankle") - p("left_knee")
        angles[11] = np.arctan2(lt[1], abs(lt[2]) + 1e-3)       # Hip_Pitch
        angles[12] = np.arctan2(lt[0], abs(lt[2]) + 1e-3) * 0.5 # Hip_Roll
        angles[13] = 0.0                                          # Hip_Yaw
        # 膝角: π - 大腿小腿夹角, 伸直=0, 弯曲=正
        angles[14] = np.pi - self._angle_between(lt, ls)
        angles[15] = 0.0  # Ankle_Pitch
        angles[16] = 0.0  # Ankle_Roll

        # ---- 右腿 ----
        rt = p("right_knee") - p("right_hip")
        rs = p("right_ankle") - p("right_knee")
        angles[17] = np.arctan2(rt[1], abs(rt[2]) + 1e-3)
        angles[18] = np.arctan2(rt[0], abs(rt[2]) + 1e-3) * 0.5
        angles[19] = 0.0
        angles[20] = np.pi - self._angle_between(rt, rs)
        angles[21] = 0.0
        angles[22] = 0.0

        return angles

    @staticmethod
    def _angle_between(v1, v2):
        dot = np.dot(v1, v2)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        return float(np.arccos(np.clip(dot / (n1 * n2), -1.0, 1.0)))

    @staticmethod
    def _signed_angle(v1: np.ndarray, v2: np.ndarray, axis: np.ndarray) -> float:
        """带符号的3D夹角：v1和v2的夹角，符号由绕axis的旋转方向决定。"""
        dot = np.dot(v1, v2)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        angle = np.arccos(np.clip(dot / (n1 * n2), -1.0, 1.0))
        cross = np.cross(v1, v2)
        sign = 1.0 if np.dot(cross, axis) >= 0 else -1.0
        return float(angle * sign)

    def get_joint_limits(self) -> Dict[str, Tuple[float, float]]:
        return JOINT_LIMITS.copy()
