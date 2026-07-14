"""
动作重定向模块 — 成员D (精确版)
精确计算每个关节的 Pitch、Roll、Yaw，直接输出与 MuJoCo actuator 顺序对齐的 23 个角度值。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from common.interfaces import Retargeting
from common.motion_data import MotionData


class RetargetingImpl(Retargeting):
    """动作重定向实现类 (精确版)"""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "mapping.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.parent_map: Dict[str, str] = config.get("parent_map", {})
        self.scale_factors: Dict[str, float] = config.get("scale_factors", {"default": 0.8})
        self.joint_limits: Dict[str, Tuple[float, float]] = config.get("joint_limits", {})

        # 加载 actuator 名称（与 t1.xml 顺序一致）
        self.actuator_names = config.get("actuator_names", [
            "AAHead_yaw", "Head_pitch",
            "Left_Shoulder_Pitch", "Left_Shoulder_Roll",
            "Left_Elbow_Pitch", "Left_Elbow_Yaw",
            "Right_Shoulder_Pitch", "Right_Shoulder_Roll",
            "Right_Elbow_Pitch", "Right_Elbow_Yaw",
            "Waist",
            "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
            "Left_Knee_Pitch",
            "Left_Ankle_Pitch", "Left_Ankle_Roll",
            "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
            "Right_Knee_Pitch",
            "Right_Ankle_Pitch", "Right_Ankle_Roll"
        ])

        self.num_actuators = len(self.actuator_names)

    def retarget(self, human_motion: MotionData, **kwargs) -> MotionData:
        if human_motion.positions is None:
            raise ValueError("human_motion.positions 不能为 None")

        positions = human_motion.positions
        T = positions.shape[0]
        scale = kwargs.get("scale", self.scale_factors.get("default", 0.8))

        joint_name_to_idx = {
            name: i for i, name in enumerate(human_motion.joint_names)
        }

        # 角度数组形状 (T, 23)
        robot_angles = np.zeros((T, self.num_actuators), dtype=np.float32)

        for t in range(T):
            frame_pos = positions[t]
            angles = self._compute_frame_angles(frame_pos, joint_name_to_idx, scale)
            robot_angles[t] = angles

        # 应用关节限位（按 actuator 名称）
        for i, name in enumerate(self.actuator_names):
            if name in self.joint_limits:
                min_val, max_val = self.joint_limits[name]
                robot_angles[:, i] = np.clip(robot_angles[:, i], min_val, max_val)

        return MotionData(
            joint_names=self.actuator_names,
            fps=human_motion.fps,
            num_frames=T,
            positions=None,
            angles=robot_angles,
            timestamps=human_motion.timestamps,
        )

    def _compute_frame_angles(
        self,
        pos: np.ndarray,
        idx_map: Dict[str, int],
        scale: float,
    ) -> np.ndarray:
        """
        计算单帧的 23 个 actuator 角度（精确版）
        返回顺序与 self.actuator_names 一致
        """
        angles = np.zeros(self.num_actuators, dtype=np.float32)

        def get_pos(joint: str) -> np.ndarray:
            return pos[idx_map[joint]] if joint in idx_map else np.zeros(3)

        def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
            """3D 向量夹角 (有符号)"""
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 < 1e-8 or norm2 < 1e-8:
                return 0.0
            cross = np.linalg.norm(np.cross(v1, v2))
            dot = np.dot(v1, v2)
            return np.arctan2(cross, dot)

        def angle_to_vertical(vec: np.ndarray, axis: str) -> float:
            """向量在指定平面与垂直方向 (0,1) 的夹角"""
            if axis == 'xz':
                v = np.array([vec[0], vec[2]])
            elif axis == 'yz':
                v = np.array([vec[1], vec[2]])
            elif axis == 'xy':
                v = np.array([vec[0], vec[1]])
            else:
                return 0.0
            vert = np.array([0, 1])
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-8:
                return 0.0
            return np.arctan2(v[0]*vert[1] - v[1]*vert[0], v[0]*vert[0] + v[1]*vert[1])

        # ---- 获取关键点 ----
        nose = get_pos('nose')
        left_shoulder = get_pos('left_shoulder')
        right_shoulder = get_pos('right_shoulder')
        left_elbow = get_pos('left_elbow')
        right_elbow = get_pos('right_elbow')
        left_wrist = get_pos('left_wrist')
        right_wrist = get_pos('right_wrist')
        left_hip = get_pos('left_hip')
        right_hip = get_pos('right_hip')
        left_knee = get_pos('left_knee')
        right_knee = get_pos('right_knee')
        left_ankle = get_pos('left_ankle')
        right_ankle = get_pos('right_ankle')

        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        spine = shoulder_center - hip_center
        neck_pos = shoulder_center

        # 缩放因子
        s = scale
        arm_s = s * 0.7
        leg_s = s * 0.8
        head_s = s * 0.5
        spine_s = s * 0.5

        # ----- 头部 (索引 0,1) -----
        head_vec = nose - neck_pos
        if np.linalg.norm(head_vec) > 1e-6:
            angles[0] = angle_to_vertical(head_vec, 'xz') * head_s   # AAHead_yaw
            angles[1] = angle_to_vertical(head_vec, 'yz') * head_s   # Head_pitch

        # ----- 躯干 (索引 10: Waist) -----
        if np.linalg.norm(spine) > 1e-6:
            angles[10] = angle_to_vertical(spine, 'xz') * spine_s    # Waist

        # ----- 左臂 -----
        if np.linalg.norm(left_shoulder - left_elbow) > 1e-6:
            upper_arm = left_elbow - left_shoulder
            forearm = left_wrist - left_elbow
            angles[2] = angle_to_vertical(upper_arm, 'yz') * arm_s   # Left_Shoulder_Pitch
            angles[3] = angle_to_vertical(upper_arm, 'xz') * arm_s   # Left_Shoulder_Roll
            angles[4] = angle_between(upper_arm, forearm) * arm_s    # Left_Elbow_Pitch
            angles[5] = angle_to_vertical(forearm, 'xz') * arm_s     # Left_Elbow_Yaw

        # ----- 右臂 -----
        if np.linalg.norm(right_shoulder - right_elbow) > 1e-6:
            upper_arm = right_elbow - right_shoulder
            forearm = right_wrist - right_elbow
            angles[6] = angle_to_vertical(upper_arm, 'yz') * arm_s   # Right_Shoulder_Pitch
            angles[7] = angle_to_vertical(upper_arm, 'xz') * arm_s   # Right_Shoulder_Roll
            angles[8] = angle_between(upper_arm, forearm) * arm_s    # Right_Elbow_Pitch
            angles[9] = angle_to_vertical(forearm, 'xz') * arm_s     # Right_Elbow_Yaw

        # ----- 左腿 -----
        if np.linalg.norm(left_hip - left_knee) > 1e-6:
            thigh = left_knee - left_hip
            shin = left_ankle - left_knee
            angles[11] = angle_to_vertical(thigh, 'yz') * leg_s      # Left_Hip_Pitch
            angles[12] = angle_to_vertical(thigh, 'xz') * leg_s      # Left_Hip_Roll
            angles[13] = angle_to_vertical(thigh, 'xy') * leg_s      # Left_Hip_Yaw
            angles[14] = angle_between(thigh, shin) * leg_s          # Left_Knee_Pitch
            angles[15] = angle_to_vertical(shin, 'yz') * leg_s       # Left_Ankle_Pitch
            angles[16] = angle_to_vertical(shin, 'xz') * leg_s       # Left_Ankle_Roll

        # ----- 右腿 -----
        if np.linalg.norm(right_hip - right_knee) > 1e-6:
            thigh = right_knee - right_hip
            shin = right_ankle - right_knee
            angles[17] = angle_to_vertical(thigh, 'yz') * leg_s      # Right_Hip_Pitch
            angles[18] = angle_to_vertical(thigh, 'xz') * leg_s      # Right_Hip_Roll
            angles[19] = angle_to_vertical(thigh, 'xy') * leg_s      # Right_Hip_Yaw
            angles[20] = angle_between(thigh, shin) * leg_s          # Right_Knee_Pitch
            angles[21] = angle_to_vertical(shin, 'yz') * leg_s       # Right_Ankle_Pitch
            angles[22] = angle_to_vertical(shin, 'xz') * leg_s       # Right_Ankle_Roll

        return angles

    def get_joint_limits(self) -> Dict[str, Tuple[float, float]]:
        """返回各关节的角度限位（弧度）"""
        return self.joint_limits.copy()