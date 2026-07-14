"""
动作重定向模块 — 成员D（含解析IK + 时序平滑 + 关键帧校准）

解析IK：用余弦定理精确计算肘关节和膝关节的弯曲角度，
替代原来的 arctan2 向量夹角估算，精度更高。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from scipy.signal import savgol_filter

from common.interfaces import Retargeting
from common.motion_data import MotionData


class RetargetingImpl(Retargeting):
    """动作重定向实现类（含解析IK）"""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "mapping.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.scale_factors: Dict[str, float] = config.get("scale_factors", {"default": 0.8})
        self.joint_limits: Dict[str, Tuple[float, float]] = config.get("joint_limits", {})

        # 加载 actuator 名称（与 t1.xml 顺序一致）
        self.actuator_names = config.get("actuator_names", config.get("mujoco_actuator_order", [
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
        ]))
        self.joint_mapping: Dict[str, Dict] = config.get("joint_mapping", {})
        self.num_actuators = len(self.actuator_names)

        # 平滑参数
        self.smooth_window = config.get("smooth_window", 11)
        self.smooth_polyorder = config.get("smooth_polyorder", 3)
        self.enable_keyframe_calibration = config.get("enable_keyframe_calibration", True)

        # ---- 解析IK所需的肢体长度（单位：米，MediaPipe坐标） ----
        # 这些值可以根据实际人体比例调整
        self.upper_arm_length = 0.28   # 上臂长度
        self.forearm_length = 0.26     # 前臂长度
        self.thigh_length = 0.42       # 大腿长度
        self.shin_length = 0.40        # 小腿长度

    def _rotate_to_t1(self, pos: np.ndarray) -> np.ndarray:
        """MediaPipe (Y-up) → T1 (Z-up)"""
        R = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        return R @ pos

    def _get_keypoints(self, pos: np.ndarray, idx_map: Dict[str, int]) -> Dict[str, np.ndarray]:
        """获取所有关键点（自动坐标系转换）"""
        keypoints = {}
        for name, idx in idx_map.items():
            if idx < len(pos):
                keypoints[name] = self._rotate_to_t1(pos[idx])
            else:
                keypoints[name] = np.zeros(3)
        return keypoints

    # ============================================================
    # 解析IK：余弦定理求肘关节角度
    # ============================================================
    def _solve_elbow_angle(self, shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float:
        """
        用余弦定理计算肘关节弯曲角度（0~π）
        输入：肩、肘、腕的三维坐标（已转换到T1坐标系）
        返回：肘关节弯曲角（弧度）
        """
        # 上臂向量：肩→肘
        upper = elbow - shoulder
        # 前臂向量：肘→腕
        forearm = wrist - elbow

        # 实际测量的上臂和前臂长度
        len_upper = np.linalg.norm(upper)
        len_forearm = np.linalg.norm(forearm)

        # 肩到腕的距离
        shoulder_to_wrist = np.linalg.norm(wrist - shoulder)

        # 余弦定理：cos(肘角) = (a² + b² - c²) / (2ab)
        # 其中 a = 上臂长度, b = 前臂长度, c = 肩到腕距离
        a = len_upper
        b = len_forearm
        c = shoulder_to_wrist

        if a < 1e-8 or b < 1e-8:
            return 0.0

        # 限制 c 在有效范围内
        c = np.clip(c, abs(a - b) + 0.001, a + b - 0.001)

        cos_angle = (a*a + b*b - c*c) / (2 * a * b)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        return np.arccos(cos_angle)

    # ============================================================
    # 解析IK：余弦定理求膝关节角度
    # ============================================================
    def _solve_knee_angle(self, hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> float:
        """
        用余弦定理计算膝关节弯曲角度（0~π）
        输入：髋、膝、踝的三维坐标（已转换到T1坐标系）
        返回：膝关节弯曲角（弧度）
        """
        thigh = knee - hip
        shin = ankle - knee

        len_thigh = np.linalg.norm(thigh)
        len_shin = np.linalg.norm(shin)

        hip_to_ankle = np.linalg.norm(ankle - hip)

        a = len_thigh
        b = len_shin
        c = hip_to_ankle

        if a < 1e-8 or b < 1e-8:
            return 0.0

        c = np.clip(c, abs(a - b) + 0.001, a + b - 0.001)

        cos_angle = (a*a + b*b - c*c) / (2 * a * b)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        return np.arccos(cos_angle)

    # ============================================================
    # 方向角计算（用于肩、髋的朝向）
    # ============================================================
    def _compute_direction_angle(self, vec: np.ndarray, axis: str) -> float:
        """计算向量在指定平面与垂直方向的夹角（有符号）"""
        if np.linalg.norm(vec) < 1e-8:
            return 0.0

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

    # ============================================================
    # 核心：单帧角度计算
    # ============================================================
    def _compute_frame_angles(self, pos: np.ndarray, idx_map: Dict[str, int], scale: float) -> np.ndarray:
        """计算单帧的23个actuator角度（含解析IK）"""
        angles = np.zeros(self.num_actuators, dtype=np.float32)

        # 获取关键点（已转换到T1坐标系）
        kps = self._get_keypoints(pos, idx_map)

        # ---- 提取关键点坐标 ----
        nose = kps.get('nose', np.zeros(3))
        left_shoulder = kps.get('left_shoulder', np.zeros(3))
        right_shoulder = kps.get('right_shoulder', np.zeros(3))
        left_elbow = kps.get('left_elbow', np.zeros(3))
        right_elbow = kps.get('right_elbow', np.zeros(3))
        left_wrist = kps.get('left_wrist', np.zeros(3))
        right_wrist = kps.get('right_wrist', np.zeros(3))
        left_hip = kps.get('left_hip', np.zeros(3))
        right_hip = kps.get('right_hip', np.zeros(3))
        left_knee = kps.get('left_knee', np.zeros(3))
        right_knee = kps.get('right_knee', np.zeros(3))
        left_ankle = kps.get('left_ankle', np.zeros(3))
        right_ankle = kps.get('right_ankle', np.zeros(3))

        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        spine = shoulder_center - hip_center

        s = scale
        arm_s = s * self.scale_factors.get('arm', 0.7)
        leg_s = s * self.scale_factors.get('leg', 0.8)
        head_s = s * self.scale_factors.get('head', 0.5)
        spine_s = s * self.scale_factors.get('spine', 0.5)

        # ---- 头部 ----
        head_vec = nose - shoulder_center
        if np.linalg.norm(head_vec) > 1e-6:
            angles[0] = self._compute_direction_angle(head_vec, 'xz') * head_s   # AAHead_yaw
            angles[1] = self._compute_direction_angle(head_vec, 'yz') * head_s   # Head_pitch

        # ---- 躯干 ----
        if np.linalg.norm(spine) > 1e-6:
            angles[10] = self._compute_direction_angle(spine, 'xz') * spine_s    # Waist

        # ---- 左臂（解析IK计算肘关节） ----
        if np.linalg.norm(left_shoulder - left_elbow) > 1e-6:
            upper_arm = left_elbow - left_shoulder
            # 肩关节方向
            angles[2] = self._compute_direction_angle(upper_arm, 'yz') * arm_s   # Left_Shoulder_Pitch
            angles[3] = self._compute_direction_angle(upper_arm, 'xz') * arm_s   # Left_Shoulder_Roll
            # 肘关节（余弦定理）
            elbow_angle = self._solve_elbow_angle(left_shoulder, left_elbow, left_wrist)
            angles[4] = elbow_angle * arm_s                                       # Left_Elbow_Pitch
            # 肘部扭角（简化：用前臂的水平投影）
            forearm = left_wrist - left_elbow
            angles[5] = self._compute_direction_angle(forearm, 'xz') * arm_s     # Left_Elbow_Yaw

        # ---- 右臂 ----
        if np.linalg.norm(right_shoulder - right_elbow) > 1e-6:
            upper_arm = right_elbow - right_shoulder
            angles[6] = self._compute_direction_angle(upper_arm, 'yz') * arm_s   # Right_Shoulder_Pitch
            angles[7] = self._compute_direction_angle(upper_arm, 'xz') * arm_s   # Right_Shoulder_Roll
            elbow_angle = self._solve_elbow_angle(right_shoulder, right_elbow, right_wrist)
            angles[8] = elbow_angle * arm_s                                      # Right_Elbow_Pitch
            forearm = right_wrist - right_elbow
            angles[9] = self._compute_direction_angle(forearm, 'xz') * arm_s     # Right_Elbow_Yaw

        # ---- 左腿（解析IK计算膝关节） ----
        if np.linalg.norm(left_hip - left_knee) > 1e-6:
            thigh = left_knee - left_hip
            angles[11] = self._compute_direction_angle(thigh, 'yz') * leg_s      # Left_Hip_Pitch
            angles[12] = self._compute_direction_angle(thigh, 'xz') * leg_s      # Left_Hip_Roll
            angles[13] = self._compute_direction_angle(thigh, 'xy') * leg_s      # Left_Hip_Yaw
            knee_angle = self._solve_knee_angle(left_hip, left_knee, left_ankle)
            angles[14] = knee_angle * leg_s                                      # Left_Knee_Pitch
            # 踝关节
            shin = left_ankle - left_knee
            angles[15] = self._compute_direction_angle(shin, 'yz') * leg_s       # Left_Ankle_Pitch
            angles[16] = self._compute_direction_angle(shin, 'xz') * leg_s       # Left_Ankle_Roll

        # ---- 右腿 ----
        if np.linalg.norm(right_hip - right_knee) > 1e-6:
            thigh = right_knee - right_hip
            angles[17] = self._compute_direction_angle(thigh, 'yz') * leg_s      # Right_Hip_Pitch
            angles[18] = self._compute_direction_angle(thigh, 'xz') * leg_s      # Right_Hip_Roll
            angles[19] = self._compute_direction_angle(thigh, 'xy') * leg_s      # Right_Hip_Yaw
            knee_angle = self._solve_knee_angle(right_hip, right_knee, right_ankle)
            angles[20] = knee_angle * leg_s                                      # Right_Knee_Pitch
            shin = right_ankle - right_knee
            angles[21] = self._compute_direction_angle(shin, 'yz') * leg_s       # Right_Ankle_Pitch
            angles[22] = self._compute_direction_angle(shin, 'xz') * leg_s       # Right_Ankle_Roll

        return angles

    # ============================================================
    # retarget：主入口
    # ============================================================
    def retarget(self, human_motion: MotionData, **kwargs) -> MotionData:
        if human_motion.positions is None:
            raise ValueError("human_motion.positions 不能为 None")

        positions = human_motion.positions
        T = positions.shape[0]
        scale = kwargs.get("scale", self.scale_factors.get("default", 0.8))

        joint_name_to_idx = {
            name: i for i, name in enumerate(human_motion.joint_names)
        }

        raw_angles = np.zeros((T, self.num_actuators), dtype=np.float32)

        for t in range(T):
            frame_pos = positions[t]
            angles = self._compute_frame_angles(frame_pos, joint_name_to_idx, scale)
            raw_angles[t] = angles

        # ---- 关键帧校准 ----
        if self.enable_keyframe_calibration and T > 0:
            offset = raw_angles[0].copy()
            calibrated_angles = raw_angles - offset
        else:
            calibrated_angles = raw_angles

        # ---- 关节限位 ----
        for i, name in enumerate(self.actuator_names):
            if name in self.joint_limits:
                min_val, max_val = self.joint_limits[name]
                calibrated_angles[:, i] = np.clip(calibrated_angles[:, i], min_val, max_val)

        # ---- 时序平滑 ----
        if T >= self.smooth_window:
            window = self.smooth_window
            if window % 2 == 0:
                window += 1
            smoothed_angles = savgol_filter(
                calibrated_angles,
                window_length=window,
                polyorder=self.smooth_polyorder,
                axis=0
            )
        else:
            smoothed_angles = calibrated_angles

        return MotionData(
            joint_names=self.actuator_names,
            fps=human_motion.fps,
            num_frames=T,
            positions=None,
            angles=smoothed_angles,
            timestamps=human_motion.timestamps,
        )

    def get_joint_limits(self) -> Dict[str, Tuple[float, float]]:
        return self.joint_limits.copy()