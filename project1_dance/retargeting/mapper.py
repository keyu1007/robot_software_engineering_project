"""
动作重定向模块 — 成员D（优化版）

新增优化：
1. 相位展开（np.unwrap）：消除 -π/π 边界跳变
2. 帧间约束：限制每帧角度变化率，防止突变
3. 降低腿部缩放因子：让机器人更稳定
4. 增大平滑窗口：减少抖动
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
    """动作重定向实现类（优化版）"""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "mapping.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.scale_factors: Dict[str, float] = config.get("scale_factors", {"default": 0.8})
        self.joint_limits: Dict[str, Tuple[float, float]] = config.get("joint_limits", {})

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

        # ---- 优化参数 ----
        self.smooth_window = config.get("smooth_window", 15)
        self.smooth_polyorder = config.get("smooth_polyorder", 3)
        self.enable_keyframe_calibration = config.get("enable_keyframe_calibration", False)
        self.max_frame_change = config.get("max_frame_change", 0.15)

        # 解析IK肢体长度
        self.upper_arm_length = 0.28
        self.forearm_length = 0.26
        self.thigh_length = 0.42
        self.shin_length = 0.40

    def _rotate_to_t1(self, pos: np.ndarray) -> np.ndarray:
        R = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        return R @ pos

    def _get_keypoints(self, pos: np.ndarray, idx_map: Dict[str, int]) -> Dict[str, np.ndarray]:
        keypoints = {}
        for name, idx in idx_map.items():
            if idx < len(pos):
                keypoints[name] = self._rotate_to_t1(pos[idx])
            else:
                keypoints[name] = np.zeros(3)
        return keypoints

    def _solve_elbow_angle(self, shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float:
        upper = elbow - shoulder
        forearm = wrist - elbow
        len_upper = np.linalg.norm(upper)
        len_forearm = np.linalg.norm(forearm)
        shoulder_to_wrist = np.linalg.norm(wrist - shoulder)
        if len_upper < 1e-8 or len_forearm < 1e-8:
            return 0.0
        c = np.clip(shoulder_to_wrist, abs(len_upper - len_forearm) + 0.001, len_upper + len_forearm - 0.001)
        cos_angle = (len_upper*len_upper + len_forearm*len_forearm - c*c) / (2 * len_upper * len_forearm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)

    def _solve_knee_angle(self, hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> float:
        thigh = knee - hip
        shin = ankle - knee
        len_thigh = np.linalg.norm(thigh)
        len_shin = np.linalg.norm(shin)
        hip_to_ankle = np.linalg.norm(ankle - hip)
        if len_thigh < 1e-8 or len_shin < 1e-8:
            return 0.0
        c = np.clip(hip_to_ankle, abs(len_thigh - len_shin) + 0.001, len_thigh + len_shin - 0.001)
        cos_angle = (len_thigh*len_thigh + len_shin*len_shin - c*c) / (2 * len_thigh * len_shin)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)

    def _compute_direction_angle(self, vec: np.ndarray, axis: str) -> float:
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

    def _compute_frame_angles(self, pos: np.ndarray, idx_map: Dict[str, int], scale: float) -> np.ndarray:
        angles = np.zeros(self.num_actuators, dtype=np.float32)
        kps = self._get_keypoints(pos, idx_map)

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
        leg_s = s * self.scale_factors.get('leg', 0.5)   # 优化：降低腿部幅度
        head_s = s * self.scale_factors.get('head', 0.5)
        spine_s = s * self.scale_factors.get('spine', 0.5)

        # ---- 头部 ----
        head_vec = nose - shoulder_center
        if np.linalg.norm(head_vec) > 1e-6:
            angles[0] = self._compute_direction_angle(head_vec, 'xz') * head_s
            angles[1] = self._compute_direction_angle(head_vec, 'yz') * head_s

        # ---- 躯干 ----
        if np.linalg.norm(spine) > 1e-6:
            angles[10] = self._compute_direction_angle(spine, 'xz') * spine_s

        # ---- 左臂 ----
        if np.linalg.norm(left_shoulder - left_elbow) > 1e-6:
            upper_arm = left_elbow - left_shoulder
            angles[2] = self._compute_direction_angle(upper_arm, 'yz') * arm_s
            angles[3] = self._compute_direction_angle(upper_arm, 'xz') * arm_s
            angles[4] = self._solve_elbow_angle(left_shoulder, left_elbow, left_wrist) * arm_s
            forearm = left_wrist - left_elbow
            angles[5] = self._compute_direction_angle(forearm, 'xz') * arm_s

        # ---- 右臂 ----
        if np.linalg.norm(right_shoulder - right_elbow) > 1e-6:
            upper_arm = right_elbow - right_shoulder
            angles[6] = self._compute_direction_angle(upper_arm, 'yz') * arm_s
            angles[7] = self._compute_direction_angle(upper_arm, 'xz') * arm_s
            angles[8] = self._solve_elbow_angle(right_shoulder, right_elbow, right_wrist) * arm_s
            forearm = right_wrist - right_elbow
            angles[9] = self._compute_direction_angle(forearm, 'xz') * arm_s

        # ---- 左腿 ----
        if np.linalg.norm(left_hip - left_knee) > 1e-6:
            thigh = left_knee - left_hip
            angles[11] = self._compute_direction_angle(thigh, 'yz') * leg_s
            angles[12] = self._compute_direction_angle(thigh, 'xz') * leg_s
            angles[13] = self._compute_direction_angle(thigh, 'xy') * leg_s
            angles[14] = self._solve_knee_angle(left_hip, left_knee, left_ankle) * leg_s
            shin = left_ankle - left_knee
            angles[15] = self._compute_direction_angle(shin, 'yz') * leg_s
            angles[16] = self._compute_direction_angle(shin, 'xz') * leg_s

        # ---- 右腿 ----
        if np.linalg.norm(right_hip - right_knee) > 1e-6:
            thigh = right_knee - right_hip
            angles[17] = self._compute_direction_angle(thigh, 'yz') * leg_s
            angles[18] = self._compute_direction_angle(thigh, 'xz') * leg_s
            angles[19] = self._compute_direction_angle(thigh, 'xy') * leg_s
            angles[20] = self._solve_knee_angle(right_hip, right_knee, right_ankle) * leg_s
            shin = right_ankle - right_knee
            angles[21] = self._compute_direction_angle(shin, 'yz') * leg_s
            angles[22] = self._compute_direction_angle(shin, 'xz') * leg_s

        return angles

    def retarget(self, human_motion: MotionData, **kwargs) -> MotionData:
        if human_motion.positions is None:
            raise ValueError("human_motion.positions 不能为 None")

        positions = human_motion.positions
        T = positions.shape[0]
        scale = kwargs.get("scale", self.scale_factors.get("default", 0.8))

        idx_map = {name: i for i, name in enumerate(human_motion.joint_names)}

        # ---- 腿长尺度校准 + 足部接地高度计算 ----
        # 机器人真实腿长（大腿+小腿）
        robot_leg_length = self.thigh_length + self.shin_length
        
        human_leg_lengths = []
        ankle_heights = []  # 记录所有帧脚踝的垂直高度（T1坐标系z轴为向上方向）
        
        for t in range(T):
            pos = positions[t]
            kps = self._get_keypoints(pos, idx_map)
            
            left_hip = kps['left_hip']
            left_ankle = kps['left_ankle']
            right_hip = kps['right_hip']
            right_ankle = kps['right_ankle']
            
            # 统计人体腿长
            left_len = np.linalg.norm(left_ankle - left_hip)
            right_len = np.linalg.norm(right_ankle - right_hip)
            human_leg_lengths.append(left_len)
            human_leg_lengths.append(right_len)
            
            # 记录脚踝高度（z轴为垂直向上方向）
            ankle_heights.append(left_ankle[2])
            ankle_heights.append(right_ankle[2])
        
        # 计算尺度修正系数：让人体腿长匹配机器人真实腿长
        avg_human_leg = np.mean(human_leg_lengths)
        # 2D恢复的关键点无真实米单位，禁用自动腿长缩放，保留手动经验比例
        leg_scale = 1.0
        # scale 保持原配置值不变
        # 计算全局最低脚踝高度，用于地面对齐
        min_ankle_z = np.min(ankle_heights)
        # 根节点垂直偏移量：让最低的脚踝刚好落在地面 z=0 处
        root_z_offset = -min_ankle_z
        print(f"[Retargeting] 腿长校准系数: {leg_scale:.3f}, 根节点接地偏移: {root_z_offset:.3f}m")

        # ---- 逐帧计算原始角度 ----
        raw_angles = np.zeros((T, self.num_actuators), dtype=np.float32)
        for t in range(T):
            raw_angles[t] = self._compute_frame_angles(positions[t], idx_map, scale)

        # ---- 1. 相位展开（消除 -π/π 边界跳变） ----
        unwrapped_angles = np.unwrap(raw_angles, axis=0)

        # ---- 2. 帧间约束（限制每帧变化率） ----
        constrained_angles = unwrapped_angles.copy()
        if T > 1:
            max_change = self.max_frame_change
            for t in range(1, T):
                diff = constrained_angles[t] - constrained_angles[t-1]
                clipped_diff = np.clip(diff, -max_change, max_change)
                constrained_angles[t] = constrained_angles[t-1] + clipped_diff

        # ---- 3. 关键帧校准（可选，默认关闭） ----
        if self.enable_keyframe_calibration and T > 0:
            offset = constrained_angles[0].copy()
            calibrated_angles = constrained_angles - offset
        else:
            calibrated_angles = constrained_angles

        # ---- 4. 关节限位 ----
        for i, name in enumerate(self.actuator_names):
            if name in self.joint_limits:
                min_val, max_val = self.joint_limits[name]
                calibrated_angles[:, i] = np.clip(calibrated_angles[:, i], min_val, max_val)

        # ---- 5. 输出平滑（Savitzky-Golay） ----
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