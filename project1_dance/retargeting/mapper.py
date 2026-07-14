"""
动作重定向模块 — 成员D（最终完整版）

实现功能：
1. 坐标系对齐：MediaPipe (Y-up) → T1 (Z-up)
2. 解析IK：余弦定理精确计算肘、膝关节角度
3. 关节映射表：从 YAML 读取，灵活配置
4. 相位展开（np.unwrap）：消除角度跳变
5. 帧间约束：限制每帧角度变化率，防止抖动
6. 输出平滑：Savitzky-Golay 滤波
7. 关键帧校准：对齐人体中立站姿与机器人 home
8. 关节限位：保护执行器
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
    """动作重定向实现类（最终完整版）"""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "mapping.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 加载配置
        self.scale_factors: Dict[str, float] = config.get("scale_factors", {"default": 0.8})
        self.joint_limits: Dict[str, Tuple[float, float]] = config.get("joint_limits", {})
        self.actuator_names = config.get("mujoco_actuator_order", [])
        self.joint_mapping: Dict[str, Dict] = config.get("joint_mapping", {})
        self.num_actuators = len(self.actuator_names)

        # 帧间约束参数
        self.max_frame_change = config.get("max_frame_change", 0.3)
        self.smooth_window = config.get("smooth_window", 11)
        self.smooth_polyorder = config.get("smooth_polyorder", 3)
        self.enable_keyframe_calibration = config.get("enable_keyframe_calibration", True)

        # 解析IK肢体长度（可根据实际数据调整）
        self.upper_arm_length = 0.28
        self.forearm_length = 0.26
        self.thigh_length = 0.42
        self.shin_length = 0.40

    # ============================================================
    # 坐标系对齐
    # ============================================================
    def _rotate_to_t1(self, pos: np.ndarray) -> np.ndarray:
        """MediaPipe (Y-up) → T1 (Z-up)"""
        R = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        return R @ pos

    def _get_keypoints(self, pos: np.ndarray, idx_map: Dict[str, int]) -> Dict[str, np.ndarray]:
        """获取关键点并自动转换坐标系"""
        keypoints = {}
        for name, idx in idx_map.items():
            if idx < len(pos):
                keypoints[name] = self._rotate_to_t1(pos[idx])
            else:
                keypoints[name] = np.zeros(3)
        return keypoints

    # ============================================================
    # 解析IK（余弦定理）
    # ============================================================
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
    # 单帧角度计算（基于映射表）
    # ============================================================
    def _compute_frame_angles(self, pos: np.ndarray, idx_map: Dict[str, int], scale: float) -> np.ndarray:
        angles = np.zeros(self.num_actuators, dtype=np.float32)
        kps = self._get_keypoints(pos, idx_map)

        # 按映射表计算每个 actuator
        for i, actuator_name in enumerate(self.actuator_names):
            if actuator_name in self.joint_mapping:
                mapping = self.joint_mapping[actuator_name]
                source = mapping.get("source", [])
                angle_type = mapping.get("type", "pitch")
                # 获取关键点坐标
                pts = [kps.get(name, np.zeros(3)) for name in source]
                # 根据 source 长度和类型计算角度
                if len(source) == 0:
                    angle = 0.0
                elif len(source) == 1:
                    vec = pts[0]
                    if angle_type == "pitch":
                        angle = np.arctan2(vec[1], vec[2])
                    elif angle_type == "roll":
                        angle = np.arctan2(vec[0], vec[2])
                    else:
                        angle = np.arctan2(vec[1], vec[0])
                elif len(source) == 2:
                    vec = pts[1] - pts[0]
                    if angle_type == "pitch":
                        angle = np.arctan2(vec[1], vec[2])
                    elif angle_type == "roll":
                        angle = np.arctan2(vec[0], vec[2])
                    else:
                        angle = np.arctan2(vec[1], vec[0])
                elif len(source) == 3:
                    v1 = pts[1] - pts[0]
                    v2 = pts[2] - pts[1]
                    # 用余弦定理求肘关节角度（特殊处理）
                    if "Elbow" in actuator_name and "Pitch" in actuator_name:
                        angle = self._solve_elbow_angle(pts[0], pts[1], pts[2])
                    elif "Knee" in actuator_name and "Pitch" in actuator_name:
                        angle = self._solve_knee_angle(pts[0], pts[1], pts[2])
                    else:
                        # 一般三点：计算两个向量在指定平面的夹角
                        if angle_type == "pitch":
                            a1 = np.array([v1[1], v1[2]])
                            a2 = np.array([v2[1], v2[2]])
                        elif angle_type == "roll":
                            a1 = np.array([v1[0], v1[2]])
                            a2 = np.array([v2[0], v2[2]])
                        else:
                            a1 = np.array([v1[0], v1[1]])
                            a2 = np.array([v2[0], v2[1]])
                        angle = np.arctan2(
                            a1[0]*a2[1] - a1[1]*a2[0],
                            a1[0]*a2[0] + a1[1]*a2[1]
                        )
                elif len(source) == 4:
                    center1 = (pts[0] + pts[1]) / 2
                    center2 = (pts[2] + pts[3]) / 2
                    vec = center2 - center1
                    if angle_type == "pitch":
                        angle = np.arctan2(vec[1], vec[2])
                    elif angle_type == "roll":
                        angle = np.arctan2(vec[0], vec[2])
                    else:
                        angle = np.arctan2(vec[1], vec[0])
                else:
                    angle = 0.0

                # 应用缩放
                scale_factor = self.scale_factors.get("default", 0.8)
                angles[i] = angle * scale * scale_factor
            else:
                angles[i] = 0.0

        return angles

    # ============================================================
    # 主入口 retarget
    # ============================================================
    def retarget(self, human_motion: MotionData, **kwargs) -> MotionData:
        if human_motion.positions is None:
            raise ValueError("human_motion.positions 不能为 None")

        positions = human_motion.positions
        T = positions.shape[0]
        scale = kwargs.get("scale", self.scale_factors.get("default", 0.8))

        idx_map = {name: i for i, name in enumerate(human_motion.joint_names)}

        # ---- 1. 逐帧计算原始角度 ----
        raw_angles = np.zeros((T, self.num_actuators), dtype=np.float32)
        for t in range(T):
            raw_angles[t] = self._compute_frame_angles(positions[t], idx_map, scale)

        # ---- 2. 相位展开（角度连续性修复） ----
        unwrapped_angles = np.unwrap(raw_angles, axis=0)

        # ---- 3. 帧间约束（限制每帧变化率） ----
        constrained_angles = unwrapped_angles.copy()
        if T > 1:
            max_change = self.max_frame_change
            for t in range(1, T):
                diff = constrained_angles[t] - constrained_angles[t-1]
                clipped_diff = np.clip(diff, -max_change, max_change)
                constrained_angles[t] = constrained_angles[t-1] + clipped_diff

        # ---- 4. 关键帧校准（对齐第一帧到 home） ----
        if self.enable_keyframe_calibration and T > 0:
            offset = constrained_angles[0].copy()
            calibrated_angles = constrained_angles - offset
        else:
            calibrated_angles = constrained_angles

        # ---- 5. 关节限位 ----
        for i, name in enumerate(self.actuator_names):
            if name in self.joint_limits:
                min_val, max_val = self.joint_limits[name]
                calibrated_angles[:, i] = np.clip(calibrated_angles[:, i], min_val, max_val)

        # ---- 6. 输出平滑（Savitzky-Golay） ----
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

        # ---- 7. 返回 MotionData ----
        return MotionData(
            joint_names=self.actuator_names,
            fps=human_motion.fps,
            num_frames=T,
            positions=None,
            angles=smoothed_angles,
            timestamps=human_motion.timestamps,
        )

    # ============================================================
    # 接口方法
    # ============================================================
    def get_joint_limits(self) -> Dict[str, Tuple[float, float]]:
        return self.joint_limits.copy()