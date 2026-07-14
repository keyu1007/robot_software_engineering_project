"""
数据清洗模块

对原始姿态数据进行去噪、插值、平滑、异常值剔除。
"""

import numpy as np
from scipy import interpolate, signal

from common.interfaces import MotionCleaner
from common.motion_data import MotionData


class MotionCleanerImpl(MotionCleaner):
    """MotionCleaner 接口的实现类"""

    def clean(self, motion: MotionData, **kwargs) -> MotionData:
        """清洗原始动作数据"""
        if motion.positions is None:
            return motion

        smooth_window = kwargs.get("smooth_window", 5)
        interpolate_method = kwargs.get("interpolate_method", "linear")
        filter_type = kwargs.get("filter_type", "savgol")

        positions = motion.positions.copy()
        T, J, _ = positions.shape

        # 1. 检测异常值（阈值2.5，更敏感）
        outlier_mask = self.detect_outliers(motion, threshold=2.5)

        # 2. 对每个关节每个坐标轴进行插值修复
        for j in range(J):
            for axis in range(3):
                data = positions[:, j, axis].copy()
                outlier_indices = np.where(outlier_mask[:, j])[0]

                if len(outlier_indices) > 0:
                    data_with_nan = data.copy()
                    data_with_nan[outlier_indices] = np.nan

                    valid_mask = ~np.isnan(data_with_nan)
                    valid_indices = np.where(valid_mask)[0]
                    valid_values = data_with_nan[valid_mask]

                    if len(valid_indices) > 1:
                        interp_func = interpolate.interp1d(
                            valid_indices,
                            valid_values,
                            kind=interpolate_method,
                            fill_value="extrapolate",
                            bounds_error=False
                        )
                        data = interp_func(np.arange(T))
                    elif len(valid_indices) == 1:
                        data = np.full(T, valid_values[0])
                    else:
                        data = np.zeros(T)

                    positions[:, j, axis] = data

        # 3. 平滑滤波
        positions = self._smooth_positions(positions, smooth_window, filter_type)

        cleaned_motion = MotionData(
            joint_names=motion.joint_names.copy(),
            fps=motion.fps,
            num_frames=motion.num_frames,
            positions=positions,
            angles=motion.angles.copy() if motion.angles is not None else None,
            timestamps=motion.timestamps.copy() if motion.timestamps is not None else None,
        )

        return cleaned_motion

    def detect_outliers(self, motion: MotionData, threshold: float = 3.0) -> np.ndarray:
        """
        检测并标记异常帧/关节。

        使用三种方法联合检测：
        1. 滑动窗口局部 Z-score（检测局部突变）
        2. 全局 MAD（中位数绝对偏差）检测大幅跳变
        3. 绝对值阈值检测（防止微小波动被误判）
        """
        if motion.positions is None:
            return np.zeros((motion.num_frames, len(motion.joint_names)), dtype=bool)

        positions = motion.positions
        T, J, _ = positions.shape
        outlier_mask = np.zeros((T, J), dtype=bool)

        # 窗口大小：取总帧数的30%，至少7帧，最多50帧
        window_size = max(7, min(50, int(T * 0.3)))
        if window_size % 2 == 0:
            window_size += 1

        for j in range(J):
            for axis in range(3):
                data = positions[:, j, axis].copy()

                # 预处理NaN
                if not np.isfinite(data).all():
                    valid_mask = np.isfinite(data)
                    valid_indices = np.where(valid_mask)[0]
                    valid_values = data[valid_mask]
                    if len(valid_indices) >= 2:
                        interp_func = interpolate.interp1d(
                            valid_indices,
                            valid_values,
                            kind='linear',
                            fill_value='extrapolate',
                            bounds_error=False
                        )
                        data = interp_func(np.arange(T))
                    elif len(valid_indices) == 1:
                        data = np.full(T, valid_values[0])
                    else:
                        data = np.zeros(T)

                # ---- 方法1: 滑动窗口局部 Z-score ----
                velocity = np.diff(data, prepend=data[0])
                half_win = window_size // 2

                for t in range(T):
                    start = max(0, t - half_win)
                    end = min(T, t + half_win + 1)
                    window_vel = velocity[start:end]

                    if len(window_vel) < 3:
                        continue

                    mean = np.mean(window_vel)
                    std = np.std(window_vel)

                    if std > 1e-6:
                        z = abs(velocity[t] - mean) / std
                        if z > threshold:
                            outlier_mask[t, j] = True

                # ---- 方法2: 全局 MAD 检测大幅跳变 ----
                accel = np.diff(velocity, prepend=velocity[0])
                accel_median = np.median(accel)
                accel_mad = np.median(np.abs(accel - accel_median))

                # 最小跳变幅度阈值（避免微小波动被误判为异常）
                min_jump_threshold = 0.3

                if accel_mad > 1e-6:
                    for t in range(T):
                        jump_score = abs(accel[t] - accel_median) / accel_mad
                        # 检查实际数据变化是否超过最小阈值
                        if t > 0:
                            data_change = abs(data[t] - data[t-1])
                        else:
                            data_change = abs(data[t] - data[t+1]) if t < T - 1 else 0
                        if jump_score > 5.0 and data_change > min_jump_threshold:
                            outlier_mask[t, j] = True

                # ---- 方法3: 绝对值阈值检测（捕获极端异常值） ----
                # 如果数据绝对值超过 10.0，直接标记为异常（人体关节位置不可能超过10米）
                abs_threshold = 10.0
                for t in range(T):
                    if abs(data[t]) > abs_threshold:
                        outlier_mask[t, j] = True

        return outlier_mask

    def _smooth_positions(self, positions: np.ndarray, window: int, filter_type: str) -> np.ndarray:
        """对位置数据进行平滑滤波"""
        T, J, _ = positions.shape

        if T < window + 2 or T < 3:
            return positions

        smoothed = positions.copy()

        for j in range(J):
            for axis in range(3):
                data = positions[:, j, axis].copy()

                # 确保没有NaN
                if not np.isfinite(data).all():
                    valid_mask = np.isfinite(data)
                    valid_indices = np.where(valid_mask)[0]
                    valid_values = data[valid_mask]

                    if len(valid_indices) >= 2:
                        interp_func = interpolate.interp1d(
                            valid_indices,
                            valid_values,
                            kind='linear',
                            fill_value='extrapolate',
                            bounds_error=False
                        )
                        data = interp_func(np.arange(T))
                    elif len(valid_indices) == 1:
                        data = np.full(T, valid_values[0])
                    else:
                        data = np.zeros(T)

                # 平滑
                if filter_type == "savgol":
                    if len(data) >= window + 2:
                        win = window if window % 2 == 1 else window + 1
                        if win > len(data):
                            win = len(data) if len(data) % 2 == 1 else len(data) - 1
                        polyorder = min(2, win - 1)
                        if polyorder >= 1 and win >= polyorder + 2:
                            smoothed[:, j, axis] = signal.savgol_filter(
                                data,
                                window_length=win,
                                polyorder=polyorder
                            )
                        else:
                            smoothed[:, j, axis] = data
                    else:
                        smoothed[:, j, axis] = data
                elif filter_type == "butter":
                    if len(data) > 4:
                        b, a = signal.butter(4, 0.3, btype='low')
                        smoothed[:, j, axis] = signal.filtfilt(b, a, data)
                    else:
                        smoothed[:, j, axis] = data
                else:
                    kernel = np.ones(window) / window
                    smoothed[:, j, axis] = np.convolve(data, kernel, mode='same')

        return smoothed