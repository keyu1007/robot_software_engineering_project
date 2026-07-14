"""
数据清洗模块 —— 单元测试
"""

import numpy as np
import pytest

from common.motion_data import MotionData, BOOSTER_T1_JOINT_NAMES
from project1_dance.motion_cleaner.cleaner import MotionCleanerImpl


class TestMotionCleaner:
    """MotionCleanerImpl 单元测试"""

    def setup_method(self):
        self.cleaner = MotionCleanerImpl()
        self.J = len(BOOSTER_T1_JOINT_NAMES)

    def _create_motion_data(self, positions, fps=30):
        T = positions.shape[0]
        return MotionData(
            joint_names=BOOSTER_T1_JOINT_NAMES.copy(),
            fps=fps,
            num_frames=T,
            positions=positions,
            angles=None,
            timestamps=np.arange(T) / fps,
        )

    # ============================================================
    # 测试 clean()
    # ============================================================

    def test_clean_no_outliers(self):
        """测试：输入无异常数据，输出应与输入一致"""
        T, J = 50, self.J
        positions = np.random.randn(T, J, 3) * 0.1
        motion = self._create_motion_data(positions)
        result = self.cleaner.clean(motion)
        assert result.positions is not None
        assert result.positions.shape == (T, J, 3)
        assert np.all(np.isfinite(result.positions))

    def test_clean_with_nan(self):
        """测试：输入含 NaN，输出无 NaN"""
        T, J = 50, self.J
        positions = np.random.randn(T, J, 3) * 0.1
        nan_indices = np.random.choice(T * J * 3, size=int(T * J * 3 * 0.05), replace=False)
        for idx in nan_indices:
            t = (idx // (J * 3)) % T
            j = (idx // 3) % J
            axis = idx % 3
            positions[t, j, axis] = np.nan
        motion = self._create_motion_data(positions)
        result = self.cleaner.clean(motion)
        assert result.positions is not None
        assert np.all(np.isfinite(result.positions))

    def test_clean_with_outliers(self):
        """测试：输入含异常跳变值，异常点被修复"""
        T, J = 50, self.J
        positions = np.random.randn(T, J, 3) * 0.1
        for j in range(min(J, 5)):
            positions[10, j, 0] = 50.0
            positions[20, j, 1] = -40.0
        motion = self._create_motion_data(positions)
        result = self.cleaner.clean(motion)
        assert result.positions is not None
        assert np.all(np.isfinite(result.positions))
        for j in range(min(J, 5)):
            assert abs(result.positions[10, j, 0]) < 5.0
            assert abs(result.positions[20, j, 1]) < 5.0

    def test_clean_only_one_valid_point(self):
        """修复验证：仅1帧有效数据，输出所有帧被填充为相同值"""
        T, J = 50, self.J
        positions = np.full((T, J, 3), np.nan)
        valid_value = 1.234
        positions[0, 0, 0] = valid_value
        motion = self._create_motion_data(positions)
        result = self.cleaner.clean(motion)
        assert result.positions is not None
        assert np.all(np.isfinite(result.positions))
        np.testing.assert_array_almost_equal(
            result.positions[:, 0, 0],
            np.full(T, valid_value),
            decimal=6
        )

    def test_clean_all_nan(self):
        """测试：所有帧全为 NaN，输出全零"""
        T, J = 50, self.J
        positions = np.full((T, J, 3), np.nan)
        motion = self._create_motion_data(positions)
        result = self.cleaner.clean(motion)
        assert result.positions is not None
        assert np.all(np.isfinite(result.positions))
        np.testing.assert_array_almost_equal(result.positions, np.zeros((T, J, 3)), decimal=6)

    def test_clean_with_small_window(self):
        """测试：窗口大小为1，不做平滑，但仍需处理异常"""
        T, J = 50, self.J
        positions = np.random.randn(T, J, 3) * 0.1
        positions[10, 0, 0] = 100.0
        motion = self._create_motion_data(positions)
        result = self.cleaner.clean(motion, smooth_window=1)
        assert result.positions is not None
        assert np.all(np.isfinite(result.positions))
        assert abs(result.positions[10, 0, 0]) < 10.0

    def test_clean_butter_filter(self):
        """测试：使用 Butterworth 滤波"""
        T, J = 50, self.J
        positions = np.random.randn(T, J, 3) * 0.1
        motion = self._create_motion_data(positions)
        result = self.cleaner.clean(motion, filter_type="butter")
        assert result.positions is not None
        assert np.all(np.isfinite(result.positions))

    # ============================================================
    # 测试 detect_outliers()
    # ============================================================

    def test_detect_outliers_no_outliers(self):
        """测试：无异常数据，允许少量误报（< 1%）"""
        T, J = 50, self.J
        positions = np.random.randn(T, J, 3) * 0.1
        motion = self._create_motion_data(positions)
        mask = self.cleaner.detect_outliers(motion)
        assert mask.shape == (T, J)
        assert mask.dtype == bool
        # 允许少量误报（< 1% 的点被标记为异常）
        outlier_ratio = np.sum(mask) / (T * J)
        assert outlier_ratio < 0.01, f"误报率过高: {outlier_ratio*100:.2f}%"

    def test_detect_outliers_with_nan(self):
        """测试：输入含 NaN，detect_outliers 仍能正常执行"""
        T, J = 50, self.J
        positions = np.random.randn(T, J, 3) * 0.1
        positions[10, 0, 0] = np.nan
        motion = self._create_motion_data(positions)
        mask = self.cleaner.detect_outliers(motion)
        assert mask.shape == (T, J)
        assert mask.dtype == bool

    def test_detect_outliers_has_outliers(self):
        """修复验证：含大幅跳变数据，异常点应被检测到"""
        T, J = 50, self.J
        positions = np.random.randn(T, J, 3) * 0.1
        positions[10, 0, 0] = 50.0
        positions[20, 0, 0] = -40.0
        motion = self._create_motion_data(positions)
        mask = self.cleaner.detect_outliers(motion, threshold=3.0)
        assert mask.shape == (T, J)
        assert mask[10, 0] == True, "第10帧异常未被检测"
        assert mask[20, 0] == True, "第20帧异常未被检测"

    # ============================================================
    # 测试集成场景
    # ============================================================

    def test_integration_mock_data(self):
        """测试：用 mock 数据测试完整清洗流程"""
        T, J = 100, self.J
        positions = np.random.randn(T, J, 3) * 0.2
        positions[5:10, 0, :] = np.nan
        positions[30, 1, 0] = 100.0
        positions[40, 2, 1] = -80.0
        motion = self._create_motion_data(positions)
        result = self.cleaner.clean(motion, smooth_window=5, filter_type="savgol")
        assert result.positions is not None
        assert np.all(np.isfinite(result.positions))
        assert result.positions.shape == (T, J, 3)
        assert abs(result.positions[30, 1, 0]) < 10.0
        assert abs(result.positions[40, 2, 1]) < 10.0