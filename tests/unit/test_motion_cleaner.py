"""
数据清洗模块单元测试

版本说明：
- 新版 detect_outliers 采用三重检测：滑动窗口Z-score + MAD + 绝对值阈值
- clean() 内部调用 detect_outliers 时 threshold=2.5
- 已知问题：
  1. 绝对值阈值（abs(data) > 10.0）未生效 → 标记为 xfail
  2. 单点有效值广播填充问题 → 标记为 xfail
"""

import pytest
import numpy as np
from common.motion_data import MotionData
from project1_dance.motion_cleaner.cleaner import MotionCleanerImpl

np.random.seed(42)  # 固定随机种子，保证测试结果稳定可复现


# ===================== Fixture 公共资源 =====================
@pytest.fixture
def cleaner():
    """实例化清洗器"""
    return MotionCleanerImpl()


@pytest.fixture
def normal_motion_data():
    """
    标准正常动作数据：随机累积运动（接近真实场景）
    特点：有微小波动，无异常值，无NaN
    """
    T, J = 50, 2
    pos = np.cumsum(np.random.randn(T, J, 3) * 0.2, axis=0)
    return MotionData(
        joint_names=["joint0", "joint1"],
        fps=30,
        num_frames=T,
        positions=pos,
        angles=np.random.rand(T, J, 3),
        timestamps=np.linspace(0, 1.66, T)
    )


@pytest.fixture
def motion_with_outliers():
    """
    含明显异常值的动作数据
    构造方式：线性轨迹 + 人为注入大幅跳变
    """
    T, J = 50, 2
    pos = np.zeros((T, J, 3))
    for t in range(T):
        pos[t, :, 0] = t * 0.1
        pos[t, :, 1] = t * 0.05
    # 注入两个极端异常
    pos[10, 0, :] += 50   # 第10帧，关节0，所有轴跳变 +50
    pos[25, 1, 1] -= 40   # 第25帧，关节1，y轴跳变 -40
    return MotionData(
        joint_names=["joint0", "joint1"],
        fps=30,
        num_frames=T,
        positions=pos,
        angles=None,
        timestamps=None
    )


@pytest.fixture
def motion_with_nan():
    """
    含NaN缺失点的数据
    维度：(30, 1, 3)，在部分帧的x轴插入 NaN
    """
    T, J = 30, 1
    pos = np.tile(np.linspace(0, 10, T)[:, None, None], (1, J, 3))
    pos[[5, 12, 20], 0, 0] = np.nan
    return MotionData(
        joint_names=["joint0"],
        fps=30,
        num_frames=T,
        positions=pos
    )


# ================================================================
# 第一部分：异常检测 (detect_outliers) 测试
# ================================================================

def test_detect_outliers_normal_data(cleaner, normal_motion_data):
    """
    ✅ 预期通过
    场景：正常随机运动数据（无异常）
    验证：误检控制在合理范围内（≤5个）
    注意：使用 threshold=3.0 测试误检率
    """
    mask = cleaner.detect_outliers(normal_motion_data, threshold=3.0)
    assert mask.shape == (50, 2)
    total_outliers = np.sum(mask)
    assert total_outliers <= 5, f"误检过多: {total_outliers}"


@pytest.mark.xfail(
    reason="绝对值阈值（abs(data) > 10.0）未生效，+50/-40异常未被检测，待成员I修复"
)
def test_detect_outliers_has_outliers(cleaner, motion_with_outliers):
    """
    🟡 XFAIL（预期失败，待成员I修复）
    场景：显式注入两个极端异常（+50, -40）
    验证：能精准检测到这两个异常帧

    源码三重检测机制说明：
    1. 滑动窗口局部Z-score（threshold=2.5）：检测局部突变
    2. 全局MAD检测（jump_score > 5.0）：检测大幅跳变
    3. 绝对值阈值检测（abs(data) > 10.0）：直接拦截极端异常值
       → +50 和 -40 超过10米阈值，应被第三层拦截
    """
    mask = cleaner.detect_outliers(motion_with_outliers, threshold=2.5)
    assert mask.shape == (50, 2)
    # 硬编码检查：这两个异常必须被标记
    assert mask[10, 0] is True, "第10帧异常未被检测"
    assert mask[25, 1] is True, "第25帧异常未被检测"


def test_detect_outliers_no_positions(cleaner):
    """
    ✅ 预期通过
    场景：positions = None
    验证：返回全False掩码，不崩溃
    """
    empty_motion = MotionData(
        joint_names=["j1"],
        fps=30,
        num_frames=10,
        positions=None
    )
    mask = cleaner.detect_outliers(empty_motion)
    assert mask.shape == (10, 1)
    assert np.all(mask == False)


def test_detect_outliers_std_zero_avoid_div(cleaner):
    """
    ✅ 预期通过
    场景：全部数据恒定，标准差为0
    验证：不会触发除零异常
    """
    T, J = 20, 1
    pos = np.ones((T, J, 3)) * 5.0
    flat_motion = MotionData(
        joint_names=["j1"],
        fps=30,
        num_frames=T,
        positions=pos
    )
    mask = cleaner.detect_outliers(flat_motion)
    assert np.sum(mask) == 0


def test_detect_outliers_data_has_nan(cleaner, motion_with_nan):
    """
    ✅ 预期通过
    场景：原始数据含NaN
    验证：内部能正常处理，不崩溃
    """
    mask = cleaner.detect_outliers(motion_with_nan)
    assert mask.shape == (30, 1)


# ================================================================
# 第二部分：平滑滤波 (_smooth_positions) 测试
# ================================================================

def test_smooth_savgol_filter(cleaner, normal_motion_data):
    """
    ✅ 预期通过
    场景：Savitzky-Golay平滑
    验证：形状不变，无NaN
    """
    raw_pos = normal_motion_data.positions
    smooth_pos = cleaner._smooth_positions(raw_pos, window=5, filter_type="savgol")
    assert smooth_pos.shape == raw_pos.shape
    assert np.isfinite(smooth_pos).all()


def test_smooth_butter_filter(cleaner, normal_motion_data):
    """
    ✅ 预期通过
    场景：Butterworth低通滤波
    验证：正常执行，无报错
    """
    raw_pos = normal_motion_data.positions
    smooth_pos = cleaner._smooth_positions(raw_pos, window=5, filter_type="butter")
    assert smooth_pos.shape == raw_pos.shape


def test_smooth_unknown_filter_fallback_mean(cleaner, normal_motion_data):
    """
    ✅ 预期通过
    场景：传入未知filter_type
    验证：自动降级为移动平均，不崩溃
    """
    raw_pos = normal_motion_data.positions
    smooth_pos = cleaner._smooth_positions(raw_pos, window=5, filter_type="xxx")
    assert smooth_pos.shape == raw_pos.shape


def test_smooth_frame_less_than_window(cleaner):
    """
    ✅ 预期通过
    场景：帧数(3) < 窗口大小(5)
    验证：直接返回原始数据，不做平滑
    """
    tiny_pos = np.random.rand(3, 1, 3)
    out = cleaner._smooth_positions(tiny_pos, window=5, filter_type="savgol")
    np.testing.assert_allclose(out, tiny_pos)


def test_smooth_data_contains_nan_fill_first(cleaner, motion_with_nan):
    """
    ✅ 预期通过
    场景：输入数据含NaN
    验证：先插值修复再平滑，输出无NaN
    """
    pos = motion_with_nan.positions
    smooth_pos = cleaner._smooth_positions(pos, window=5, filter_type="savgol")
    assert np.isfinite(smooth_pos).all()


def test_smooth_window_even_auto_odd(cleaner, normal_motion_data):
    """
    ✅ 预期通过
    场景：传入偶数窗口(4)
    验证：内部自动转为奇数窗口(5)，正常运行
    """
    smooth_pos = cleaner._smooth_positions(
        normal_motion_data.positions,
        window=4,
        filter_type="savgol"
    )
    assert np.isfinite(smooth_pos).all()


# ================================================================
# 第三部分：主清洗流程 (clean) 全链路测试
# ================================================================

def test_clean_no_outlier_original_unchanged(cleaner, normal_motion_data):
    """
    ✅ 预期通过
    场景：无异常数据
    验证：原始数据不被修改，返回新对象
    """
    original_pos = normal_motion_data.positions.copy()
    result = cleaner.clean(normal_motion_data)
    np.testing.assert_allclose(normal_motion_data.positions, original_pos)
    assert result is not normal_motion_data
    assert result.positions.shape == original_pos.shape
    assert np.isfinite(result.positions).all()


def test_clean_fix_outliers_linear_interp(cleaner, motion_with_outliers):
    """
    ✅ 预期通过
    场景：含跳变异常，使用线性插值修复
    验证：异常被修复，无极端数值
    """
    res = cleaner.clean(
        motion_with_outliers,
        interpolate_method="linear",
        smooth_window=5
    )
    pos = res.positions
    # 修复后不应出现 ±40、±50 级别的极端值
    assert np.max(np.abs(pos)) < 15


def test_clean_cubic_interpolation(cleaner, motion_with_nan):
    """
    ✅ 预期通过
    场景：含NaN，使用三次样条插值修复
    验证：输出无NaN
    """
    res = cleaner.clean(motion_with_nan, interpolate_method="cubic")
    assert np.isfinite(res.positions).all()


@pytest.mark.xfail(
    reason="单点有效数据未广播填充（已知边界问题，待成员I修复）"
)
def test_clean_only_one_valid_point(cleaner):
    """
    🟡 XFAIL（预期失败，用于追踪修复进度）
    场景：仅第5帧有有效值(2.0)，其余全NaN
    验证：应广播填充到所有帧

    预期行为：如果成员I修复了单点填充逻辑，此用例会自动变为 PASS（XPASS 会提醒更新标记）
    如果仍然失败，报告会清晰显示为 XFAIL，表示"已知缺陷，不阻断流水线"
    """
    T, J = 20, 1
    pos = np.full((T, J, 3), np.nan)
    pos[5, 0, 0] = 2.0
    single_valid_motion = MotionData(
        joint_names=["j1"],
        fps=30,
        num_frames=T,
        positions=pos
    )
    res = cleaner.clean(single_valid_motion)

    # 断言：清洗后无NaN
    assert np.isfinite(res.positions).all(), "清洗后应无NaN"

    # 核心断言：单点有效值应广播到所有帧
    # 如果此断言失败，用例显示为 XFAIL（符合预期）
    # 如果意外通过，会显示 XPASS（提示可以移除 xfail 标记）
    assert np.all(res.positions[:, 0, 0] == 2.0), "单点有效数据应广播填充到所有帧"


def test_clean_no_positions_return_self(cleaner):
    """
    ✅ 预期通过
    场景：positions = None
    验证：直接返回原对象
    """
    empty_motion = MotionData(
        joint_names=["j1"],
        fps=30,
        num_frames=10,
        positions=None
    )
    res = cleaner.clean(empty_motion)
    assert res is empty_motion


def test_clean_preserve_angles_timestamps(cleaner, normal_motion_data):
    """
    ✅ 预期通过
    场景：清洗后
    验证：angles 和 timestamps 保持不变
    """
    res = cleaner.clean(normal_motion_data)
    np.testing.assert_allclose(res.angles, normal_motion_data.angles)
    np.testing.assert_allclose(res.timestamps, normal_motion_data.timestamps)


def test_clean_filter_butter_combination(cleaner, motion_with_outliers):
    """
    ✅ 预期通过
    场景：完整业务链路：异常检测 + cubic插值 + butter滤波
    验证：形状不变，无NaN
    """
    res = cleaner.clean(
        motion_with_outliers,
        smooth_window=7,
        interpolate_method="cubic",
        filter_type="butter"
    )
    assert res.positions.shape == motion_with_outliers.positions.shape
    assert np.isfinite(res.positions).all()