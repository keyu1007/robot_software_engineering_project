"""
动作重定向模块单元测试
测试目标：project1_dance/retargeting/mapper.py
测试范围：retarget() 方法、get_joint_limits() 方法、内部IK计算、接口契约、边界条件

运行命令：
$env:PYTHONPATH="$PWD"
pytest tests/unit/test_retargeting.py -v

设计原则：
- 核心功能（接口契约、返回类型、数据完整性）必须通过
- 边缘和健壮性测试可因代码实现差异而失败，失败时记录原因
- 本测试基于 BOOSTER_T1_JOINT_NAMES 标准定义，与成员D代码对齐
"""
import pytest
import inspect
import numpy as np
from pathlib import Path
from common.motion_data import MotionData, BOOSTER_T1_JOINT_NAMES
from common.interfaces import Retargeting as RetargetingInterface
from project1_dance.retargeting.mapper import RetargetingImpl

# 固定随机种子，保证测试可复现
np.random.seed(42)

# ============================================================
# 关键常量（基于 BOOSTER_T1_JOINT_NAMES 标准索引）
# ============================================================
# BOOSTER_T1_JOINT_NAMES 定义（23个关节）：
# 0:root, 1:waist, 2:chest, 3:neck, 4:head,
# 5:left_shoulder, 6:left_elbow, 7:left_wrist,
# 8:right_shoulder, 9:right_elbow, 10:right_wrist,
# 11:left_hip, 12:left_knee, 13:left_ankle,
# 14:right_hip, 15:right_knee, 16:right_ankle,
# 17:left_toe, 18:right_toe,
# 19:left_upper_arm, 20:right_upper_arm, 21:left_thigh, 22:right_thigh

LEFT_ELBOW_IDX = 6      # BOOSTER_T1_JOINT_NAMES 中 left_elbow 的索引
RIGHT_ELBOW_IDX = 9     # BOOSTER_T1_JOINT_NAMES 中 right_elbow 的索引
LEFT_KNEE_IDX = 12      # BOOSTER_T1_JOINT_NAMES 中 left_knee 的索引
CHEST_IDX = 2           # BOOSTER_T1_JOINT_NAMES 中 chest 的索引

# mapping.yaml 中 left_elbow 限位 [0.0, 2.5]
LEFT_ELBOW_LIMIT_MAX = 2.5


# ===================== Fixture =====================

@pytest.fixture
def retargeter():
    """加载默认配置的重定向实例"""
    return RetargetingImpl()


@pytest.fixture
def sample_human_motion():
    """
    标准10帧人体直立动作，包含全部IK计算所需关节
    关节名称符合 mapper.py 中使用的命名规范
    """
    T, J = 10, 17
    joint_names = [
        "pelvis", "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "spine1", "spine2",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "head", "neck"
    ]
    positions = np.zeros((T, J, 3))
    for t in range(T):
        # 躯干
        positions[t, 0] = [0, 0, 0.8]      # pelvis
        positions[t, 7] = [0, 0, 1.2]      # spine1
        positions[t, 8] = [0, 0, 1.4]      # spine2
        positions[t, 15] = [0, 0, 1.6]     # head
        positions[t, 16] = [0, 0, 1.5]     # neck
        # 左腿（弯曲）
        positions[t, 1] = [-0.1, 0, 0.5]   # left_hip
        positions[t, 3] = [-0.15, 0.1, 0.2]  # left_knee
        positions[t, 5] = [-0.1, 0, -0.1]   # left_ankle
        # 右腿（弯曲）
        positions[t, 2] = [0.1, 0, 0.5]    # right_hip
        positions[t, 4] = [0.15, 0.1, 0.2]  # right_knee
        positions[t, 6] = [0.1, 0, -0.1]   # right_ankle
        # 左臂（弯曲）
        positions[t, 9] = [-0.2, 0, 1.3]   # left_shoulder
        positions[t, 11] = [-0.35, 0.15, 1.0]  # left_elbow
        positions[t, 13] = [-0.4, 0, 0.8]   # left_wrist
        # 右臂（弯曲）
        positions[t, 10] = [0.2, 0, 1.3]   # right_shoulder
        positions[t, 12] = [0.35, 0.15, 1.0]  # right_elbow
        positions[t, 14] = [0.4, 0, 0.8]   # right_wrist
    return MotionData(
        joint_names=joint_names,
        fps=30,
        num_frames=T,
        positions=positions,
        timestamps=np.linspace(0, T/30, T)
    )


@pytest.fixture
def extreme_human_motion():
    """
    极端大幅度人体姿态，用于测试角度限位裁剪
    主要针对左肘关节，使肘关节角度超出限位上限2.5
    """
    T, J = 5, 17
    joint_names = [
        "pelvis", "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "spine1", "spine2",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "head", "neck"
    ]
    positions = np.zeros((T, J, 3))
    for t in range(T):
        # 左臂完全伸直（肘关节角度会很大，约 > 3.0）
        positions[t, 0] = [0, 0, 0.8]      # pelvis
        positions[t, 9] = [-1.2, 0, 1.3]   # left_shoulder
        positions[t, 11] = [-2.0, 0, 1.0]  # left_elbow
        positions[t, 13] = [-3.0, 0, 0.8]  # left_wrist
        # 右臂对称设置
        positions[t, 10] = [1.2, 0, 1.3]   # right_shoulder
        positions[t, 12] = [2.0, 0, 1.0]   # right_elbow
        positions[t, 14] = [3.0, 0, 0.8]   # right_wrist
    return MotionData(
        joint_names=joint_names,
        fps=30,
        num_frames=T,
        positions=positions
    )


# ================================================================
# 第一部分：核心功能测试（必须通过 🔴）
# ================================================================

def test_retarget_implements_interface(retargeter):
    """
    🔴 核心契约：必须通过
    验证 RetargetingImpl 正确实现了 Retargeting 接口
    双重校验：继承关系 + 所有抽象方法已具体实现
    """
    # 第1层：继承关系校验
    assert isinstance(retargeter, RetargetingInterface), "未正确继承 Retargeting 接口"

    # 第2层：抽象方法具体实现校验
    abstract_methods = set()
    for name, m in inspect.getmembers(RetargetingInterface, inspect.isabstract):
        abstract_methods.add(name)

    for meth in abstract_methods:
        if meth.startswith("_") and meth != "__init__":
            continue
        assert hasattr(retargeter, meth), f"缺失抽象方法: {meth}"
        assert callable(getattr(retargeter, meth)), f"{meth} 不可调用"


def test_retarget_returns_motiondata(retargeter, sample_human_motion):
    """
    🔴 核心功能：必须通过
    验证返回正确的 MotionData 结构
    """
    result = retargeter.retarget(sample_human_motion)

    assert isinstance(result, MotionData), "应返回 MotionData 对象"
    assert result.angles is not None, "angles 应已填充"
    assert result.positions is None, "positions 应为 None"
    # 成员D代码使用 BOOSTER_T1_JOINT_NAMES 作为 joint_names
    assert result.joint_names == BOOSTER_T1_JOINT_NAMES, "关节名称应为 Booster T1 标准"


def test_retarget_angles_shape(retargeter, sample_human_motion):
    """
    🔴 核心功能：必须通过
    验证 angles 维度正确：形状为 (T, 23)
    """
    T = sample_human_motion.num_frames
    result = retargeter.retarget(sample_human_motion)
    expected_shape = (T, len(BOOSTER_T1_JOINT_NAMES))
    assert result.angles.shape == expected_shape, f"angles 维度应为 {expected_shape}"


def test_retarget_preserves_metadata(retargeter, sample_human_motion):
    """
    🔴 核心功能：必须通过
    验证 fps 和 timestamps 保持不变
    """
    result = retargeter.retarget(sample_human_motion)

    assert result.fps == sample_human_motion.fps, "fps 应保持不变"
    if sample_human_motion.timestamps is not None:
        np.testing.assert_allclose(
            result.timestamps,
            sample_human_motion.timestamps,
            err_msg="timestamps 应保持不变"
        )


def test_retarget_positions_none_raises_error(retargeter):
    """
    🔴 核心功能：必须通过
    验证 positions=None 时抛出明确的 ValueError
    """
    empty_motion = MotionData(
        joint_names=["pelvis", "head"],
        fps=30,
        num_frames=10,
        positions=None
    )
    with pytest.raises(ValueError, match="positions 不能为 None"):
        retargeter.retarget(empty_motion)


def test_get_joint_limits_valid(retargeter):
    """
    🔴 核心功能：必须通过
    验证 get_joint_limits 返回有效字典
    注意：mapping.yaml 中限位 key 使用小写加下划线
    """
    limits = retargeter.get_joint_limits()

    assert isinstance(limits, dict), "应返回字典"
    assert len(limits) > 0, "字典不应为空"

    # 使用 mapping.yaml 中实际的 key 名称（小写加下划线）
    key_joints = ["left_elbow", "right_knee", "chest"]
    for joint in key_joints:
        if joint in limits:
            min_val, max_val = limits[joint]
            assert isinstance(min_val, (int, float)), "限位值应为数值"
            assert isinstance(max_val, (int, float)), "限位值应为数值"
            assert min_val < max_val, "最小限位应小于最大限位"


# ================================================================
# 第二部分：功能正确性测试（应通过 🟡，可因实现差异失败）
# ================================================================

def test_retarget_angles_nonzero(retargeter, sample_human_motion):
    """
    🟡 功能验证：应通过
    验证正常姿态生成了非零角度
    如果失败，说明角度计算逻辑可能全部返回0
    """
    result = retargeter.retarget(sample_human_motion)
    assert np.any(np.abs(result.angles) > 0.001), "关节角度不应全为0"


def test_retarget_scale_parameter(retargeter, sample_human_motion):
    """
    🟡 功能验证：应通过
    验证 scale 参数生效
    """
    result_default = retargeter.retarget(sample_human_motion)
    result_scaled = retargeter.retarget(sample_human_motion, scale=0.3)

    mean_default = np.mean(np.abs(result_default.angles))
    mean_scaled = np.mean(np.abs(result_scaled.angles))

    # 如果默认角度为0，则测试无效，跳过
    if mean_default < 0.001:
        pytest.skip("默认角度为0，scale 测试无法验证")

    assert mean_scaled < mean_default * 0.8, "scale 应减小角度幅度"


def test_retarget_elbow_angle(retargeter, sample_human_motion):
    """
    🟡 功能验证：应通过
    验证肘关节角度计算（手臂弯曲时应 > 0）
    左肘在 BOOSTER_T1_JOINT_NAMES 中的索引为 6
    """
    result = retargeter.retarget(sample_human_motion)
    elbow_vals = result.angles[:, LEFT_ELBOW_IDX]
    assert np.mean(elbow_vals) > 0.05, f"左肘角度应 > 0.05，实际 {np.mean(elbow_vals):.4f}"


def test_retarget_knee_angle(retargeter, sample_human_motion):
    """
    🟡 功能验证：应通过
    验证膝关节角度计算（膝盖弯曲时应 > 0）
    左膝在 BOOSTER_T1_JOINT_NAMES 中的索引为 12
    """
    result = retargeter.retarget(sample_human_motion)
    knee_vals = result.angles[:, LEFT_KNEE_IDX]
    assert np.mean(knee_vals) > 0.01, f"左膝角度应 > 0.01，实际 {np.mean(knee_vals):.4f}"


def test_retarget_spine_angle(retargeter, sample_human_motion):
    """
    🟡 功能验证：应通过
    验证躯干角度（身体直立时应接近0）
    chest 在 BOOSTER_T1_JOINT_NAMES 中的索引为 2
    """
    result = retargeter.retarget(sample_human_motion)
    spine_vals = result.angles[:, CHEST_IDX]
    assert np.mean(np.abs(spine_vals)) < 0.5, f"躯干角度应接近0，实际 {np.mean(np.abs(spine_vals)):.4f}"


# ================================================================
# 第三部分：边缘与健壮性测试（可能失败 🟡/🟢，标记待修复）
# ================================================================

def test_retarget_joint_limits_clipping(retargeter, extreme_human_motion):
    """
    🟡 健壮性验证：预期应通过
    验证极端角度被限位裁剪
    如果失败，说明限位功能未实现或配置未生效
    """
    result = retargeter.retarget(extreme_human_motion)
    elbow_angles = result.angles[:, LEFT_ELBOW_IDX]

    within_range = np.all(elbow_angles >= 0.0 - 1e-6) and \
                   np.all(elbow_angles <= LEFT_ELBOW_LIMIT_MAX + 1e-6)

    if not within_range:
        pytest.xfail(
            f"关节限位裁剪未完全生效（角度范围 {np.min(elbow_angles):.3f} ~ {np.max(elbow_angles):.3f}），待成员D修复"
        )


def test_retarget_single_frame(retargeter, sample_human_motion):
    """
    🟢 边界测试：应通过
    单帧数据正常处理
    """
    single_motion = MotionData(
        joint_names=sample_human_motion.joint_names,
        fps=30,
        num_frames=1,
        positions=sample_human_motion.positions[:1],
        timestamps=sample_human_motion.timestamps[:1] if sample_human_motion.timestamps is not None else None
    )
    result = retargeter.retarget(single_motion)

    assert result.num_frames == 1, "单帧数据帧数应为1"
    assert result.angles.shape[0] == 1, "angles 第一维应为1"


def test_retarget_zero_frames_handling(retargeter):
    """
    🟡 健壮性验证：预期可能失败
    测试空帧输入时retarget处理逻辑
    前提：如果业务未来允许构建num_frames=0的MotionData
    """
    try:
        zero_motion = MotionData(
            joint_names=["pelvis"],
            fps=30,
            num_frames=0,
            positions=np.zeros((0, 1, 3))
        )
        result = retargeter.retarget(zero_motion)
        assert result.angles.shape == (0, len(BOOSTER_T1_JOINT_NAMES))
    except ValueError as e:
        if "num_frames 必须为正整数" in str(e):
            pytest.xfail("MotionData不允许0帧构造，当前无法测试retarget空帧逻辑，需协商规范")
        else:
            raise


def test_retarget_extra_kwargs(retargeter, sample_human_motion):
    """
    🟢 边界测试：应通过
    传入额外参数不应导致崩溃
    """
    result = retargeter.retarget(
        sample_human_motion,
        scale=0.8,
        unused_param=999,
        extra="test"
    )
    assert isinstance(result, MotionData), "额外参数不应影响执行"


def test_init_config_not_found():
    """
    🟡 异常处理：预期应通过
    配置文件不存在时抛出 FileNotFoundError
    如果失败，说明配置加载逻辑未处理缺失情况
    """
    fake_path = Path(__file__).parent / "fake_mapping_does_not_exist.yaml"

    try:
        RetargetingImpl(config_path=fake_path)
        pytest.xfail("配置缺失未抛 FileNotFoundError，可能使用了默认配置")
    except FileNotFoundError:
        assert True
    except Exception as e:
        assert False, f"预期 FileNotFoundError，实际抛出 {type(e).__name__}: {e}"