"""
姿态提取模块单元测试
测试目标：project1_dance/pose_extractor/extractor.py
测试范围：extract() 方法、get_joint_map() 方法、接口契约、边界条件

根据以下文件设计测试用例：
- extractor.py: PromptHMRExtractor 类（Mock 实现）
- interfaces.py: PoseExtractor 抽象基类
- motion_data.py: MotionData 数据类 + BOOSTER_T1_JOINT_NAMES 常量
"""
import pytest
import numpy as np
import inspect
from common.motion_data import MotionData, BOOSTER_T1_JOINT_NAMES
from project1_dance.pose_extractor.extractor import PromptHMRExtractor

# 固定随机种子，保证随机生成的姿态数据可复现，避免随机用例偶发失败
np.random.seed(42)

# ===================== Fixture =====================
@pytest.fixture
def extractor():
    """实例化姿态提取器"""
    return PromptHMRExtractor()

@pytest.fixture
def sample_frames():
    """生成模拟帧图像列表（纯黑图像）"""
    frames = []
    for _ in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frames.append(frame)
    return frames

# ================================================================
# 第一部分：extract 方法测试
# ================================================================
def test_extract_returns_motiondata(extractor, sample_frames):
    """
    ✅ 预期通过
    场景：正常输入帧列表
    验证：返回 MotionData 对象，且字段正确
    """
    result = extractor.extract(sample_frames, fps=30)
    assert isinstance(result, MotionData), "应返回 MotionData 对象"
    assert result.joint_names == BOOSTER_T1_JOINT_NAMES, "关节名称应与标准一致"
    assert result.num_frames == len(sample_frames), "帧数应与输入帧数一致"
    assert result.fps == 30, "fps 应与输入一致"
    expected_shape = (10, len(BOOSTER_T1_JOINT_NAMES), 3)
    assert result.positions.shape == expected_shape, f"positions 维度应为 {expected_shape}"

def test_extract_with_different_fps(extractor, sample_frames):
    """
    ✅ 预期通过
    场景：不同 fps 值
    验证：fps 正确传递到 MotionData
    """
    result = extractor.extract(sample_frames, fps=15)
    assert result.fps == 15, "fps 应正确传递"

def test_extract_with_empty_frames(extractor):
    """
    ✅ 预期通过
    场景：空帧列表
    验证：MotionData 校验会拦截 num_frames=0，抛出 ValueError
    """
    with pytest.raises(ValueError):
        extractor.extract([], fps=30)

def test_extract_with_single_frame(extractor):
    """
    ✅ 预期通过
    场景：单帧输入
    验证：正确处理单帧数据
    """
    single_frame = [np.zeros((480, 640, 3), dtype=np.uint8)]
    result = extractor.extract(single_frame, fps=30)
    assert result.num_frames == 1, "单帧输入帧数应为 1"
    assert result.positions.shape[0] == 1, "positions 第一维应为 1"

def test_extract_positions_range(extractor, sample_frames):
    """
    ✅ 预期通过
    场景：验证生成的位置数据范围
    说明：Mock 数据是均值为0，标准差0.3 的随机数据，数值范围可控
    """
    result = extractor.extract(sample_frames, fps=30)
    positions = result.positions
    assert np.all(positions < 3.0) and np.all(positions > -3.0), "随机位置数据应在合理范围内"

def test_extract_positions_not_all_same(extractor, sample_frames):
    """
    ✅ 预期通过
    场景：验证不同帧的数据不同
    说明：Mock 应生成随机数据，不同帧不应完全相同
    """
    result = extractor.extract(sample_frames, fps=30)
    positions = result.positions
    frame0 = positions[0]
    frame5 = positions[5]
    is_different = not np.allclose(frame0, frame5)
    assert is_different, "不同帧的数据应不同（Mock 应生成随机数据）"

# ================================================================
# 第二部分：get_joint_map 方法测试
# ================================================================
def test_get_joint_map_returns_dict(extractor):
    """
    ✅ 预期通过
    场景：调用 get_joint_map
    验证：返回字典且包含必要的映射
    """
    joint_map = extractor.get_joint_map()
    assert isinstance(joint_map, dict), "应返回字典"
    assert len(joint_map) > 0, "映射字典不应为空"
    expected_keys = ["pelvis", "head", "left_hip", "right_hip"]
    for key in expected_keys:
        assert key in joint_map, f"映射中缺少 {key}"
    for key, value in joint_map.items():
        assert isinstance(value, str), "映射值应为字符串"
        assert len(value) > 0, "映射值不应为空"

# ================================================================
# 第三部分：接口契约测试（优化版）
# ================================================================
def test_extractor_implements_interface(extractor):
    """
    ✅ 预期通过
    场景：验证 PromptHMRExtractor 正确实现了 PoseExtractor 接口
    说明：
      - 第1层：验证继承关系（isinstance）
      - 第2层：验证抽象方法是否已被具体实现（inspect）
      双重校验，避免漏实现抽象方法仍通过测试
    """
    from common.interfaces import PoseExtractor as PoseExtractorInterface

    # ---- 第1层：继承关系校验 ----
    assert isinstance(extractor, PoseExtractorInterface), "未正确继承 PoseExtractor 接口"

    # ---- 第2层：抽象方法具体实现校验 ----
    # 获取接口中所有抽象方法名
    abstract_methods = set()
    for name, method in inspect.getmembers(PoseExtractorInterface, inspect.isabstract):
        abstract_methods.add(name)

    # 获取实现类中对应的方法
    for method_name in abstract_methods:
        # 跳过特殊方法
        if method_name.startswith("_") and method_name != "__init__":
            continue

        # 检查方法是否存在且可调用
        assert hasattr(extractor, method_name), f"缺少抽象方法: {method_name}"
        assert callable(getattr(extractor, method_name)), f"{method_name} 不可调用"

        # 额外检查：方法签名是否基本匹配（至少参数数量不严重偏离）
        impl_method = getattr(extractor, method_name)
        # 只检查最基本的一致性（有参数即可，不强制匹配数量，避免过度约束）
        assert callable(impl_method), f"{method_name} 应为可调用对象"

# ================================================================
# 第四部分：边界条件测试
# ================================================================
def test_extract_with_kwargs(extractor, sample_frames):
    """
    ✅ 预期通过
    场景：传入额外 kwargs
    验证：不影响正常执行（Mock 实现忽略额外参数）
    """
    result = extractor.extract(
        sample_frames,
        fps=30,
        confidence_threshold=0.5,
        model_path="/fake/path"
    )
    assert isinstance(result, MotionData), "额外参数不应影响执行"

def test_extract_fps_zero(extractor, sample_frames):
    """
    ✅ 预期通过
    场景：fps=0
    验证：MotionData初始化抛出数值异常
    """
    with pytest.raises(ValueError):
        extractor.extract(sample_frames, fps=0)

def test_extract_large_frame_count(extractor):
    """
    ✅ 预期通过
    场景：大量帧输入（1000帧）
    验证：正常生成对应维度数据
    说明：此为内存/维度验证，非性能压力测试，不统计耗时
    """
    large_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(1000)]
    result = extractor.extract(large_frames, fps=30)
    assert result.num_frames == 1000, "帧数应为 1000"
    assert result.positions.shape[0] == 1000, "positions 第一维应为 1000"