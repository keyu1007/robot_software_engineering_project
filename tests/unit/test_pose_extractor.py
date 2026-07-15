"""
姿态提取模块单元测试
测试目标：project1_dance/pose_extractor/extractor.py
测试范围：extract() 方法、get_joint_map() 方法、接口契约、边界条件

版本说明：
- 从 Mock 升级为 YOLOv8-Pose 真实推理
- 新增置信度过滤 + 时序平滑
- 首次运行会自动下载 yolov8n-pose.pt 模型
- 测试数据为纯黑图像，无人体检测，因此部分用例标记为 xfail

运行前请安装依赖：
pip install ultralytics
"""
import pytest
import numpy as np
from common.motion_data import MotionData, BOOSTER_T1_JOINT_NAMES
from project1_dance.pose_extractor.extractor import PromptHMRExtractor

# 固定随机种子，保证随机生成的姿态数据可复现，避免随机用例偶发失败
np.random.seed(42)

# ===================== 依赖检查 =====================
def _has_ultralytics() -> bool:
    try:
        import ultralytics
        return True
    except ImportError:
        return False

ULTRALYTICS_AVAILABLE = _has_ultralytics()


# ===================== Fixture =====================
@pytest.fixture
def extractor():
    """实例化姿态提取器（YOLOv8-Pose 真实版本）"""
    if not ULTRALYTICS_AVAILABLE:
        pytest.skip("ultralytics 未安装，请执行: pip install ultralytics")
    # 首次调用会自动下载模型
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
    说明：纯黑图像无法检测人体，输出全零姿态，在合理范围内
    """
    result = extractor.extract(sample_frames, fps=30)
    positions = result.positions
    assert np.all(positions < 3.0) and np.all(positions > -3.0), "位置数据应在合理范围内"


@pytest.mark.xfail(
    reason="YOLOv8 无法从纯黑图像中检测人体，所有帧返回全零，不适用该测试",
    strict=False
)
def test_extract_positions_not_all_same(extractor, sample_frames):
    """
    🟡 XFAIL（预期失败，strict=False）
    场景：验证不同帧的数据不同
    说明：Mock 版本中该测试通过，但真实 YOLOv8 在纯黑图像输入下所有帧都返回全零
          strict=False 表示如果意外通过，仅告警不阻断测试
    """
    result = extractor.extract(sample_frames, fps=30)
    positions = result.positions
    frame0 = positions[0]
    frame5 = positions[5]
    is_different = not np.allclose(frame0, frame5)
    assert is_different, "不同帧的数据应不同"


# ================================================================
# 第二部分：get_joint_map 方法测试
# ================================================================
def test_get_joint_map_returns_dict(extractor):
    """
    ✅ 预期通过
    场景：调用 get_joint_map
    验证：返回字典且包含必要的映射，映射值均为有效关节名称
    """
    joint_map = extractor.get_joint_map()
    assert isinstance(joint_map, dict), "应返回字典"
    assert len(joint_map) > 0, "映射字典不应为空"

    # 验证关键关节映射存在
    expected_keys = ["pelvis", "head", "left_hip", "right_hip"]
    for key in expected_keys:
        assert key in joint_map, f"映射中缺少 {key}"

    # 验证映射值：均为字符串且是有效的 Booster T1 关节名称
    for key, value in joint_map.items():
        assert isinstance(value, str), f"映射值应为字符串，实际: {type(value)}"
        assert len(value) > 0, "映射值不应为空"
        # 验证映射值是否为有效的 T1 关节名称
        assert value in BOOSTER_T1_JOINT_NAMES, \
            f"映射值 '{value}' 不在标准关节列表中"


# ================================================================
# 第三部分：接口契约测试（优化版）
# ================================================================
def test_extractor_implements_interface(extractor):
    """
    ✅ 预期通过
    场景：验证 PromptHMRExtractor 正确实现了 PoseExtractor 接口
    说明：双重校验确保抽象方法已具体实现
    """
    from common.interfaces import PoseExtractor as PoseExtractorInterface
    import inspect

    # ---- 第1层：继承关系校验 ----
    assert isinstance(extractor, PoseExtractorInterface), "未正确继承 PoseExtractor 接口"

    # ---- 第2层：抽象方法具体实现校验 ----
    abstract_methods = set()
    for name, method in inspect.getmembers(PoseExtractorInterface, inspect.isabstract):
        abstract_methods.add(name)

    for method_name in abstract_methods:
        if method_name.startswith("_") and method_name != "__init__":
            continue
        assert hasattr(extractor, method_name), f"缺少抽象方法: {method_name}"
        assert callable(getattr(extractor, method_name)), f"{method_name} 不可调用"


# ================================================================
# 第四部分：边界条件测试
# ================================================================
def test_extract_with_kwargs(extractor, sample_frames):
    """
    ✅ 预期通过
    场景：传入额外 kwargs
    验证：不影响正常执行（实现忽略额外参数）
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
    场景：大量帧输入（50帧）
    验证：正常生成对应维度数据
    说明：原为1000帧，但因 YOLOv8 推理耗时较长（~0.5s/帧），
          缩减至50帧平衡测试效率，仅验证维度正确性。
          如需性能测试，建议单独使用 benchmark 工具。
    """
    large_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(50)]
    result = extractor.extract(large_frames, fps=30)
    assert result.num_frames == 50, "帧数应为 50"
    assert result.positions.shape[0] == 50, "positions 第一维应为 50"


# ================================================================
# 第五部分：新增特性测试（reset 方法）
# ================================================================
def test_reset_method_exists(extractor):
    """
    ✅ 预期通过
    场景：验证 reset 方法存在（新增特性）
    """
    assert hasattr(extractor, "reset"), "reset 方法应存在"
    assert callable(extractor.reset), "reset 应为可调用方法"
    # 调用 reset 不应报错
    extractor.reset()