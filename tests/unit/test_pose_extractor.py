import sys
import os
import numpy as np

# 补全项目根路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from project1_dance.pose_extractor.extractor import PromptHMRExtractor


def test_pose_extract_single_frame():
    """单帧姿态提取测试"""
    extractor = PromptHMRExtractor()

    # 生成一张假图做快速测试
    dummy_frame = np.random.randint(0, 255, (720, 1080, 3), dtype=np.uint8)
    result = extractor.extract([dummy_frame], fps=30)

    # 验证输出格式符合项目标准
    assert result.num_frames == 1
    assert result.positions is not None
    assert len(result.positions.shape) == 3  # (帧数, 关节数, 3)
    assert result.positions.shape[2] == 3
    print("✅ 单帧姿态提取测试通过")


def test_pose_extract_multi_frames():
    """多帧姿态提取测试"""
    extractor = PromptHMRExtractor()
    frames = [
        np.random.randint(0, 255, (720, 1080, 3), dtype=np.uint8) for _ in range(10)
    ]
    result = extractor.extract(frames, fps=30)

    assert result.num_frames == 10
    assert result.positions.shape == (10, 23, 3)
    print("✅ 多帧姿态提取测试通过")


def test_joint_map():
    """关节映射表测试"""
    extractor = PromptHMRExtractor()
    joint_map = extractor.get_joint_map()

    assert isinstance(joint_map, dict)
    assert len(joint_map) > 0
    print("✅ 关节映射表测试通过")


def test_empty_frames():
    """边界测试：传入空帧列表，应触发MotionData格式校验异常"""
    extractor = PromptHMRExtractor()
    try:
        extractor.extract([], fps=30)
        print("❌ 空帧测试未通过：未抛出预期异常")
        assert False, "空帧输入应当触发校验错误"
    except ValueError as e:
        print("✅ 空帧边界测试通过：正确触发格式校验")


def test_zero_fps():
    """边界测试：帧率为0，应触发帧率校验异常"""
    extractor = PromptHMRExtractor()
    dummy_frame = np.random.randint(0, 255, (720, 1080, 3), dtype=np.uint8)
    try:
        extractor.extract([dummy_frame], fps=0)
        print("❌ 零帧率测试未通过：未抛出预期异常")
        assert False, "帧率为0应当触发校验错误"
    except ValueError as e:
        print("✅ 零帧率边界测试通过：正确触发帧率校验")


def test_negative_fps():
    """边界测试：帧率为负数，应触发帧率校验异常"""
    extractor = PromptHMRExtractor()
    dummy_frame = np.random.randint(0, 255, (720, 1080, 3), dtype=np.uint8)
    try:
        extractor.extract([dummy_frame], fps=-5)
        print("❌ 负帧率测试未通过：未抛出预期异常")
        assert False, "负帧率应当触发校验错误"
    except ValueError as e:
        print("✅ 负帧率边界测试通过：正确触发帧率校验")


def test_large_batch_frames():
    """边界测试：大批量帧输入，验证输出形状与稳定性"""
    extractor = PromptHMRExtractor()
    # 模拟1000帧长视频
    frames = [
        np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8) for _ in range(1000)
    ]
    result = extractor.extract(frames, fps=30)

    assert result.num_frames == 1000
    assert result.positions.shape == (1000, 23, 3)
    print("✅ 大批量帧输入测试通过：输出形状正确")


if __name__ == "__main__":
    test_pose_extract_single_frame()
    test_pose_extract_multi_frames()
    test_joint_map()
    test_empty_frames()
    test_zero_fps()
    test_negative_fps()
    test_large_batch_frames()
    print("\n🎉 全部测试通过")
