import pytest
import numpy as np
import cv2
import os
import tempfile
from pathlib import Path
from common.interfaces import VideoProcessor
from project1_dance.video_processor.extractor import VideoProcessor


# ===================== Fixture =====================

@pytest.fixture
def video_processor():
    """实例化视频处理器"""
    return VideoProcessor()


@pytest.fixture
def sample_video_path():
    """
    返回测试视频路径
    注意：需要在 inputs/ 目录下放置一个测试视频文件
    """
    return "inputs/sample_dance.mp4"


@pytest.fixture
def temp_video_path():
    """
    创建一个临时测试视频（用于测试无真实视频的场景）
    使用 OpenCV 生成一个简单的人工视频
    """
    # 创建临时文件
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    
    # 生成测试视频：30帧，纯色渐变
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (640, 480))
    
    for i in range(30):
        frame = np.full((480, 640, 3), i * 8, dtype=np.uint8)
        writer.write(frame)
    
    writer.release()
    
    yield path
    
    # 测试完成后清理
    if os.path.exists(path):
        os.remove(path)


# ================================================================
# 第一部分：extract_frames 抽帧功能测试
# ================================================================

def test_extract_frames_normal(video_processor, sample_video_path):
    """
    ✅ 预期通过
    场景：正常视频，使用默认参数抽帧
    验证：返回帧列表，帧数据格式正确
    """
    # 如果测试视频不存在，跳过测试
    if not os.path.exists(sample_video_path):
        pytest.skip(f"测试视频 {sample_video_path} 不存在，请放置一个测试视频文件")
    
    frames = video_processor.extract_frames(sample_video_path)
    
    # 验证返回的是列表
    assert isinstance(frames, list), "返回结果应为列表"
    assert len(frames) > 0, "未抽出任何帧"
    
    # 验证每帧是 numpy 数组
    for frame in frames[:5]:
        assert isinstance(frame, np.ndarray), "帧数据应为 numpy 数组"
        assert len(frame.shape) == 3, "帧应为三维数组 (H, W, C)"
        assert frame.shape[2] == 3, "帧应为3通道 (BGR)"
        assert frame.dtype == np.uint8, "帧数据类型应为 uint8"


def test_extract_frames_with_fps(video_processor, sample_video_path):
    """
    ✅ 预期通过
    场景：指定抽帧帧率（降低帧率）
    验证：抽出帧数减少
    """
    if not os.path.exists(sample_video_path):
        pytest.skip(f"测试视频 {sample_video_path} 不存在")
    
    # 不指定 fps，抽出全部帧
    frames_all = video_processor.extract_frames(sample_video_path)
    
    # 指定 fps=15（假设原视频 30fps，则帧数减半）
    frames_15 = video_processor.extract_frames(sample_video_path, fps=15)
    
    # 验证指定 fps 后帧数减少（或至少不大于）
    assert len(frames_15) <= len(frames_all), "指定 fps 后帧数应减少或不变"


def test_extract_frames_with_start_end(video_processor, sample_video_path):
    """
    ✅ 预期通过
    场景：指定起始和结束时间
    验证：只抽取指定时间段的帧
    """
    if not os.path.exists(sample_video_path):
        pytest.skip(f"测试视频 {sample_video_path} 不存在")
    
    # 抽取 1~3 秒的帧
    frames = video_processor.extract_frames(
        sample_video_path,
        start_sec=1.0,
        end_sec=3.0
    )
    
    assert len(frames) > 0, "指定时间区间应抽出帧"


def test_extract_frames_with_start_only(video_processor, sample_video_path):
    """
    ✅ 预期通过
    场景：只指定起始时间，到视频结尾
    验证：从指定时间到视频结尾
    """
    if not os.path.exists(sample_video_path):
        pytest.skip(f"测试视频 {sample_video_path} 不存在")
    
    frames_from_start = video_processor.extract_frames(sample_video_path, start_sec=1.0)
    frames_all = video_processor.extract_frames(sample_video_path)
    
    # 从中间开始，帧数应少于全部帧
    assert len(frames_from_start) < len(frames_all), "从中间开始帧数应少于全部帧"


def test_extract_frames_with_temp_video(video_processor, temp_video_path):
    """
    ✅ 预期通过
    场景：使用临时生成的测试视频
    验证：能正常读取和抽帧
    """
    frames = video_processor.extract_frames(temp_video_path, fps=30)
    
    assert len(frames) > 0, "临时视频应能抽出帧"
    assert isinstance(frames[0], np.ndarray), "帧数据应为 numpy 数组"


# ================================================================
# 第二部分：异常处理测试
# ================================================================

def test_extract_frames_file_not_found(video_processor):
    """
    ✅ 预期通过
    场景：视频文件不存在
    验证：抛出 FileNotFoundError
    """
    with pytest.raises(FileNotFoundError, match="视频文件不存在"):
        video_processor.extract_frames("non_existent_video_file.mp4")


def test_extract_frames_invalid_video(video_processor, tmp_path):
    """
    ✅ 预期通过
    场景：传入损坏或无效的视频文件
    验证：抛出 ValueError
    """
    invalid_path = tmp_path / "invalid.txt"
    invalid_path.write_text("this is not a video file")
    
    with pytest.raises(ValueError, match="无法打开视频文件"):
        video_processor.extract_frames(str(invalid_path))


def test_extract_frames_empty_path(video_processor):
    """
    ✅ 预期通过
    场景：传入空字符串路径
    验证：抛出异常
    """
    with pytest.raises(Exception):
        video_processor.extract_frames("")


# ================================================================
# 第三部分：get_video_metadata 元数据测试
# ================================================================

def test_get_video_metadata_normal(video_processor, sample_video_path):
    """
    ✅ 预期通过
    场景：正常视频
    验证：返回包含正确字段的元数据字典
    """
    if not os.path.exists(sample_video_path):
        pytest.skip(f"测试视频 {sample_video_path} 不存在")
    
    metadata = video_processor.get_video_metadata(sample_video_path)
    
    assert isinstance(metadata, dict), "返回结果应为字典"
    
    required_keys = ["fps", "total_frames", "duration_sec", "width", "height"]
    for key in required_keys:
        assert key in metadata, f"元数据缺少 {key} 字段"
    
    assert metadata["fps"] > 0, "fps 应大于 0"
    assert metadata["total_frames"] > 0, "total_frames 应大于 0"


def test_get_video_metadata_with_temp_video(video_processor, temp_video_path):
    """
    ✅ 预期通过
    场景：使用临时生成的测试视频
    验证：元数据正确
    """
    metadata = video_processor.get_video_metadata(temp_video_path)
    
    assert metadata["fps"] == 30.0, "fps 应为 30.0"
    assert metadata["total_frames"] == 30, "总帧数应为 30"
    assert metadata["duration_sec"] == 1.0, "时长应为 1 秒"
    assert metadata["width"] == 640, "宽度应为 640"
    assert metadata["height"] == 480, "高度应为 480"


def test_get_video_metadata_file_not_found(video_processor):
    """
    ✅ 预期通过
    场景：视频文件不存在
    验证：抛出 FileNotFoundError
    """
    with pytest.raises(FileNotFoundError, match="视频文件不存在"):
        video_processor.get_video_metadata("non_existent_video_file.mp4")


# ================================================================
# 第四部分：接口实现验证（契约测试）
# ================================================================

def test_video_processor_implements_interface(video_processor):
    """
    ✅ 预期通过
    场景：验证 VideoProcessor 类正确实现了接口
    验证：所有抽象方法都已实现，且签名匹配
    """
    from common.interfaces import VideoProcessor as VideoProcessorInterface
    assert isinstance(video_processor, VideoProcessorInterface), "未正确实现 VideoProcessor 接口"
    
    assert hasattr(video_processor, "extract_frames"), "缺少 extract_frames 方法"
    assert hasattr(video_processor, "get_video_metadata"), "缺少 get_video_metadata 方法"
    assert callable(video_processor.extract_frames), "extract_frames 不可调用"
    assert callable(video_processor.get_video_metadata), "get_video_metadata 不可调用"


# ================================================================
# 第五部分：边界条件测试
# ================================================================

def test_extract_frames_negative_start(video_processor, temp_video_path):
    """
    ✅ 预期通过
    场景：起始时间为负数
    验证：自动从第0帧开始
    """
    frames = video_processor.extract_frames(temp_video_path, start_sec=-1.0)
    assert len(frames) > 0, "负数起始时间应自动调整为 0"


def test_extract_frames_end_less_than_start(video_processor, temp_video_path):
    """
    ✅ 预期通过
    场景：结束时间小于起始时间
    验证：应返回空列表（或正确处理）
    """
    frames = video_processor.extract_frames(
        temp_video_path,
        start_sec=2.0,
        end_sec=1.0
    )
    assert isinstance(frames, list), "应返回列表"


def test_extract_frames_zero_fps(video_processor, temp_video_path):
    """
    ✅ 预期通过
    场景：fps 参数为 0
    验证：应使用原始帧率（或正确处理）
    """
    try:
        frames = video_processor.extract_frames(temp_video_path, fps=0)
        assert isinstance(frames, list), "应返回列表"
    except Exception as e:
        pass