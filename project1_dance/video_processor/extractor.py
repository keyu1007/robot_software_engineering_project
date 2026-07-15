"""
视频处理模块 - 成员E
职责：读取输入视频，按固定帧率抽帧，输出图像列表
"""

import os
import cv2
import numpy as np
from typing import List, Optional, Union
from pathlib import Path
from common.interfaces import VideoProcessor


class VideoProcessor(VideoProcessor):
    """
    视频处理类，继承自 common.interfaces.VideoProcessor
    实现 extract_frames() 和 get_video_metadata() 方法
    """

    def extract_frames(
        self,
        video_path: Union[str, Path],
        fps: Optional[int] = None,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
    ) -> List[np.ndarray]:
        """
        从视频中抽取帧图像

        Args:
            video_path: 输入视频文件路径
            fps: 抽帧目标帧率，为 None 时使用视频原始帧率
            start_sec: 起始时间（秒），默认 0
            end_sec: 结束时间（秒），为 None 时到视频结尾

        Returns:
            List[np.ndarray]: 帧图像列表，每帧为 (H, W, 3) 的 BGR ndarray (uint8)

        Raises:
            FileNotFoundError: 视频文件不存在
            ValueError: 视频无法打开或解码
        """
        video_path = Path(video_path)

        # 1. 检查文件是否存在
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        # 2. 打开视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        # 3. 获取视频元信息
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps if original_fps > 0 else 0

        # 4. 确定起始和结束帧
        start_frame = int(start_sec * original_fps) if original_fps > 0 else 0
        end_frame = (
            int(end_sec * original_fps)
            if end_sec is not None and original_fps > 0
            else total_frames
        )
        start_frame = max(0, start_frame)
        end_frame = min(total_frames, end_frame)

        # 5. 确定抽帧间隔
        if fps is not None and fps > 0 and original_fps > 0:
            interval = max(1, int(original_fps / fps))
        else:
            interval = 1  # 每帧都取

        # 6. 跳转到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 7. 抽帧
        frames = []
        frame_count = start_frame
        while frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if (frame_count - start_frame) % interval == 0:
                frames.append(frame)  # OpenCV 默认 BGR，接口要求 BGR
            frame_count += 1

        cap.release()
        return frames

    def get_video_metadata(
        self,
        video_path: Union[str, Path],
    ) -> dict:
        """
        读取视频元信息

        Args:
            video_path: 输入视频文件路径

        Returns:
            dict: 包含 keys: fps, total_frames, duration_sec, width, height
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps if fps > 0 else 0

        cap.release()

        return {
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": duration_sec,
            "width": width,
            "height": height,
        }
