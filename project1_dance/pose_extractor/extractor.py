"""
姿态提取模块 - Mock 实现
严格遵循 Booster T1 项目统一 PoseExtractor 接口规范，
输出标准 MotionData 格式，用于模块联调与全流程验证。
"""

import sys
import os
import numpy as np
from typing import List

# 自动补全项目根路径，解决模块导入问题
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 导入项目公共接口、数据结构与标准关节定义
from common.interfaces import PoseExtractor
from common.motion_data import MotionData, BOOSTER_T1_JOINT_NAMES
from common import setup_logger

logger = setup_logger("pose_extractor")


class PromptHMRExtractor(PoseExtractor):
    """
    姿态提取器 Mock 实现
    完全对齐项目标准 PoseExtractor 接口，输出合规 MotionData，支持上下游模块联调
    """

    def __init__(self, **kwargs):
        super().__init__()
        logger.info("姿态提取模块（Mock版）初始化完成")

    def extract(
        self,
        frames: List[np.ndarray],
        fps: int,
        **kwargs,
    ) -> MotionData:
        total_frames = len(frames)
        logger.info(f"姿态提取：处理 {total_frames} 帧，帧率 {fps}")

        # 生成与 Booster T1 标准关节数对齐的模拟关节位置数据
        dummy_positions = (
            np.random.randn(total_frames, len(BOOSTER_T1_JOINT_NAMES), 3) * 0.3
        )

        motion_data = MotionData(
            joint_names=BOOSTER_T1_JOINT_NAMES,
            fps=fps,
            num_frames=total_frames,
            positions=dummy_positions,
        )
        return motion_data

    def get_joint_map(self) -> dict:
        """返回通用人体关节到 Booster T1 标准关节的映射表"""
        joint_map = {
            "pelvis": "root",
            "left_hip": "left_hip",
            "right_hip": "right_hip",
            "left_knee": "left_knee",
            "right_knee": "right_knee",
            "left_ankle": "left_ankle",
            "right_ankle": "right_ankle",
            "spine1": "waist",
            "spine2": "chest",
            "left_shoulder": "left_shoulder",
            "right_shoulder": "right_shoulder",
            "left_elbow": "left_elbow",
            "right_elbow": "right_elbow",
            "left_wrist": "left_wrist",
            "right_wrist": "right_wrist",
            "head": "head",
        }
        return joint_map
