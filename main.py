#!/usr/bin/env python3
"""
Booster T1 舞蹈重定向流水线

使用方式:
    python main.py --video inputs/dance.mp4          # 真实模式
    python main.py --mock                           # Mock 模拟测试
    python main.py --video inputs/dance.mp4 --output outputs/result.mp4
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from common import setup_logger
# 修正拼写错误：CfonfigLoader → ConfigLoader
from common.config_loader import ConfigLoader
from common.mock_factory import (
    create_mock_video_processor,
    create_mock_pose_extractor,
    create_mock_retargeting,
    create_mock_mujoco_player,
)
from common.motion_data import MotionData
from project1_dance.motion_cleaner.cleaner import MotionCleanerImpl


logger = setup_logger("BoosterT1")


def main():
    parser = argparse.ArgumentParser(description="Booster T1 舞蹈重定向流水线")
    parser.add_argument(
        "--video",
        type=str,
        help="输入视频路径（如 inputs/dance.mp4）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/result.mp4",
        help="输出视频路径（默认 outputs/result.mp4）"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="使用 Mock 模式（跳过真实模型调用，用于测试流水线）"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径（默认 config.yaml）"
    )

    args = parser.parse_args()

    # ===================== 1. 加载配置 =====================
    config = ConfigLoader(args.config)
    mode = "mock" if args.mock else "real"

    # 增加逻辑：真实模式下必须提供视频文件
    if mode == "real" and args.video is None:
        logger.error("真实模式(real)运行时必须指定 --video 视频路径！")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Booster T1 流水线启动")
    logger.info(f"  模式:      {'Mock 模拟' if mode == 'mock' else '真实模式'}")
    logger.info(f"  输入视频:  {args.video if args.video else '(无)'}")
    logger.info(f"  输出视频:  {args.output}")
    logger.info(f"  配置文件:  {args.config}")
    logger.info("=" * 60)

    # ===================== 2. Step 1:视频处理 =====================
    logger.info("── Step 1/5: 视频处理 ──")
    if mode == "mock":
        video_proc = create_mock_video_processor()
        if args.video:
            frames = video_proc.extract_frames(args.video)
        else:
            frames = video_proc.extract_frames("mock.mp4")
    else:
        from project1_dance.video_processor.extractor import VideoProcessorImpl
        video_proc = VideoProcessorImpl()
        frames = video_proc.extract_frames(args.video)

    logger.info(f"  ✓ 已抽取 {len(frames)} 帧，shape={frames[0].shape if frames else 'N/A'}")

    # ===================== 3. Step 2:姿态提取 =====================
    logger.info("── Step 2/5: 姿态提取 ──")
    if mode == "mock":
        pose_ext = create_mock_pose_extractor()
        motion = pose_ext.extract(frames, fps=30)
    else:
        from project1_dance.pose_extractor.extractor import PoseExtractorImpl
        pose_ext = PoseExtractorImpl()
        motion = pose_ext.extract(frames, fps=30)

    logger.info(f"  ✓ MotionData: {motion.num_joints} 关节, {motion.num_frames} 帧, {motion.duration:.1f} 秒")

    # ===================== 4. Step3:数据清洗 =====================
    logger.info("── Step 3/5: 数据清洗 ──")
    cleaner = MotionCleanerImpl()
    outlier_mask = cleaner.detect_outliers(motion, threshold=3.0)
    outlier_count = int(outlier_mask.sum())
    logger.info(f"  检测到 {outlier_count} 个异常值")
    cleaned_motion = cleaner.clean(motion, smooth_window=5, filter_type="savgol")
    logger.info(f"  ✓ 清洗完成: {cleaned_motion.num_joints} 关节, {cleaned_motion.num_frames} 帧")

    # ===================== 5. Step4:动作重定向 =====================
    logger.info("── Step 4/5: 动作重定向 ──")
    if mode == "mock":
        retarget = create_mock_retargeting()
        robot_motion = retarget.retarget(cleaned_motion)
    else:
        from project1_dance.retargeting.mapper import RetargetingImpl
        retarget = RetargetingImpl()
        robot_motion = retarget.retarget(cleaned_motion)

    logger.info(f"  ✓ Robot MotionData: {robot_motion.num_joints} 关节, {robot_motion.num_frames} 帧")

    # ===================== 6. Step5:MuJoCo仿真播放 =====================
    logger.info("── Step 5/5: MuJoCo 仿真播放 ──")
    if mode == "mock":
        player = create_mock_mujoco_player()
        player.load_model("scene.xml")
        rendered_frames = player.play(
            robot_motion,
            output_path=args.output if args.video else None,
            render=True
        )
    else:
        from project1_dance.mujoco_player.player import MuJoCoPlayerImpl
        player = MuJoCoPlayerImpl()
        player.load_model("scene.xml")
        rendered_frames = player.play(
            robot_motion,
            output_path=args.output,
            render=True
        )

    logger.info(f"  ✓ 仿真完成: {len(rendered_frames)} 帧已渲染")

    # ===================== 7.汇总 =====================
    logger.info("=" * 60)
    logger.info("流水线执行成功")
    logger.info(
        f"  帧图像 {len(frames)} → 姿态 {motion.num_frames} 帧 → "
        f"清洗 {cleaned_motion.num_frames} 帧 → "
        f"机器人 {robot_motion.num_frames} 帧 → "
        f"渲染 {len(rendered_frames)} 帧"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()