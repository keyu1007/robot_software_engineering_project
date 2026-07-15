"""
姿态提取模块 - YOLOv8-Pose 真实实现
严格遵循 Booster T1 项目统一 PoseExtractor 接口规范，
输出标准 MotionData 格式，集成置信度过滤与时序平滑。
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

# 姿态提取依赖
try:
    from ultralytics import YOLO
except ImportError:
    logger.error("未安装ultralytics依赖，请执行 pip install ultralytics")
    raise


class PromptHMRExtractor(PoseExtractor):
    """
    姿态提取器真实实现：基于YOLOv8-Pose
    完全对齐项目标准PoseExtractor接口，输出合规MotionData
    集成单关键点置信度过滤 + 时序低通平滑，缓解姿态抽搐
    """

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        device: str = "cpu",
        keypoint_conf_threshold: float = 0.5,
        smooth_alpha: float = 0.2,
        enable_smooth: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.keypoint_conf_threshold = keypoint_conf_threshold
        self.smooth_alpha = smooth_alpha
        self.enable_smooth = enable_smooth

        # 加载YOLOv8-Pose模型
        logger.info(f"加载姿态提取模型: {model_path} (设备: {device})")
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # COCO 17关键点 -> Booster T1标准关节 映射构建
        self._build_joint_mapping()
        logger.info("姿态提取模块（真实版）初始化完成")

    def _build_joint_mapping(self):
        """
        构建COCO 17关键点到Booster T1标准关节的映射与计算规则
        COCO 17关键点索引：
        0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear
        5:left_shoulder, 6:right_shoulder, 7:left_elbow, 8:right_elbow
        9:left_wrist, 10:right_wrist, 11:left_hip, 12:right_hip
        13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle
        """
        # 顺序与 BOOSTER_T1_JOINT_NAMES 100% 对齐
        self.t1_joint_rules = [
            ("root", lambda kp: (kp[11] + kp[12]) / 2),
            ("waist", lambda kp: (kp[11] + kp[12] + kp[5] + kp[6]) / 4),
            ("chest", lambda kp: (kp[5] + kp[6]) / 2),
            ("neck", lambda kp: ((kp[5] + kp[6]) / 2 + kp[0]) / 2),
            ("head", lambda kp: kp[0]),
            ("left_shoulder", lambda kp: kp[5]),
            ("left_elbow", lambda kp: kp[7]),
            ("left_wrist", lambda kp: kp[9]),
            ("right_shoulder", lambda kp: kp[6]),
            ("right_elbow", lambda kp: kp[8]),
            ("right_wrist", lambda kp: kp[10]),
            ("left_hip", lambda kp: kp[11]),
            ("left_knee", lambda kp: kp[13]),
            ("left_ankle", lambda kp: kp[15]),
            ("right_hip", lambda kp: kp[12]),
            ("right_knee", lambda kp: kp[14]),
            ("right_ankle", lambda kp: kp[16]),
            ("left_toe", lambda kp: kp[15] + np.array([5.0, 0.0])),
            ("right_toe", lambda kp: kp[16] + np.array([5.0, 0.0])),
            ("left_upper_arm", lambda kp: (kp[5] + kp[7]) / 2),
            ("right_upper_arm", lambda kp: (kp[6] + kp[8]) / 2),
            ("left_thigh", lambda kp: (kp[11] + kp[13]) / 2),
            ("right_thigh", lambda kp: (kp[12] + kp[14]) / 2),
        ]
        # 校验顺序与标准关节名完全一致
        joint_names = [name for name, _ in self.t1_joint_rules]
        assert joint_names == list(BOOSTER_T1_JOINT_NAMES), "关节顺序与标准定义不一致"

    def _coco_to_t1(self, coco_keypoints: np.ndarray) -> np.ndarray:
        """
        将单帧COCO 17关键点转换为Booster T1标准关节坐标
        输入形状: [17, 2] (x,y)
        输出形状: [23, 3] (x,y,z)，z轴暂补0兼容格式
        """
        t1_joints = np.zeros((len(BOOSTER_T1_JOINT_NAMES), 3), dtype=np.float32)
        for i, (_, calc_func) in enumerate(self.t1_joint_rules):
            xy = calc_func(coco_keypoints)
            t1_joints[i, :2] = xy
        return t1_joints

    def reset(self):
        """重置平滑缓存，切换视频/场景时调用，避免历史数据干扰"""
        self._last_smooth_joints = None

    def extract(
        self,
        frames: List[np.ndarray],
        fps: int,
        **kwargs,
    ) -> MotionData:
        total_frames = len(frames)
        logger.info(f"姿态提取：处理 {total_frames} 帧，帧率 {fps}")

        all_joint_positions = []
        self._last_smooth_joints = None

        for frame_idx, frame in enumerate(frames):
            # 1. 模型推理
            results = self.model(frame, verbose=False)[0]

            # 无有效检测结果（无检测框/无关键点），沿用历史帧或返回空
            if (
                results.boxes is None
                or len(results.boxes) == 0
                or results.keypoints is None
                or len(results.keypoints) == 0
            ):
                if self.enable_smooth and self._last_smooth_joints is not None:
                    all_joint_positions.append(self._last_smooth_keypoints.copy())
                else:
                    all_joint_positions.append(
                        np.zeros((len(BOOSTER_T1_JOINT_NAMES), 3), dtype=np.float32)
                    )
                continue

            # 2. 取置信度最高的单人体
            max_conf_idx = int(np.argmax(results.boxes.conf.cpu().numpy()))
            raw_xy = results.keypoints.xy[max_conf_idx].cpu().numpy()
            raw_conf = results.keypoints.conf[max_conf_idx].cpu().numpy()
            bbox_conf = results.boxes.conf[max_conf_idx].cpu().item()

            # 3. 关键点置信度过滤：低置信度点标记为无效，后续沿用历史值
            valid_mask = raw_conf >= self.keypoint_conf_threshold
            filtered_xy = raw_xy.copy()
            filtered_xy[~valid_mask] = np.nan

            # 4. 转换为T1标准关节
            current_joints = self._coco_to_t1(filtered_xy)

            # 5. 时序低通平滑（解决抽搐抖动）
            if self.enable_smooth:
                if self._last_smooth_joints is None:
                    # 首帧初始化，无效点补0
                    smooth_joints = current_joints.copy()
                    smooth_joints[np.isnan(smooth_joints)] = 0.0
                else:
                    smooth_joints = self._last_smooth_joints.copy()
                    # 仅有效关节点参与平滑更新，无效点保留上一帧结果
                    valid_joint_mask = ~np.isnan(current_joints).any(axis=1)
                    smooth_joints[valid_joint_mask] = (
                        self.smooth_alpha * current_joints[valid_joint_mask]
                        + (1 - self.smooth_alpha)
                        * self._last_smooth_joints[valid_joint_mask]
                    )
                self._last_smooth_joints = smooth_joints
            else:
                smooth_joints = current_joints.copy()
                smooth_joints[np.isnan(smooth_joints)] = 0.0

            all_joint_positions.append(smooth_joints)

        # 6. 组装标准MotionData输出
        positions = np.stack(all_joint_positions, axis=0)
        motion_data = MotionData(
            joint_names=BOOSTER_T1_JOINT_NAMES,
            fps=fps,
            num_frames=total_frames,
            positions=positions,
        )

        logger.info(f"姿态提取完成，输出 {total_frames} 帧运动数据")
        return motion_data

    def get_joint_map(self) -> dict:
        """返回通用人体关节到Booster T1标准关节的映射表"""
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


# 兼容主入口调用别名
PoseExtractorImpl = PromptHMRExtractor
