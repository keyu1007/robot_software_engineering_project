"""
从舞蹈视频提取 Booster T1 关节轨迹
MediaPipe 3D姿态 → Retargeting重定向 → 机器人关节角度
用法: python scripts/extract_trajectory.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mtp
from mediapipe.tasks.python import vision

from common.motion_data import MotionData, BOOSTER_T1_JOINT_NAMES
from project1_dance.retargeting.mapper import RetargetingImpl
from scipy.signal import savgol_filter

VIDEO_PATH = "inputs/dance.mp4"
OUTPUT_NPY = "outputs/trajectory.npy"
FPS_TARGET = 30

# MediaPipe 关键点 → BOOSTER_T1_JOINT_NAMES 位置索引
MP_TO_T1 = {
    "head": 0,            # nose
    "neck": None,         # computed
    "chest": None,        # computed
    "root": None,         # computed
    "waist": None,        # computed
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_toe": 31, "right_toe": 32,
    "left_upper_arm": None, "right_upper_arm": None,
    "left_thigh": None, "right_thigh": None,
}


def build_positions(landmarks_3d):
    """从 MediaPipe 33个3D关键点构建 BOOSTER_T1_JOINT_NAMES 格式的 positions (23, 3)"""
    k = np.array([[landmarks_3d[i].x, landmarks_3d[i].y, landmarks_3d[i].z] for i in range(33)])
    positions = np.zeros((23, 3))

    def set_pos(t1_name, mp_idx):
        idx = BOOSTER_T1_JOINT_NAMES.index(t1_name)
        if mp_idx is not None:
            positions[idx] = k[mp_idx]

    # 直接映射
    set_pos("head", 0)
    set_pos("left_shoulder", 11); set_pos("right_shoulder", 12)
    set_pos("left_elbow", 13); set_pos("right_elbow", 14)
    set_pos("left_wrist", 15); set_pos("right_wrist", 16)
    set_pos("left_hip", 23); set_pos("right_hip", 24)
    set_pos("left_knee", 25); set_pos("right_knee", 26)
    set_pos("left_ankle", 27); set_pos("right_ankle", 28)
    set_pos("left_toe", 31); set_pos("right_toe", 32)

    # 计算中间点
    m_sh = (k[11] + k[12]) / 2
    m_hip = (k[23] + k[24]) / 2
    set_pos("neck", None)
    positions[BOOSTER_T1_JOINT_NAMES.index("neck")] = m_sh
    set_pos("chest", None)
    positions[BOOSTER_T1_JOINT_NAMES.index("chest")] = m_sh
    set_pos("waist", None)
    positions[BOOSTER_T1_JOINT_NAMES.index("waist")] = m_hip
    set_pos("root", None)
    positions[BOOSTER_T1_JOINT_NAMES.index("root")] = m_hip

    # 中间段
    set_pos("left_upper_arm", None)
    positions[BOOSTER_T1_JOINT_NAMES.index("left_upper_arm")] = (k[11] + k[13]) / 2
    set_pos("right_upper_arm", None)
    positions[BOOSTER_T1_JOINT_NAMES.index("right_upper_arm")] = (k[12] + k[14]) / 2
    set_pos("left_thigh", None)
    positions[BOOSTER_T1_JOINT_NAMES.index("left_thigh")] = (k[23] + k[25]) / 2
    set_pos("right_thigh", None)
    positions[BOOSTER_T1_JOINT_NAMES.index("right_thigh")] = (k[24] + k[26]) / 2

    return positions


def extract_trajectory(video_path):
    # MediaPipe
    base_options = mtp.BaseOptions(model_asset_path="models/pose_landmarker_lite.task")
    options = vision.PoseLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.VIDEO,
        num_poses=1, min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5, min_tracking_confidence=0.5,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps_in / FPS_TARGET))

    # 重定向器
    retargeting = RetargetingImpl()

    all_positions = []
    frame_idx = 0
    saved = 0

    print(f"视频: {fps_in:.0f}fps, {total}帧, 步长={step}")
    print("提取3D姿态...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(frame_idx / fps_in * 1000)
        result = detector.detect_for_video(mp_img, ts)

        if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
            positions = build_positions(result.pose_world_landmarks[0])
            all_positions.append(positions)
            saved += 1
        elif all_positions:
            all_positions.append(all_positions[-1].copy())
            saved += 1

        frame_idx += 1
        if saved % 50 == 0:
            print(f"  {saved} 帧...")

    cap.release()
    detector.close()
    print(f"第一遍: {saved} 帧 3D 姿态")

    if saved < 10:
        return None

    # 构建 MotionData → 重定向
    positions_arr = np.array(all_positions)
    human_motion = MotionData(
        joint_names=list(BOOSTER_T1_JOINT_NAMES),
        fps=FPS_TARGET,
        num_frames=saved,
        positions=positions_arr,
        angles=None,
        timestamps=np.arange(saved) / FPS_TARGET,
    )

    print("重定向 → 机器人关节角度...")
    robot_motion = retargeting.retarget(human_motion, scale=0.8)
    angles = robot_motion.angles

    if angles is None:
        print("重定向失败")
        return None

    # 平滑
    for j in range(angles.shape[1]):
        try:
            w = min(9, max(3, saved // 3))
            if w % 2 == 0:
                w -= 1
            if w >= 3:
                angles[:, j] = savgol_filter(angles[:, j], w, 2)
        except Exception:
            pass

    # 统计
    print(f"\n完成: {saved} 帧")
    print("运动关节:")
    for j in range(angles.shape[1]):
        rng = angles[:, j].max() - angles[:, j].min()
        if rng > 0.02:
            print(f"  [{j:2d}] {robot_motion.joint_names[j]:22s} [{angles[:, j].min():+.2f} ~ {angles[:, j].max():+.2f}]")

    return angles, saved


def save_trajectory(angles, num_frames):
    import os
    os.makedirs("outputs", exist_ok=True)
    np.save(OUTPUT_NPY, angles)
    print(f"\n轨迹: {OUTPUT_NPY} ({os.path.getsize(OUTPUT_NPY)} bytes)")
    motion = MotionData(
        joint_names=list(BOOSTER_T1_JOINT_NAMES), fps=FPS_TARGET,
        num_frames=num_frames, angles=angles,
        timestamps=np.arange(num_frames) / FPS_TARGET,
    )
    print(f"MotionData: {motion.num_joints}关节, {motion.num_frames}帧, {motion.duration:.1f}s")


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    result = extract_trajectory(VIDEO_PATH)
    if result is not None:
        angles, num_frames = result
        save_trajectory(angles, num_frames)
        print("\npython scripts/demo_video.py")
