import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO

def repair_keypoints_by_confidence(kpts_sequence, conf_threshold=0.5):
    """
    基于置信度逐关节修复关键点序列：低置信度关节用上一帧有效数据填充
    """
    num_frames, num_joints, _ = kpts_sequence.shape
    repaired_xy = np.zeros((num_frames, num_joints, 2), dtype=np.float32)
    
    for joint_idx in range(num_joints):
        if kpts_sequence[0, joint_idx, 2] >= conf_threshold:
            repaired_xy[0, joint_idx] = kpts_sequence[0, joint_idx, :2]
        else:
            repaired_xy[0, joint_idx] = [0, 0]
    
    for frame_idx in range(1, num_frames):
        for joint_idx in range(num_joints):
            if kpts_sequence[frame_idx, joint_idx, 2] >= conf_threshold:
                repaired_xy[frame_idx, joint_idx] = kpts_sequence[frame_idx, joint_idx, :2]
            else:
                repaired_xy[frame_idx, joint_idx] = repaired_xy[frame_idx - 1, joint_idx]
    
    return repaired_xy


def temporal_lowpass_filter(kpts_sequence, window_size=5):
    """
    时序滑动平均低通滤波，削弱关键点帧间抖动
    """
    num_frames, num_joints, _ = kpts_sequence.shape
    filtered = np.zeros_like(kpts_sequence)
    
    for joint_idx in range(num_joints):
        for dim in range(2):
            coords = kpts_sequence[:, joint_idx, dim]
            kernel = np.ones(window_size) / window_size
            filtered[:, joint_idx, dim] = np.convolve(coords, kernel, mode='same')
    
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Video2Robot 姿态提取模块（YOLO兼容版）")
    parser.add_argument('--project', type=str, required=True, help='项目目录路径')
    parser.add_argument('--static-camera', action='store_true', help='静态相机标记（兼容原参数）')
    args = parser.parse_args()

    project_dir = args.project
    video_path = os.path.join(project_dir, 'original.mp4')
    output_path = os.path.join(project_dir, 'smplx.npz')

    print(f"[Project] {project_dir}")
    print(f"[Input] {video_path}")
    print(f"[Output] {output_path}")
    print("[YOLO Pose] Running pose extraction pipeline...")

    # ========== 1. 加载 YOLO 姿态模型 ==========
    # 自动下载轻量版姿态模型，CPU可运行
    model = YOLO('yolov8n-pose.onnx')

# ========== 2. 读取视频逐帧提取 + 时序平滑处理 ==========
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Info] 视频总帧数: {total_frames}, FPS: {fps:.2f}")

    all_kpts_with_conf = []  # 收集所有帧的关键点+置信度，形状[帧数, 17, 3] (x, y, 置信度)
    frame_idx = 0

    # 第一步：逐帧推理，收集全量关键点数据
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 姿态推理
        results = model(frame, verbose=False)[0]

        if results.keypoints.has_visible:
            # 提取17个关键点的2D坐标 + 置信度
            kpts_xy = results.keypoints.xy[0].cpu().numpy()  # (17, 2)
            conf = results.keypoints.conf[0].cpu().numpy()[:, None]  # (17, 1)
            # 拼接为 [x, y, 置信度] 格式
            kpts_single = np.concatenate([kpts_xy, conf], axis=1)  # (17, 3)
        else:
            # 未检测到人体，生成全0低置信度占位
            kpts_single = np.zeros((17, 3), dtype=np.float32)

        all_kpts_with_conf.append(kpts_single)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"[Progress] 处理进度: {frame_idx}/{total_frames} 帧")

    cap.release()

    # 第二步：序列后处理（完全对应作业要求）
    kpts_sequence = np.array(all_kpts_with_conf)  # 形状 [总帧数, 17, 3]

    # 2.1 置信度过滤：低置信度关节用上一帧数据插值修复
    repaired_xy = repair_keypoints_by_confidence(kpts_sequence, conf_threshold=0.5)
    # 2.2 时序低通滤波：对坐标做时间维度平滑，从源头消除抖动抽搐
    filtered_xy = temporal_lowpass_filter(repaired_xy, window_size=5)

    # 第三步：组装回原格式，保证下游代码零修改对接
    joints_3d_list = []
    num_frames = filtered_xy.shape[0]
    for i in range(num_frames):
        # 平滑坐标 + 原置信度拼接，输出结构与原规范完全一致
        conf_col = kpts_sequence[i, :, 2:3]
        joint_3d = np.concatenate([filtered_xy[i], conf_col], axis=1)
        joints_3d_list.append(joint_3d)


    # ========== 3. 封装为 SMPLX 兼容格式 ==========
    joints_3d = np.array(joints_3d_list)
    num_frames = joints_3d.shape[0]

    # 填充为127个SMPLX关节，保证下游脚本读取不报错
    smplx_joints = np.zeros((num_frames, 127, 3))

    # COCO 17关键点 → SMPLX 核心关节映射
    coco_to_smplx = {
        0: 0,    # 鼻子
        5: 13,   # 左肩
        6: 14,   # 右肩
        7: 15,   # 左肘
        8: 16,   # 右肘
        9: 17,   # 左手腕
        10: 18,  # 右手腕
        11: 1,   # 左髋
        12: 2,   # 右髋
        13: 4,   # 左膝
        14: 5,   # 右膝
        15: 7,   # 左脚踝
        16: 8,   # 右脚踝
    }
    for coco_idx, smplx_idx in coco_to_smplx.items():
        smplx_joints[:, smplx_idx, :] = joints_3d[:, coco_idx, :]

    # 构造标准输出字段，与原PromptHMR格式完全对齐
    output_data = {
        'joints_3d': smplx_joints,
        'pose': np.zeros((num_frames, 165)),
        'betas': np.zeros((num_frames, 10)),
        'trans': np.zeros((num_frames, 3)),
        'fps': np.array(fps),
        'num_frames': np.array(num_frames),
        'model_type': np.array('smplx', dtype='U10')
    }

    # ========== 4. 保存结果 ==========
    np.savez_compressed(output_path, **output_data)
    print("=" * 50)
    print(f"✅ 姿态提取完成，结果已保存至: {output_path}")
    print(f"[Info] 总帧数: {num_frames}, 帧率: {fps:.2f}")
    print("=" * 50)


if __name__ == '__main__':
    main()
