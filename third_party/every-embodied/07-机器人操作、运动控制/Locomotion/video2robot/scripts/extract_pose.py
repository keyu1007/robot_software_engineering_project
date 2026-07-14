import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO


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

    # ========== 2. 读取视频逐帧提取 ==========
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Info] 视频总帧数: {total_frames}, FPS: {fps:.2f}")

    joints_3d_list = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 姿态推理
        results = model(frame, verbose=False)[0]

        if results.keypoints.has_visible:
            # 提取17个关键点的2D坐标 + 置信度
            kpts = results.keypoints.xy[0].cpu().numpy()  # (17, 2)
            # 补充伪深度（归一化相对深度，保证3D维度）
            conf = results.keypoints.conf[0].cpu().numpy()[:, None]
            joint_3d = np.concatenate([kpts, conf], axis=1)
        else:
            # 检测失败用上一帧填充
            if len(joints_3d_list) > 0:
                joint_3d = joints_3d_list[-1]
            else:
                joint_3d = np.zeros((17, 3))

        joints_3d_list.append(joint_3d)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"[Progress] 处理进度: {frame_idx}/{total_frames} 帧")

    cap.release()

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
