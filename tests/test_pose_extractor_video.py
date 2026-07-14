import cv2
import sys
sys.path.append(".")
from project1_dance.pose_extractor.extractor import PromptHMRExtractor


def test_with_dance_video():
    video_path = "inputs/test_dance.mp4"

    # 1. 读取视频并抽帧（模拟上游VideoProcessor的输出）
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 无法打开视频文件")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"✅ 视频读取完成，共 {len(frames)} 帧，原始帧率: {fps}")

    # 2. 调用姿态提取模块
    extractor = PromptHMRExtractor()
    motion_data = extractor.extract(frames, fps)

    # 3. 格式合规性校验
    print("\n📊 输出格式校验：")
    print(f"关节名称列表长度: {len(motion_data.joint_names)}")
    print(f"总帧数: {motion_data.num_frames}")
    print(f"帧率: {motion_data.fps}")
    print(f"positions 形状: {motion_data.positions.shape}")
    print(f"timestamps 形状: {motion_data.timestamps.shape}")

    # 4. 调用深度校验方法
    issues = motion_data.validate()
    if len(issues) == 0:
        print("\n🎉 全部校验通过！MotionData格式完全符合项目规范")
    else:
        print("\n❌ 校验失败：")
        for issue in issues:
            print(f"  - {issue}")


if __name__ == "__main__":
    test_with_dance_video()
