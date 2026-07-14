# 姿态提取模块

## 功能说明
严格遵循项目统一 `PoseExtractor` 抽象接口，输出 Booster T1 标准 `MotionData` 格式，支持上下游模块联调。

## 调用方式
```python
from project1_dance.pose_extractor.extractor import PromptHMRExtractor

# 初始化提取器
extractor = PromptHMRExtractor()

# 执行姿态提取
# frames: list[np.ndarray]，每一帧为(H, W, 3)的图像数组
# fps: 视频帧率
motion = extractor.extract(frames, fps=30)

# 获取关节映射表
joint_map = extractor.get_joint_map()

#输出说明
#返回标准 MotionData 对象
#包含 23 个 Booster T1 标准关节
#关节位置字段：motion.positions，形状为 (帧数, 23, 3)，单位：米


---

### 后续操作
写完后按 `Ctrl+O` 回车保存，`Ctrl+X` 退出，然后执行提交推送即可：
```bash
git add project1_dance/pose_extractor/README.md
git commit -m "docs(pose): 补充姿态提取模块使用文档"
git push origin feat/pose_extractor
