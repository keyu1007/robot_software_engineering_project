"""
Booster T1 MuJoCo 仿真播放器
成员F — 实现 MuJoCoPlayer 接口
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from common.interfaces import MuJoCoPlayer
from common.motion_data import MotionData
from common.logger import setup_logger

logger = setup_logger("MuJoCoPlayer")

# 自由关节占前7个 qpos（x, y, z, qw, qx, qy, qz）
FREE_DOF = 7


class MuJoCoPlayerImpl(MuJoCoPlayer):
    """使用 MuJoCo 物理引擎播放 Booster T1 机器人动作数据。"""

    def __init__(self, width: int = 640, height: int = 480):
        self._width = width
        self._height = height
        self._model: Optional["mujoco.MjModel"] = None
        self._data: Optional["mujoco.MjData"] = None
        self._renderer: Optional["mujoco.Renderer"] = None
        self._home_qpos: Optional[np.ndarray] = None
        self._fps: int = 30

    # ------------------------------------------------------------------
    # load_model
    # ------------------------------------------------------------------
    def load_model(self, model_path: Union[str, Path]) -> None:
        import mujoco

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        self._data = mujoco.MjData(self._model)

        if self._model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self._model, self._data, 0)

        self._home_qpos = self._data.qpos.copy()
        self._renderer = mujoco.Renderer(self._model, self._height, self._width)

        logger.info(
            f"模型已加载: {model_path} (actuators={self._model.nu}, qpos={self._model.nq})"
        )

    # ------------------------------------------------------------------
    # play
    # ------------------------------------------------------------------
    def play(
        self,
        robot_motion: MotionData,
        output_path: Optional[Union[str, Path]] = None,
        render: bool = True,
        **kwargs,
    ) -> List[np.ndarray]:
        import mujoco

        if self._model is None or self._data is None or self._home_qpos is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        angles = robot_motion.angles
        if angles is None:
            raise ValueError("robot_motion.angles 为 None")

        T, J_in = angles.shape
        J = min(J_in, self._model.nu)

        fps = kwargs.get("fps", robot_motion.fps)
        loop = kwargs.get("loop", 1)

        rendered_frames: List[np.ndarray] = []
        video_writer = None

        if output_path is not None:
            import imageio

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            video_writer = imageio.get_writer(
                output_path, fps=fps, format="FFMPEG", codec="libx264"
            )
            logger.info(f"视频输出: {output_path}")

        # 缓存关节限位（通过 actuator → joint 映射获取）
        jnt_limits = []
        for j in range(J):
            jnt_id = self._model.actuator_trnid[j][0]
            lo = float(self._model.jnt_range[jnt_id][0])
            hi = float(self._model.jnt_range[jnt_id][1])
            jnt_limits.append((lo, hi))

        try:
            for _ in range(loop):
                for t in range(T):
                    # 驱动关节
                    for j in range(J):
                        val = float(self._home_qpos[FREE_DOF + j]) + float(angles[t, j])
                        lo, hi = jnt_limits[j]
                        self._data.qpos[FREE_DOF + j] = np.clip(val, lo, hi)

                    # 身体倾斜：从腰关节推算（腿已锁，用腰扭+肩差模拟体态）
                    waist = float(self._data.qpos[FREE_DOF + 10])
                    sh_L = float(self._data.qpos[FREE_DOF + 2])
                    sh_R = float(self._data.qpos[FREE_DOF + 6])
                    lean_fwd = (sh_L + sh_R) / 2 * 0.02
                    lean_side = waist * 0.04

                    # 绕 X(前倾) 和 Y(侧倾) 微旋转自由关节
                    qw = 1.0 - abs(lean_fwd) - abs(lean_side)
                    qx = lean_fwd * 0.5
                    qy = lean_side * 0.5
                    qz = 0.0
                    qn = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
                    self._data.qpos[0:3] = self._home_qpos[0:3]
                    self._data.qpos[3:7] = np.array([qw, qx, qy, qz]) / qn
                    self._data.qvel[:] = 0.0
                    mujoco.mj_forward(self._model, self._data)

                    if render and self._renderer is not None:
                        self._renderer.update_scene(self._data)
                        frame = self._renderer.render()
                        rendered_frames.append(frame.copy())

                        if video_writer is not None:
                            video_writer.append_data(frame)
        finally:
            if video_writer is not None:
                video_writer.close()
                logger.info(f"视频已保存: {output_path} ({len(rendered_frames)} 帧)")

        logger.info(f"播放完成: {len(rendered_frames)} 帧")
        return rendered_frames

    # ------------------------------------------------------------------
    # get_physics_state
    # ------------------------------------------------------------------
    def get_physics_state(self) -> dict:
        if self._data is None:
            return {"qpos": np.zeros(0), "qvel": np.zeros(0), "time": 0.0}
        return {
            "qpos": self._data.qpos.copy(),
            "qvel": self._data.qvel.copy(),
            "time": float(self._data.time),
        }
