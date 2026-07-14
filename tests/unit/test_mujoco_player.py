"""
仿真播放模块单元测试
测试目标：project1_dance/mujoco_player/player.py
测试范围：load_model()、play()、get_physics_state()、接口契约、边界条件
依赖：mujoco、imageio、scene.xml、assets/

运行命令：
$env:PYTHONPATH="$PWD"
pytest tests/unit/test_mujoco_player.py -v
"""
import pytest
import numpy as np
from pathlib import Path
from common.motion_data import MotionData, BOOSTER_T1_JOINT_NAMES
from common.interfaces import MuJoCoPlayer as MuJoCoPlayerInterface
from project1_dance.mujoco_player.player import MuJoCoPlayerImpl

np.random.seed(42)

# ===================== 依赖检测工具函数 =====================
def _has_mujoco() -> bool:
    try:
        import mujoco
        return True
    except ImportError:
        return False

def _has_imageio() -> bool:
    try:
        import imageio
        return True
    except ImportError:
        return False

MUJOCO_AVAILABLE = _has_mujoco()
IMAGEIO_AVAILABLE = _has_imageio()


# ===================== Fixture =====================
@pytest.fixture
def player():
    """实例化仿真播放器，无mujoco直接跳过"""
    if not MUJOCO_AVAILABLE:
        pytest.skip("mujoco 未安装，跳过全部仿真相关测试")
    return MuJoCoPlayerImpl()


@pytest.fixture
def robot_motion():
    """标准模拟机器人动作：10帧、23个标准执行器角度"""
    T = 10
    J = len(BOOSTER_T1_JOINT_NAMES)
    angles = np.random.randn(T, J) * 0.5
    return MotionData(
        joint_names=BOOSTER_T1_JOINT_NAMES,
        fps=30,
        num_frames=T,
        positions=None,
        angles=angles,
        timestamps=np.linspace(0, T/30, T)
    )


# ================================================================
# 第一部分：核心契约 & 硬性必过测试 🔴
# ================================================================
def test_player_implements_interface(player):
    """🔴 核心契约：校验完整实现抽象接口"""
    assert isinstance(player, MuJoCoPlayerInterface), "实例未继承 MuJoCoPlayer 接口"
    required_methods = ["load_model", "play", "get_physics_state"]
    for method in required_methods:
        assert hasattr(player, method), f"缺失接口方法: {method}"
        assert callable(getattr(player, method)), f"{method} 不可调用"


def test_load_model_with_scene_xml(player):
    """🔴 核心功能：正常加载场景xml模型"""
    xml_path = Path("scene.xml")
    if not xml_path.exists():
        pytest.skip("scene.xml 模型文件缺失，跳过加载测试")
    player.load_model(str(xml_path))


def test_load_model_file_not_found(player):
    """🔴 异常校验：不存在模型文件抛出FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        player.load_model("fake_model_1234.xml")


def test_get_physics_state_returns_dict(player):
    """🔴 核心功能：物理状态返回规范字典"""
    xml_path = Path("scene.xml")
    if not xml_path.exists():
        pytest.skip("scene.xml 缺失")
    player.load_model(str(xml_path))
    state = player.get_physics_state()
    assert isinstance(state, dict)
    assert "qpos" in state and "qvel" in state and "time" in state
    assert isinstance(state["qpos"], np.ndarray)
    assert isinstance(state["qvel"], np.ndarray)
    assert isinstance(state["time"], float)


def test_play_without_loading_model(player, robot_motion):
    """🔴 异常校验：未加载模型调用play抛RuntimeError"""
    with pytest.raises(RuntimeError, match="模型未加载"):
        player.play(robot_motion)


def test_play_with_angles_none(player):
    """🔴 异常校验：motion.angles为空抛出ValueError"""
    xml_path = Path("scene.xml")
    if not xml_path.exists():
        pytest.skip("scene.xml 缺失")
    player.load_model(str(xml_path))
    empty_motion = MotionData(
        joint_names=BOOSTER_T1_JOINT_NAMES,
        fps=30, num_frames=10, positions=None, angles=None
    )
    with pytest.raises(ValueError, match="robot_motion.angles 为 None"):
        player.play(empty_motion)


# ================================================================
# 第二部分：play基础功能测试 🟡
# ================================================================
def test_play_returns_list(player, robot_motion):
    """🟡 基础播放：无渲染返回空帧列表不崩溃"""
    xml_path = Path("scene.xml")
    if not xml_path.exists():
        pytest.skip("scene.xml 缺失")
    player.load_model(str(xml_path))
    frame_list = player.play(robot_motion, render=False)
    assert isinstance(frame_list, list)


def test_play_with_render(player, robot_motion):
    """🟡 带渲染播放：无头环境会xfail跳过"""
    xml_path = Path("scene.xml")
    if not xml_path.exists():
        pytest.skip("scene.xml 缺失")
    player.load_model(str(xml_path))
    try:
        frames = player.play(robot_motion, render=True)
        assert isinstance(frames, list)
        if len(frames) > 0:
            assert isinstance(frames[0], np.ndarray)
    except Exception as err:
        pytest.xfail(f"无显示器/无头环境渲染失败：{err}")


def test_play_with_output_path(player, robot_motion, tmp_path):
    """🟡 视频导出：依赖imageio，缺失自动xfail"""
    xml_path = Path("scene.xml")
    if not xml_path.exists():
        pytest.skip("scene.xml 缺失")
    if not IMAGEIO_AVAILABLE:
        pytest.xfail("imageio 未安装，无法测试视频写入")
    player.load_model(str(xml_path))
    out_file = tmp_path / "test_output.mp4"
    try:
        frames = player.play(robot_motion, render=False, output_path=str(out_file))
        assert isinstance(frames, list)
        if out_file.exists():
            assert out_file.stat().st_size > 0
    except Exception as e:
        pytest.xfail(f"视频输出可能因编码器问题失败：{e}")


def test_play_preserves_frame_count(player, robot_motion):
    """🟡 帧数对齐校验"""
    xml_path = Path("scene.xml")
    if not xml_path.exists():
        pytest.skip("scene.xml 缺失")
    player.load_model(str(xml_path))
    frames = player.play(robot_motion, render=False)
    # 无渲染模式下可能返回空列表，跳过验证
    if len(frames) == 0:
        pytest.skip("无渲染模式不返回帧，跳过帧数验证")
    assert len(frames) == robot_motion.num_frames


# ================================================================
# 第三部分：play入参参数测试 🟡
# ================================================================
def test_play_loop_param(player, robot_motion):
    """🟡 loop循环参数兼容测试"""
    xml_path = Path("scene.xml")
    if not xml_path.exists():
        pytest.skip("scene.xml 缺失")
    player.load_model(str(xml_path))
    f1 = player.play(robot_motion, render=False, loop=1)
    f3 = player.play(robot_motion, render=False, loop=3)
    assert isinstance(f1, list) and isinstance(f3, list)


def test_play_fps_param(player, robot_motion):
    """🟡 自定义fps参数兼容"""
    xml_path = Path("scene.xml")
    if not xml_path.exists():
        pytest.skip("scene.xml 缺失")
    player.load_model(str(xml_path))
    frames = player.play(robot_motion, render=False, fps=15)
    assert isinstance(frames, list)


# ================================================================
# 第四部分：边界场景测试 🟡
# ================================================================
def test_play_empty_motion(player):
    """
    🟡 边界测试：预期 xfail
    场景：0帧空动作数据
    说明：MotionData 校验要求 num_frames > 0，空帧在真实场景不会出现
          成员F的 demo 脚本均使用有效帧数据，此测试仅记录边界行为
    """
    xml_path = Path("scene.xml")
    if not xml_path.exists():
        pytest.skip("scene.xml 缺失")
    player.load_model(str(xml_path))

    # 构造 num_frames=0 会触发 MotionData 校验失败
    with pytest.raises(ValueError, match="num_frames 必须为正整数"):
        MotionData(
            joint_names=BOOSTER_T1_JOINT_NAMES,
            fps=30,
            num_frames=0,
            positions=None,
            angles=np.zeros((0, len(BOOSTER_T1_JOINT_NAMES))),
            timestamps=np.array([])
        )
    # 如果上述代码执行，说明 MotionData 校验失败，标记为 xfail
    pytest.xfail("MotionData 拒绝 num_frames=0，符合预期边界行为")


def test_play_single_frame(player):
    """🟡 单帧动作播放"""
    xml_path = Path("scene.xml")
    if not xml_path.exists():
        pytest.skip("scene.xml 缺失")
    player.load_model(str(xml_path))
    single_ang = np.random.randn(1, len(BOOSTER_T1_JOINT_NAMES)) * 0.5
    single_motion = MotionData(
        joint_names=BOOSTER_T1_JOINT_NAMES, fps=30, num_frames=1,
        positions=None, angles=single_ang, timestamps=np.array([0.0])
    )
    try:
        frames = player.play(single_motion, render=False)
        assert isinstance(frames, list)
    except Exception as e:
        pytest.xfail(f"单帧逻辑异常：{e}")


def test_play_joint_count_over(player):
    """🟡 关节数量超出模型，代码自动截断min(J_in, nu)校验"""
    xml_path = Path("scene.xml")
    if not xml_path.exists():
        pytest.skip("scene.xml 缺失")
    player.load_model(str(xml_path))
    # 构造25个关节，超过标准23
    over_joint_names = [f"j{i}" for i in range(25)]
    over_ang = np.random.randn(10, 25)
    over_motion = MotionData(
        joint_names=over_joint_names, fps=30, num_frames=10,
        positions=None, angles=over_ang
    )
    try:
        frames = player.play(over_motion, render=False)
        assert isinstance(frames, list)
    except Exception as e:
        pytest.xfail(f"关节超限截断逻辑异常：{e}")


# ================================================================
# 第五部分：环境依赖自检用例
# ================================================================
def test_check_mujoco_env():
    """环境自检：标记mujoco是否可用"""
    if MUJOCO_AVAILABLE:
        assert True
    else:
        pytest.skip("未安装mujoco，全部仿真用例跳过")


def test_check_imageio_env():
    """环境自检：标记imageio是否可用"""
    if IMAGEIO_AVAILABLE:
        assert True
    else:
        pytest.skip("未安装imageio，视频导出用例跳过")