"""
===============================================================================
# MuJoCo-Sim2Sim 主程序运行
===============================================================================
用法:
    python run_sim2sim.py [--config configs/go2w_robot_lab.yaml]
                          [--policy PATH|none] [--cmd VX VY WZ]

按键: ↑/↓ 前后, ←/→ 左右, Q/E 转向, R 复位
===============================================================================
"""

import argparse     # CLI工具
import time         # 真实时钟
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from omegaconf import OmegaConf     # .yaml文件读取库

from sim2sim.command import KeyboardCommand     # command指令
from sim2sim.mapping import build_joint_order, quat_rotate_inverse  # 仿真关节映射/四元数转换
from sim2sim.policy import make_policy  # policy网络加载


def main():
    # CLI附加参数，后续通过 args.config、args.policy、args.cmd 即可提取对应配置
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/go2w_robot_lab.yaml")
    parser.add_argument("--policy", default=None, help="覆盖 policy_onnx, 填 none 表示零动作")
    parser.add_argument("--cmd", type=float, nargs=3, metavar=("VX", "VY", "WZ"),help="固定指令，给定时不启用键盘")
    args = parser.parse_args()

    # 锁定根目录位置，赋于变量root
    root = Path(__file__).resolve().parent

    # 读取yaml配置文件
    cfg = OmegaConf.load(root / args.config)

    # 从yaml文件中读取scene.xml路径
    scene = Path(cfg.scene_xml)
    if not scene.is_absolute():
        scene = root / scene

    # 读取policy路径
    if args.policy is not None:
        policy_path = None if args.policy.lower() == "none" else Path(args.policy)
    else:
        policy_path = Path(cfg.policy_onnx) if cfg.policy_onnx is not None else None
    if policy_path is not None and not policy_path.is_absolute():
        policy_path = root / policy_path

    # 自定义推理函数infer（如果是None Policy则返回16个0的action）
    infer = make_policy(policy_path) if policy_path is not None else (lambda obs: np.zeros(16, dtype=np.float32))

    # 从yaml文件中读取仿真dt、策略分频倍数、腿部自由度数量
    sim_dt = float(cfg.sim_dt)
    decimation = int(cfg.decimation)
    leg_count = int(cfg.leg_joint_count)

    # Mujoco初始化（model+data）
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = sim_dt
    data = mujoco.MjData(model)

    # 从yaml文件中读取Mujoco和IsaacSim的关节顺序转换列表
    mj_to_isaac, isaac_to_mj_obs, isaac_to_mj_ctrl = build_joint_order(cfg)

    # 标定默认关节角度q-default
    default_q = np.array(list(cfg.default_joint_pos), dtype=np.float64)

    # 标定action缩放因子scale
    leg_scale = np.array([
        float(cfg.hip_action_scale) if "_hip_joint" in name else float(cfg.leg_action_scale)
        for name in list(cfg.isaaclab_joint_names)[:leg_count]
    ])

    # 初始化last-action为零动作
    last_action = np.zeros(16, dtype=np.float32)

    # command相关
    if args.cmd is not None:    # 固定指令模式
        fixed_cmd = np.array(args.cmd, dtype=np.float64)
        get_cmd = lambda: fixed_cmd
        consume_reset = lambda: False
    else:                       # 键盘遥控
        keyboard = KeyboardCommand(cfg)
        get_cmd, consume_reset = keyboard.command, keyboard.consume_reset

    # reset函数
    def reset():
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = (0.0, 0.0, float(cfg.init_base_height))
        data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        data.qpos[7:23] = default_q[isaac_to_mj_obs]
        last_action[:] = 0.0
        mujoco.mj_forward(model, data)  # 手动计算一次，清理reset前的脏数据

    reset()

    # 注册一个mujoco.viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 先获取一次真实时间，每次循环计算完之后最后加上dt作为“预期下一轮开始的时间”，保持仿真与真实时间一致
        t_next = time.perf_counter()

        # 主循环
        while viewer.is_running():
            # 检测reset
            if consume_reset():
                reset()
            #=============================================
            # Mujoco中获取observation
            #=============================================
            # command获取
            cmd = get_cmd()

            # 获取广义坐标与速度（关节）并转换为网络训练时的关节顺序（isaaclab）
            q = data.qpos[7:23][mj_to_isaac]
            qd = data.qvel[6:22][mj_to_isaac]

            # 网络中使用相对default的角度q-rel
            q_rel = q - default_q

            # robot-lab训练中轮位置观测为0
            if cfg.zero_wheel_joint_pos:
                q_rel[leg_count:] = 0.0

            # 组建observation
            obs = np.concatenate([
                data.qvel[3:6] * float(cfg.ang_vel_scale),          # 机体系角速度
                quat_rotate_inverse(data.qpos[3:7], [0.0, 0.0, -1.0]),  # projected_gravity
                cmd,
                q_rel * float(cfg.joint_pos_scale),
                qd * float(cfg.joint_vel_scale),
                last_action,
            ])
            #=============================================
            # Policy网络输入obs进行推理
            #=============================================
            # 从yaml配置中获取观测裁切
            clip = float(cfg.obs_clip)

            # 带裁切推理
            action = infer(np.clip(obs, -clip, clip))

            # 记录上次action
            last_action[:] = action

            # 输出绝对目标（关节位置+速度）
            q_des = default_q[:leg_count] + action[:leg_count] * leg_scale
            v_des = np.clip(action[leg_count:] * float(cfg.wheel_action_scale), -float(cfg.velocity_limit), float(cfg.velocity_limit))

            #=============================================
            # action赋入Mujoco执行器
            #=============================================
            # 策略分频并ctrl（例如仿真0.02s，分频decimation=5，那么0.1s内5个仿真步用同一次输出action）
            for i in range(decimation):
                q = data.qpos[7:23][mj_to_isaac]
                qd = data.qvel[6:22][mj_to_isaac]
                tau = np.empty(16)
                tau[:leg_count] = (
                    float(cfg.leg_kp) * (q_des - q[:leg_count])
                    - float(cfg.leg_kd) * qd[:leg_count]
                )
                tau[leg_count:] = float(cfg.wheel_kd) * (v_des - qd[leg_count:])
                data.ctrl[:] = np.clip(
                    tau, -float(cfg.effort_limit), float(cfg.effort_limit)
                )[isaac_to_mj_ctrl]
                mujoco.mj_step(model, data)

            # 把最新状态同步到可视化窗口
            viewer.sync()

            # 一轮运行之后，计算下一次推理应该在什么时刻，若快于该时刻则sleep休眠
            t_next += sim_dt * decimation
            sleep = t_next - time.perf_counter()
            if sleep > 0.0:
                time.sleep(sleep)
            else:
                t_next = time.perf_counter()


if __name__ == "__main__":
    main()
