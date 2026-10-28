"""IsaacLab -> MuJoCo Sim2Sim 主循环。

用法:
    python run_sim2sim.py [--config configs/go2w_robot_lab.yaml]
                          [--policy PATH|none] [--cmd VX VY WZ]

按键: ↑/↓ 前后, ←/→ 左右, Q/E 转向, R 复位（松手即停）。
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from omegaconf import OmegaConf

from sim2sim.command import KeyboardCommand
from sim2sim.mapping import build_joint_order, quat_rotate_inverse
from sim2sim.policy import make_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/go2w_robot_lab.yaml")
    parser.add_argument("--policy", default=None, help="覆盖 policy_onnx；填 none 表示零动作")
    parser.add_argument("--cmd", type=float, nargs=3, metavar=("VX", "VY", "WZ"),
                        help="固定指令，给定时不启用键盘")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg = OmegaConf.load(root / args.config)

    scene = Path(cfg.scene_xml)
    if not scene.is_absolute():
        scene = root / scene

    if args.policy is not None:
        policy_path = None if args.policy.lower() == "none" else Path(args.policy)
    else:
        policy_path = Path(cfg.policy_onnx) if cfg.policy_onnx is not None else None
    if policy_path is not None and not policy_path.is_absolute():
        policy_path = root / policy_path

    infer = make_policy(policy_path) if policy_path is not None else (
        lambda obs: np.zeros(16, dtype=np.float32))

    sim_dt = float(cfg.sim_dt)
    decimation = int(cfg.decimation)
    leg_count = int(cfg.leg_joint_count)

    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = sim_dt
    data = mujoco.MjData(model)

    mj_to_isaac, isaac_to_mj_obs, isaac_to_mj_ctrl = build_joint_order(cfg)
    default_q = np.array(list(cfg.default_joint_pos), dtype=np.float64)
    leg_scale = np.array([
        float(cfg.hip_action_scale) if "_hip_joint" in name else float(cfg.leg_action_scale)
        for name in list(cfg.isaaclab_joint_names)[:leg_count]
    ])
    last_action = np.zeros(16, dtype=np.float32)

    if args.cmd is not None:
        fixed_cmd = np.array(args.cmd, dtype=np.float64)
        get_cmd = lambda: fixed_cmd
        consume_reset = lambda: False
    else:
        keyboard = KeyboardCommand(cfg)
        get_cmd, consume_reset = keyboard.command, keyboard.consume_reset

    def reset():
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = (0.0, 0.0, float(cfg.init_base_height))
        data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        data.qpos[7:23] = default_q[isaac_to_mj_obs]
        last_action[:] = 0.0
        mujoco.mj_forward(model, data)

    reset()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t_next = time.perf_counter()
        while viewer.is_running():
            if consume_reset():
                reset()
            cmd = get_cmd()

            q = data.qpos[7:23][mj_to_isaac]
            qd = data.qvel[6:22][mj_to_isaac]
            q_rel = q - default_q
            if cfg.zero_wheel_joint_pos:
                q_rel[leg_count:] = 0.0

            obs = np.concatenate([
                data.qvel[3:6] * float(cfg.ang_vel_scale),          # 机体系角速度
                quat_rotate_inverse(data.qpos[3:7], [0.0, 0.0, -1.0]),  # projected_gravity
                cmd,
                q_rel * float(cfg.joint_pos_scale),
                qd * float(cfg.joint_vel_scale),
                last_action,
            ])
            clip = float(cfg.obs_clip)
            action = infer(np.clip(obs, -clip, clip))
            last_action[:] = action

            q_des = default_q[:leg_count] + action[:leg_count] * leg_scale
            v_des = np.clip(
                action[leg_count:] * float(cfg.wheel_action_scale),
                -float(cfg.velocity_limit),
                float(cfg.velocity_limit),
            )

            for _ in range(decimation):
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

            viewer.sync()

            t_next += sim_dt * decimation
            sleep = t_next - time.perf_counter()
            if sleep > 0.0:
                time.sleep(sleep)
            else:
                t_next = time.perf_counter()


if __name__ == "__main__":
    main()
