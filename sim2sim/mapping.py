"""关节顺序映射与四元数工具。"""

import numpy as np


def indices_for_order(source_order, target_order) -> np.ndarray:
    """计算用于 source_array[indices] -> target_array 的映射索引。"""
    return np.array([source_order.index(name) for name in target_order], dtype=np.int64)


def build_joint_order(cfg):
    """返回 (mj->isaac_obs, isaac->mj_obs, isaac->mj_ctrl) 三个索引数组。"""
    isaac = list(cfg.isaaclab_joint_names)
    mj_obs = list(cfg.mujoco_obs_joint_names)
    mj_ctrl = list(cfg.mujoco_actuator_names)
    mj_to_isaac_obs = indices_for_order(mj_obs, isaac)
    isaac_to_mj_obs = indices_for_order(isaac, mj_obs)
    isaac_to_mj_ctrl = indices_for_order(isaac, mj_ctrl)
    return mj_to_isaac_obs, isaac_to_mj_obs, isaac_to_mj_ctrl


def quat_rotate_inverse(q_wxyz, v):
    """R(q)^T · v，用于 projected_gravity。q 为 wxyz。"""
    w, x, y, z = q_wxyz
    v = np.asarray(v, dtype=np.float64)
    q_vec = np.array([x, y, z])
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * w * 2.0
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c
