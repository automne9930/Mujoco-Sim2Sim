"""
===============================================================================
# 底层执行器模型：把策略输出的目标量换算成 MuJoCo 的 ctrl 力矩
===============================================================================
"""
import numpy as np


# 经典PD控制器
class PDMotor:
    # 初始化
    def __init__(self, kp, kd, effort_limit=None):
        # 在初始化阶段完成类型转换与维度对齐
        self.kp = np.asarray(kp, dtype=np.float64)
        self.kd = np.asarray(kd, dtype=np.float64)
        self.effort_limit = (
            None if effort_limit is None else np.asarray(effort_limit, dtype=np.float64)
        )

    # 计算函数
    def adjust(self, q, qd, q_des=None, v_des=0.0):
        tau = self.kd * (v_des - qd)
        if q_des is not None:
            tau += self.kp * (q_des - q)
        if self.effort_limit is not None:
            tau = np.clip(tau, -self.effort_limit, self.effort_limit)
        return tau
