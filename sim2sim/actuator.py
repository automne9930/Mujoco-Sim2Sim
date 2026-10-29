"""
===============================================================================
# 底层执行器模型：把策略输出的目标量换算成 MuJoCo 的 ctrl 力矩
===============================================================================
"""
import numpy as np


# 经典PD控制器
class PDMotor:
    # 加载参数
    def __init__(self, kp, kd, effort_limit=None):
        self.kp = float(kp)
        self.kd = float(kd)
        self.effort_limit = None if effort_limit is None else float(effort_limit)

    # PD计算（q、qd分别代表当前位置、速度）
    # 位置控制则输入 adjust(q, qd, q_des)/速度控制则输入 adjust(q, qd, v_des)
    def adjust(self, q, qd, q_des=None, v_des=0.0):
        tau = self.kd * (np.asarray(v_des, dtype=np.float64) - qd)
        if q_des is not None:
            tau = tau + self.kp * (np.asarray(q_des, dtype=np.float64) - q)
        if self.effort_limit is not None:
            tau = np.clip(tau, -self.effort_limit, self.effort_limit)
        return tau
