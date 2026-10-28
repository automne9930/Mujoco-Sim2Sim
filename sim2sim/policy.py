"""ONNX 策略封装。"""

import numpy as np
import onnxruntime as ort


def make_policy(path):
    """加载 ONNX，返回 obs -> action 的函数。"""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    def infer(obs):
        x = obs.astype(np.float32)[None, :]
        return session.run([output_name], {input_name: x})[0][0]

    return infer
