#=================================
# 本项目为Mujoco的Sim2Sim移植框架
#=================================
import time
from pathlib import Path

import mujoco
import mujoco.viewer

script_dir = Path(__file__).resolve().parent    # 意为当前Mujoco.py脚本所在的目录
xml_path = script_dir / "go2w.xml"             # 意为在脚本目录下寻找xml文件



model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

# 3. 创建非阻塞可视化窗口
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # 4. 推进一个物理仿真步
        mujoco.mj_step(model, data)

        # 5. 把最新状态同步到可视化窗口
        viewer.sync()

        # 6. 让仿真速度接近真实时间
        remaining_time = model.opt.timestep - (time.time() - step_start)
        if remaining_time > 0:
            time.sleep(remaining_time)