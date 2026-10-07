import mujoco
import numpy as np

xml = """
<mujoco>
  <worldbody>
    <light pos="0 0 1"/>
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 0.5">
      <joint type="free"/>
      <geom type="box" size=".1 .1 .1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
for _ in range(100):
    mujoco.mj_step(model, data)
print("ok, qpos =", np.round(data.qpos, 3))