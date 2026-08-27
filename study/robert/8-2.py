# /// script
# dependencies = [
#   "mujoco",
#   "pybullet",
#   "numpy",
# ]
# ///
import mujoco

import pybullet as p
import time
import numpy as np

def benchmark_mujoco(steps=1000):
    xml_content = """
    <mujoco>
        <worldbody>
            <geom type="plane" size="10 10 0.1"/>
            <body name="bot" pos="0 0 1">
                <joint type="free"/>
                <geom type="sphere" size=".1" mass="1"/>
            </body>
        </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)
    
    start = time.time()
    for _ in range(steps):
        mujoco.mj_step(model, data)
    end = time.time()
    
    return end - start

def benchmark_pybullet(steps=1000):
    p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    planeId = p.createCollisionShape(p.GEOM_PLANE)
    p.createMultiBody(0, planeId)
    
    sphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
    p.createMultiBody(1, sphereId, basePosition=[0, 0, 1])
    
    start = time.time()
    for _ in range(steps):
        p.stepSimulation()
    end = time.time()
    
    p.disconnect()
    return end - start

if __name__ == "__main__":
    ITERATIONS = 5000
    print(f"Benchmarking {ITERATIONS} steps...")
    
    mj_time = benchmark_mujoco(ITERATIONS)
    pb_time = benchmark_pybullet(ITERATIONS)
    
    print("-" * 30)
    print(f"MuJoCo   : {mj_time:.4f} sec ({ITERATIONS/mj_time:.2f} FPS)")
    print(f"PyBullet : {pb_time:.4f} sec ({ITERATIONS/pb_time:.2f} FPS)")
    print("-" * 30)
    
    if mj_time < pb_time:
        print(f"MuJoCo is {pb_time/mj_time:.2f}x faster in this test.")
    else:
        print(f"PyBullet is {mj_time/pb_time:.2f}x faster in this test.")
