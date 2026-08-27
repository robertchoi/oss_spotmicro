# /// script
# dependencies = [
#   "mujoco",
# ]
# ///
import mujoco

import mujoco.viewer
import time
import os

def run_simulation(model_path="../../urdf/spot_micro.xml"):
    # Attempt to find the model file
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        # MuJoCo can attempt to load URDFs directly, though MJCF is preferred.
        model = mujoco.MjModel.from_xml_path(model_path)
    else:
        # Check if the path is relative to the project root
        project_root_path = os.path.join("../../", model_path)
        if os.path.exists(project_root_path):
            print(f"Loading model from: {project_root_path}")
            model = mujoco.MjModel.from_xml_path(project_root_path)
        else:
            print(f"Warning: {model_path} not found. Loading fallback box model.")
            xml_content = """
            <mujoco>
                <worldbody>
                    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                    <geom type="plane" size="10 10 0.1" rgba=".9 0.9 .9 1"/>
                    <body name="box" pos="0 0 1">
                        <joint type="free"/>
                        <geom type="box" size=".1 .1 .1" rgba="0 1 0 1" mass="1"/>
                    </body>
                </worldbody>
            </mujoco>
            """
            model = mujoco.MjModel.from_xml_string(xml_content)
    
    try:
        data = mujoco.MjData(model)
        
        print("Starting MuJoCo Simulation Viewer...")
        print("Close the viewer window to stop.")
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                
                # Physics step
                mujoco.mj_step(model, data)
                
                # Sync viewer
                viewer.sync()
                
                # Real-time synchronization
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Path to the SpotMicro URDF
    run_simulation("../../urdf/spotmicroai_gen.urdf.xml")


