import sys
import os
# Common 및 Kinematics 폴더를 찾기 위해 상위 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
import mujoco.viewer
from multiprocessing import Process
from Common.multiprocess_kb import KeyInterrupt
import spotmicroai
from kinematicMotion import TrottingGait

def main(command_status):
    # Robot 클래스를 MuJoCo 모드로 생성
    robot = spotmicroai.Robot(use_mujoco=True)
    trotting = TrottingGait()
    start_time = time.time()

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            elapsed = step_start - start_time

            # 키보드 입력 동기화
            result_dict = command_status.get()
            command_status.put(result_dict)

            # PyBullet 코드와 동일하게 동작
            if result_dict['StartStepping']:
                result_dict['IDstepLength'] = 60  # 보폭 6cm
                robot.feetPosition(trotting.positions(elapsed, result_dict))
            else:
                Lp_default = np.array([[120, -100, robot.W/2, 1], [120, -100, -robot.W/2, 1],
                                      [-50, -100, robot.W/2, 1], [-50, -100, -robot.W/2, 1]])
                robot.feetPosition(Lp_default)

            robot.step()
            viewer.sync()

            # 시뮬레이션 타임스텝 유지
            time_until_next_step = robot.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    KeyInputs = KeyInterrupt()
    KeyProcess = Process(target=KeyInputs.keyInterrupt, args=(1, KeyInputs.key_status, KeyInputs.command_status))
    KeyProcess.start()

    try:
        main(KeyInputs.command_status)
    except Exception as e:
        print(f"Main Loop Error: {e}")
    finally:
        if KeyProcess.is_alive():
            KeyProcess.terminate()
