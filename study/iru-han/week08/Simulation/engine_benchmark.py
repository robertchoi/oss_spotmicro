# engine_benchmark.py 수정 버전
import time
import numpy as np
import pandas as pd
from spotmicroai import Robot
from kinematicMotion import TrottingGait


def run_benchmark(engine_type="pybullet", steps=500):
    use_mujoco = (engine_type == "mujoco")
    # Robot 초기화 (통합 클래스)
    robot = Robot(use_mujoco=use_mujoco, xml_path="../urdf/spot_micro_mujoco.xml")
    gait = TrottingGait()

    # 가이트 파라미터 강제 고정 (ZeroDivision 방지)
    gait.t1 = 350
    gait.t3 = 350
    gait.Sh = 60

    comp_times = []
    imu_logs = []

    print(f"--- Benchmarking {engine_type.upper()} ---")

    for i in range(steps):
        t_start = time.perf_counter()

        # d 값을 초 단위로 증가시킴 (0.01초씩)
        d = i * 0.01

        # 보행 명령 전달
        # 키보드 입력 딕셔너리 구조에 맞춰 전달
        cmd = {'IDstepLength': -60, 'IDstepWidth': 0, 'IDstepAlpha': 0}
        target_Lp = gait.positions(d, cmd)
        robot.feetPosition(target_Lp)

        # 엔진 물리 스텝 실행
        robot.step()

        t_end = time.perf_counter()
        comp_times.append(t_end - t_start)

        # IMU 데이터 (Roll, Pitch) 추출 및 저장
        # getIMU() 반환값 구조 확인 필요
        orn, _, _ = robot.getIMU()
        imu_logs.append(orn[:2])

    avg_ms = np.mean(comp_times) * 1000
    fps = 1.0 / np.mean(comp_times)

    # 통계 분석 (분산)
    variances = np.var(imu_logs, axis=0)

    return {
        "Engine": engine_type.upper(),
        "Avg_Step_ms": round(avg_ms, 4),
        "FPS": round(fps, 2),
        "Roll_Var": round(variances[0], 8),
        "Pitch_Var": round(variances[1], 8)
    }


if __name__ == "__main__":
    # 데이터 수집 및 비교 리포트 출력
    results = []
    results.append(run_benchmark("pybullet"))
    results.append(run_benchmark("mujoco"))

    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("            SPOT MICRO ENGINE BENCHMARK REPORT")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)