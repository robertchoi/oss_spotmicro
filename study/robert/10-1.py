# 10-1.py: MuJoCo Menagerie Unitree G1 모델 로드 및 시각화
"""
Unitree G1 휴머노이드 로봇 모델을 로드하고 시각화하는 예제
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path

def get_model_path():
    """모델 경로 찾기"""
    # 현재 스크립트 위치에서 상대 경로로 찾기
    script_dir = Path(__file__).parent
    
    # 방법 1: 프로젝트 루트의 mujoco_menagerie 서브모듈
    submodule_path = script_dir.parent.parent / "mujoco_menagerie" / "unitree_g1" / "scene.xml"
    if submodule_path.exists():
        return str(submodule_path)
    
    # 방법 2: robot_descriptions 패키지 사용
    try:
        from robot_descriptions import unitree_g1_mj_description
        return unitree_g1_mj_description.MJCF_PATH
    except ImportError:
        pass
    
    raise FileNotFoundError(
        "Unitree G1 모델을 찾을 수 없습니다.\n"
        "다음 중 하나를 설치하세요:\n"
        "1. git submodule add https://github.com/google-deepmind/mujoco_menagerie.git\n"
        "2. uv add robot_descriptions"
    )

def main():
    print("=" * 60)
    print("Unitree G1 휴머노이드 로봇 - MuJoCo 시뮬레이션")
    print("=" * 60)
    
    # 모델 로드
    model_path = get_model_path()
    print(f"\n모델 경로: {model_path}")
    
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # 모델 정보 출력
    print(f"\n{'='*40}")
    print("모델 정보")
    print(f"{'='*40}")
    print(f"관절 좌표 수 (nq): {model.nq}")
    print(f"관절 속도 수 (nv): {model.nv}")
    print(f"액추에이터 수 (nu): {model.nu}")
    print(f"바디 수 (nbody): {model.nbody}")
    print(f"지오메트리 수 (ngeom): {model.ngeom}")
    print(f"센서 수 (nsensor): {model.nsensor}")
    print(f"키프레임 수 (nkey): {model.nkey}")
    print(f"시뮬레이션 타임스텝: {model.opt.timestep}s")
    
    # 관절 이름 출력
    print(f"\n{'='*40}")
    print("관절 목록")
    print(f"{'='*40}")
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if joint_name:
            print(f"  [{i:2d}] {joint_name}")
    
    # 액추에이터 이름 출력
    print(f"\n{'='*40}")
    print("액추에이터 목록")
    print(f"{'='*40}")
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if actuator_name:
            print(f"  [{i:2d}] {actuator_name}")
    
    # 초기 키프레임 적용 (서있는 자세)
    if model.nkey > 0:
        print(f"\n초기 키프레임 적용 중...")
        mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # 뷰어로 시각화
    print(f"\n{'='*40}")
    print("시각화 시작 (창을 닫으면 종료)")
    print(f"{'='*40}")
    print("\n조작법:")
    print("  - 마우스 왼쪽 드래그: 회전")
    print("  - 마우스 오른쪽 드래그: 이동")
    print("  - 스크롤: 줌")
    print("  - 스페이스바: 시뮬레이션 일시정지/재개")
    print("  - 더블클릭: 바디 선택/추적")
    print("  - ESC: 종료")
    
    # 뷰어 실행 (passive 모드)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            
            # 랜덤 액션 (작은 토크)
            # data.ctrl[:] = np.random.uniform(-0.1, 0.1, size=model.nu)
            
            # 물리 시뮬레이션 스텝
            mujoco.mj_step(model, data)
            
            # 뷰어 동기화
            viewer.sync()
            
            # 실시간 속도 유지
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    elapsed = time.time() - start_time
    print(f"\n시뮬레이션 종료 (실행 시간: {elapsed:.1f}초)")


if __name__ == "__main__":
    main()
