# Week 08: MuJoCo 시뮬레이션 및 PyBullet 비교

이번 주차에는 로봇 공학 및 강화학습에서 널리 사용되는 **MuJoCo(Multi-Joint dynamics with Contact)** 시뮬레이터를 학습합니다. 기존에 사용하던 PyBullet과의 차이점을 이해하고, 모델 변환 및 기본 구동 방법을 익힙니다.

## 1. URDF를 MJCF로 변환하기

MuJoCo는 자체적인 모델 포맷인 **MJCF (XML)**를 사용합니다. URDF 모델을 MuJoCo에서 사용하려면 변환 과정이 필요합니다.

### 1.1 MuJoCo 시뮬레이션 환경 구축 (Windows/uv 트러블슈팅)
Windows Store 버전의 Python이나 특정 보안 설정에서 `uv` 사용 시 **`os error 6000`(암호화 오류)**가 발생할 수 있습니다. 이 경우 `uv` 대신 표준 가상환경(`venv`)과 `pip`를 사용하는 것이 정신 건강에 이롭습니다.

#### 방법 A: uv가 잘 작동하는 경우 (권장)
```bash
uv init --name spot_micro
uv add mujoco pybullet numpy
uv run 8-1.py
```

#### 방법 B: uv에서 `os error 6000` 발생 시 (표준 방식)
```bash
# 1. 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate

# 2. 필수 패키지 설치
python -m pip install mujoco pybullet numpy

# 3. 실행 (uv run 대신 직접 python 사용)
# 주의: uv run은 내부적으로 새로운 가상환경을 생성하려다 위 오류를 다시 일으킬 수 있습니다.
python 8-1.py
```

> [!CAUTION]
> Windows Store에서 설치한 Python은 `uv`와 호환성 문제가 잦습니다. 가급적 [python.org](https://www.python.org/)에서 내려받은 공식 바이너리 사용을 권장합니다.

### 1.2 `urdf2mjcf` 라이브러리 활용
`uv run --with` 명령어 역시 가상환경을 생성하므로 동일한 암호화 오류가 발생할 수 있습니다. 이 경우 아래와 같이 현재 가상환경에 직접 설치하여 사용하십시오.

```bash
# 1. 라이브러리 직접 설치
python -m pip install urdf2mjcf

# 2. 직접 실행 (--output 옵션 사용 필수)
urdf2mjcf ../../urdf/spotmicroai_gen.urdf.xml --output ../../urdf/spot_micro.xml
```

**변환 예시:**
```bash
urdf2mjcf path/to/robot.urdf --output path/to/robot.xml
```

> [!TIP]
> 변환 후에는 시뮬레이션 성능 최적화를 위해 Mesh 데이터 경로(`asset`)나 마찰 계수(`friction`), 접촉 파라미터(`solimp`, `solref`)를 MJCF 파일에서 수동으로 미세 조정하는 것이 좋습니다.

### 1.3 사례 연구: Spot-Mujoco (Open Source Reference)
Boston Dynamics Spot 로봇을 MuJoCo에서 고퀄리티로 구현한 오픈소스 프로젝트 [Spot-Mujoco](https://github.com/louislelay/Spot-Mujoco)를 참고하면 큰 도움이 됩니다.

**주요 특징:**
- **MJCF 네이티브 모델링**: URDF를 단순히 변환한 것이 아니라, MuJoCo의 MJCF 언어를 사용해 최적화되었습니다.
- **모델링 방식**:
    1. 원본 `.urdf.xacro` 파일에서 파라미터 추출.
    2. `.dae` 메쉬 파일을 Blender를 통해 `.obj`로 변환 (MuJoCo 권장 사항).
    3. MuJoCo Menagerie(Anymal C 등)의 구조를 벤치마킹하여 `spot.xml` 설계.
- **학습 포인트**: 복잡한 사족 보행 로봇의 관절 계통(Kinematic Chain)과 관성(Inertia) 파라미터를 MuJoCo 형식에 어떻게 매칭시켰는지 확인할 수 있습니다.

### 1.4 분석: 기존 URDF 파일 비교 및 MuJoCo 적용 지침
현재 프로젝트의 `urdf/` 폴더에 있는 두 주요 파일의 차이점과 MuJoCo 사용 시 유의사항을 정리합니다.

| 비교 항목 | `spotmicroai_gen_ros.urdf` | `spotmicroai_gen.urdf.xml` |
| :--- | :--- | :--- |
| **태그/형식** | 표준 ROS URDF | URDF 기반 XML (시뮬레이터용) |
| **메쉬 경로** | `package://spotmicroai/urdf/stl/` | `../Parts/`, `../STL/files/` (상대 경로) |
| **Lidar 링크** | 기본 포함 (`lidar_link`) | 주석 처리됨 (비활성화) |
| **주요 용도** | ROS 패키지 연동 및 가시화 | 독립 시뮬레이션 및 데이터 로딩 |

**MuJoCo 변환 시 핵심 체크리스트:**
1. **관성 데이터(Inertial) 보정**: 현재 두 파일 모두 `mass`는 설정되어 있으나 `inertia` 값이 `100` 또는 `1000` 등으로 매우 크게 잡혀 있습니다. 이는 현실적인 디버깅을 위해 실제 값으로 계산하거나 MuJoCo의 `autocount` 기능을 활용해 메쉬 기하 구조로부터 자동 계산하는 것이 좋습니다.
2. **충돌 모델(Collision) 단순화**: 메쉬 파일을 그대로 충돌 모델로 쓰기보다는, 파일 내 정의된 `box`나 `sphere` 형태의 단순 기하 도형을 활용해 시뮬레이션 속도를 높이십시오.
3. **경로 문제**: MuJoCo에서 로드할 때는 `spotmicroai_gen.urdf.xml`처럼 상대 경로 형식을 사용하는 것이 경로 오류를 줄이는 데 유리합니다.

---

## 2. MuJoCo 시뮬레이션 구동 (Python)

DeepMind에서 MuJoCo를 인수한 후, 파이썬 바인딩이 매우 직관적으로 변했습니다.

### 2.1 기본 환경 설정
```bash
uv add mujoco
```

### 2.2 기본 구동 코드 (`8-1.py`)
`uv`를 사용해 스크립트를 실행하면 필요한 의존성을 자동으로 관리할 수 있습니다.

```bash
uv run 8-1.py
```

```python
import mujoco
import mujoco.viewer
import time

# 1. 모델 로드 (MJCF 파일 경로)
model = mujoco.MjModel.from_xml_path("spotmicroai_gen.urdf.xml")
data = mujoco.MjData(model)

# 2. 뷰어 실행
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 초기화
    start_time = time.time()
    
    while viewer.is_running():
        step_start = time.time()
        
        # 물리 엔진 스텝 진행
        mujoco.mj_step(model, data)
        
        # 뷰어 업데이트 관리를 위한 처리
        viewer.sync()
        
        # 시뮬레이션 시간 동기화
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
```

---

## 3. PyBullet vs MuJoCo 비교

두 시뮬레이터는 각각의 장단점이 분명합니다.

| 비교 항목 | PyBullet | MuJoCo |
| :--- | :--- | :--- |
| **핵심 알고리즘** | Maximal Coordinate (안정성 위주) | Generalized Coordinate (속도/정확성 위주) |
| **계산 속도** | 빠름 (단순 모델) | **매우 빠름** (복잡한 다관절 모델) |
| **수치 안정성** | 보통 (접촉 시 튕김 발생 가능) | **높음** (부드러운 접촉 모델링) |
| **사용 편의성** | 매우 쉬움 (URDF 바로 로드) | 보통 (MJCF 변환 및 튜닝 필요) |
| **학습 환경** | CPU 기반 병렬화 위주 | **GPU 기반 병렬화 (MJX)** 지원 |
| **비용** | 오픈 소스 (무료) | 오픈 소스 (Apache 2.0, 무료) |

> [!IMPORTANT]
> **MuJoCo**는 관절이 많은 로봇(Humanoid, Quadruped)의 동역학 계산에 최적화되어 있어, 강화학습(RL) 연구에서 표준으로 자리 잡았습니다. 반면 **PyBullet**은 초기 프로토타이핑과 간단한 로봇 그리퍼 조작 시뮬레이션에 강점이 있습니다.

---

## 4. 과제: 시뮬레이터 성능/안정성 비교

**목표**: 동일한 제어 알고리즘(예: Stand-up PD Control)을 PyBullet과 MuJoCo에서 각각 실행하고 결과를 비교합니다.

### 4.1 평가 지표
1. **평균 FPS (Frames Per Second)**: 단위 시간당 물리 스텝 계산 횟수.
2. **지면 접촉 안정성**: 로봇이 서 있을 때 발끝이 지면을 파고들거나(Penetration) 덜덜 떨리는 현상 관찰.
3. **CPU 점유율**: 시스템 리소스 효율성 확인.

### 4.2 실행 가이드
1. `8-2.py`를 실행하여 두 엔진의 루프 속도를 측정합니다.
2. 시뮬레이션 타임스텝(`dt`)을 0.001에서 0.01로 늘려보며, 어느 쪽이 더 먼저 무너지는지(Instability) 확인합니다.

---

## 다음 단계
- [ ] MJCF 파일 내 `contact` 옵션 이해하기
- [ ] MuJoCo의 `actuator`를 이용한 모터 제어 (Position/Velocity/Torque)
