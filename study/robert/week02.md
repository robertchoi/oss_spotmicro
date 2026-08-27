---
title: PyBullet 환경 설정과 시뮬레이션 실행
subtitle: Week 02 · SpotMicroJetson 첫 실행
badge: SpotMicro Jetson 로보틱스
desc: 가상환경 구축 · PyBullet 설치 · pybullet_automatic_gait.py 실행
author: robert
header: Week 02 · PyBullet 환경 설정과 시뮬레이션 실행
toc: true
toc_depth: 2
---

SpotMicroJetson 프로젝트의 PyBullet 환경을 설정하고, 4족 보행 시뮬레이션을 직접 실행해 봅니다. Python 가상환경 구축부터 로봇 관절 정보 분석까지 한 번에 따라갑니다.

> **예상 소요 시간**: 약 2~3시간

## 학습 목표

이번 주차를 마치면 다음을 할 수 있습니다.

1. PyBullet이 무엇이며 왜 사용하는지 설명한다.
2. Python 가상환경에 PyBullet과 의존성 패키지를 설치한다.
3. `pybullet_automatic_gait.py`를 실행해 SpotMicro 보행 시뮬레이션을 확인한다.
4. SpotMicroJetson 프로젝트의 디렉터리 구조를 이해한다.
5. 로봇 URDF의 관절 정보를 분석한다.

---

## 1. PyBullet이란?

### 1.1 PyBullet 소개

**PyBullet**은 Bullet Physics SDK의 Python 래퍼(wrapper)로, 로봇 공학·강화학습·컴퓨터 그래픽스 분야에서 널리 사용되는 물리 시뮬레이션 엔진입니다.

**주요 특징**

- **오픈소스**: Zlib 라이선스로 상업적 이용 가능
- **Python 기반**: 간단한 API로 빠른 프로토타이핑 가능
- **URDF 지원**: 로봇 모델 표준 형식 완벽 지원
- **강화학습 친화적**: OpenAI Gym 환경과 쉽게 통합
- **다양한 제어 모드**: Position, Velocity, Torque 제어 지원
- **시각화**: 3D GUI를 통한 실시간 시뮬레이션 시각화

**주요 용도**

- 로봇 시뮬레이션 및 제어 알고리즘 테스트
- 강화학습 에이전트 훈련(로봇 제어)
- 역기구학(Inverse Kinematics) 계산
- 충돌 감지 및 물리 시뮬레이션
- 가상 환경에서의 로봇 동작 검증

### 1.2 2025년 현재 PyBullet의 위치

**여전히 활발하게 사용되는 이유**

1. **교육 및 연구에 최적**
   - 대학 강의와 연구 논문에서 표준으로 사용
   - 간단한 설치와 낮은 학습 곡선
   - 풍부한 예제와 튜토리얼
2. **강화학습 커뮤니티**
   - PyBullet Gym 환경이 여전히 인기
   - 경량화된 환경으로 빠른 학습 가능
   - 많은 RL 논문의 벤치마크로 사용
3. **프로토타이핑**
   - 빠른 개발 사이클
   - Python 생태계(NumPy, PyTorch, TensorFlow)와 원활한 연동

**고려할 점**

1. **업데이트 속도**
   - 마지막 주요 업데이트: 2021년(v3.2.x)
   - 유지보수는 계속되지만 새로운 기능 추가는 느림
   - Python 3.13 등 최신 환경 지원 지연
2. **최신 대안들**
   - **MuJoCo**(2021년 오픈소스화): 더 정확한 물리 시뮬레이션
   - **Isaac Gym/Sim**(NVIDIA): GPU 가속 대규모 병렬 시뮬레이션
   - **Gazebo**: ROS 통합이 중요한 경우

**2025년 사용 권장 사항**

| 용도 | PyBullet 적합도 | 권장 이유 |
|------|----------------|-----------|
| 교육/학습 | ⭐⭐⭐⭐⭐ | 가장 배우기 쉬움 |
| 연구 프로토타이핑 | ⭐⭐⭐⭐⭐ | 빠른 개발 가능 |
| 강화학습 훈련 | ⭐⭐⭐⭐ | 검증된 환경, 다만 대규모는 Isaac Gym 고려 |
| 정밀 물리 시뮬레이션 | ⭐⭐⭐ | MuJoCo가 더 정확 |
| 실제 로봇 배포 | ⭐⭐⭐ | Gazebo + ROS가 더 적합 |
| 4족 보행 로봇 | ⭐⭐⭐⭐⭐ | SpotMicro 등 많은 오픈소스 프로젝트 |

> **결론**: PyBullet은 2025년 현재도 로봇 시뮬레이션 입문자와 연구자에게 **최고의 선택지 중 하나**입니다. 특히 SpotMicroJetson 같은 4족 보행 로봇 프로젝트에서는 풍부한 커뮤니티 지원, 검증된 예제 코드, 빠른 개발 사이클이라는 장점이 분명합니다.

---

## 2. 프로젝트 환경 설정

> **예상 소요 시간**: 약 30분

### 2.1 프로젝트 클론 및 가상환경 생성

```bash
# SpotMicroJetson 프로젝트 클론
cd d:/git
git clone https://github.com/YOUR_USERNAME/SpotMicroJetson.git
cd SpotMicroJetson

# 표준 Python venv로 가상환경 생성 (권장)
# PyBullet 빌드 이슈 방지를 위해 표준 venv 사용
python -m venv .venv

# 가상환경 활성화
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate

# Python 버전 확인
python --version
# Python 3.12.x 출력되어야 함

# pip 업그레이드
python -m pip install --upgrade pip
```

> `git clone` 주소의 `YOUR_USERNAME`은 실제 저장소 소유자 계정으로 바꿔 사용합니다.

### 2.2 의존성 설치

```bash
# Simulation 디렉토리의 requirements.txt 설치
cd Simulation
pip install -r requirements.txt

# 또는 개별 설치
pip install gym==0.17.2
pip install matplotlib
pip install numpy
pip install pybullet==3.0.6
pip install keyboard

# gym-spotmicroai 패키지 설치 (개발 모드)
pip install -e .
```

---

## 3. 시뮬레이션 실행: pybullet_automatic_gait.py

> **예상 소요 시간**: 약 15분

### 3.1 시뮬레이션 실행

```bash
# Simulation 디렉토리에서 실행
cd d:/git/SpotMicroJetson/Simulation
python pybullet_automatic_gait.py
```

**실행 시 동작**

- PyBullet GUI 창이 열립니다.
- Spot Micro 로봇이 로드됩니다.
- 키보드로 로봇을 제어할 수 있습니다.
- Height 슬라이더로 로봇 높이를 조절할 수 있습니다.

**키보드 컨트롤(예상)**

| 키 | 기능 |
|----|------|
| 화살표 키 | 이동 방향 제어 |
| 스페이스바 | Trotting gait 시작/정지 |
| ESC | 종료 |

### 3.2 코드 구조 이해

```python
# pybullet_automatic_gait.py의 주요 구성 요소

# 1. 의존성
from environment import environment        # PyBullet 환경 설정
import spotmicroai                         # Spot Micro 로봇 모델
from kinematicMotion import TrottingGait   # Trotting 보행 알고리즘
from Common.multiprocess_kb import KeyInterrupt  # 키보드 입력

# 2. 로봇 초기화
robot = spotmicroai.Robot(True, True, reset)

# 3. Trotting Gait 초기화
trotting = TrottingGait()

# 4. 메인 루프
# - 로봇 위치/자세 읽기
# - 키보드 입력 처리
# - 발 위치 계산 (걷기/정지)
# - 로봇 제어 명령 전송
```

### 3.3 문제 해결 (Troubleshooting)

```bash
# PyBullet 설치 실패 시
# -> Microsoft C++ Build Tools 설치 필요
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
# 또는 안정 버전 사용
pip install pybullet==3.0.6

# ModuleNotFoundError: No module named 'Common'
# -> SpotMicroJetson 프로젝트 루트에서 실행하거나 PYTHONPATH 설정
cd d:/git/SpotMicroJetson
set PYTHONPATH=%CD%  # Windows CMD
$env:PYTHONPATH = $PWD  # Windows PowerShell
export PYTHONPATH=$(pwd)  # Linux/Mac
cd Simulation
python pybullet_automatic_gait.py

# keyboard 라이브러리 권한 오류 (Linux)
sudo pip install keyboard
# 또는
sudo python pybullet_automatic_gait.py

# PyBullet GUI 창이 안 열릴 때
# -> environment.py에서 p.connect(p.GUI) 확인

# 가상환경 재생성이 필요한 경우
cd d:/git/SpotMicroJetson
deactivate
Remove-Item -Recurse -Force .venv  # PowerShell
# 또는 rm -rf .venv  # CMD/Linux
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
cd Simulation
pip install -r requirements.txt
pip install -e .
```

---

## 4. PyBullet 기초 학습 (선택)

> **예상 소요 시간**: 약 1시간 · SpotMicroJetson 프로젝트를 더 깊이 이해하고 싶을 때 진행합니다.

### 4.1 로봇 관절 정보 분석

SpotMicroJetson 프로젝트의 로봇 모델(URDF) 관절 정보를 분석합니다.

```python
# analyze_spotmicro.py
import sys
sys.path.append("D:/git/SpotMicroJetson")

from Simulation import environment
import pybullet as p

# 환경 초기화
env = environment.environment()

# 로봇 ID 가져오기 (environment.py에서 로드됨)
# URDF 경로: SpotMicroJetson/spot_description/urdf/spotmicro.urdf

# 관절 정보 출력
num_joints = p.getNumJoints(env.robot)
print(f"Total joints: {num_joints}\n")

print("=" * 80)
print(f"{'Index':<6} {'Name':<30} {'Type':<15} {'Range':<20}")
print("=" * 80)

for i in range(num_joints):
    info = p.getJointInfo(env.robot, i)
    joint_name = info[1].decode('utf-8')
    joint_type = info[2]
    joint_lower = info[8]
    joint_upper = info[9]

    type_names = {
        p.JOINT_REVOLUTE: "REVOLUTE",
        p.JOINT_PRISMATIC: "PRISMATIC",
        p.JOINT_FIXED: "FIXED"
    }

    if joint_type == p.JOINT_REVOLUTE:
        print(f"{i:<6} {joint_name:<30} {type_names[joint_type]:<15} [{joint_lower:.2f}, {joint_upper:.2f}]")

print("=" * 80)
```

---

## 5. 학습 정리

> **예상 소요 시간**: 약 15분

### 5.1 학습 내용 정리

**완료한 작업**

1. Python 가상환경 설정 및 PyBullet 설치
2. SpotMicroJetson 프로젝트 클론 및 환경 구축
3. 의존성 패키지 설치(gym, pybullet, numpy, matplotlib, keyboard)
4. `pybullet_automatic_gait.py` 실행 및 동작 확인
5. SpotMicro 로봇 시뮬레이션 이해

**핵심 개념**

- SpotMicroJetson 프로젝트의 디렉터리 구조
- PyBullet 시뮬레이션 환경 설정
- Trotting Gait 보행 패턴
- 키보드를 통한 로봇 제어

### 5.2 프로젝트 구조 이해

```
SpotMicroJetson/
├── Simulation/          # 시뮬레이션 관련 코드
│   ├── environment.py   # PyBullet 환경 설정
│   ├── spotmicroai.py   # 로봇 제어 클래스
│   ├── kinematics.py    # 역기구학 계산
│   ├── kinematicMotion.py  # 보행 패턴 (TrottingGait)
│   └── pybullet_automatic_gait.py  # 메인 실행 파일
├── Common/              # 공통 유틸리티
│   └── multiprocess_kb.py  # 키보드 입력 처리
└── .venv/              # 가상환경
```

### 5.3 다음 주(Week 03) 학습 계획

**Week 03 주제**: PyBullet 심화 — 기본 제어

- `setJointMotorControl2` API 상세 분석
- Position / Velocity / Torque 제어 모드 비교
- PD 게인 튜닝 방법
- Spot Mini를 서 있는 자세로 제어하기
- 순기구학(T0~T4)을 통한 다리 관절 위치 계산

---

## 6. 완료 체크리스트

- [ ] Python 가상환경 생성 및 활성화
- [ ] SpotMicroJetson 프로젝트 클론
- [ ] 의존성 패키지 설치(requirements.txt)
- [ ] `pybullet_automatic_gait.py` 실행 성공
- [ ] 로봇 움직임 관찰 및 키보드 컨트롤 테스트
- [ ] 프로젝트 구조 이해
- [ ] 다음 주 학습 계획 확인

---

## 7. 참고 자료

**PyBullet 공식 문서**

- https://pybullet.org/wordpress/
- https://github.com/bulletphysics/bullet3

**SpotMicro 관련 프로젝트**

- https://github.com/FlorianWilk/SpotMicroAI
- https://github.com/OpenQuadruped/spot_mini_mini

**관련 주제**

- Spot Mini 로봇 역기구학
- Trotting Gait 보행 패턴
- Quadruped 로봇 제어
