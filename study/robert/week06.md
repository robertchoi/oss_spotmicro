# 6주차: Trot Gait 패턴과 제자리 걷기 시뮬레이션

6주차는 4족 보행 로봇의 가장 기본적인 보행 패턴인 Trot Gait를 학습합니다. Swing/Stance phase의 개념을 이해하고, 소스 코드를 분석하여 실제로 제자리 걷기 시뮬레이션을 구현합니다.

## 학습 목표

1. Trot Gait 패턴의 이론적 이해
2. Swing/Stance phase 구현 원리
3. kinematicMotion.py의 TrottingGait 클래스 분석
4. 제자리 걷기 시뮬레이션 구현 및 실행

---

## 1. Trot Gait 패턴 이해

### 1.1 4족 보행 로봇의 Gait 종류

4족 보행 로봇에는 다양한 보행 패턴(Gait)이 있습니다:

| Gait 종류 | 특징 | 속도 | 안정성 |
|-----------|------|------|--------|
| **Walk** | 한 번에 한 발만 들음 | 느림 | 매우 높음 |
| **Trot** | 대각선 다리 쌍이 함께 움직임 | 중간 | 높음 |
| **Pace** | 같은 쪽 다리가 함께 움직임 | 중간 | 중간 |
| **Gallop** | 앞다리와 뒷다리가 따로 움직임 | 빠름 | 낮음 |

### 1.2 Trot Gait란?

Trot Gait는 **대각선에 위치한 두 다리가 동시에 움직이는** 보행 패턴입니다.

```
[Trot Gait 다리 쌍]

   Front                     Front
    ┌─────────┐              ┌─────────┐
    │ FL   FR │              │ FL   FR │
    │ ●    ○  │  ────→       │ ○    ●  │
    │         │   Phase      │         │
    │ ○    ●  │   전환       │ ●    ○  │
    │ RL   RR │              │ RL   RR │
    └─────────┘              └─────────┘
   Rear                      Rear

   ● = Swing (발을 들고 앞으로 이동)
   ○ = Stance (발이 지면에 닿아 지지)
```

**Pair 1**: Front Left (FL) + Rear Right (RR)
**Pair 2**: Front Right (FR) + Rear Left (RL)

### 1.3 Trot Gait의 장점

1. **정적 안정성**: 항상 2개의 발이 지면에 있어 삼각형 지지 기반 형성
2. **동적 균형**: 대각선 다리가 균형을 잡아 놓쳐도 안정적
3. **에너지 효율**: 중간 속도에서 효율적인 에너지 사용
4. **구현 용이성**: 2개의 phase만 관리하면 됨

---

## 2. Swing/Stance Phase 구현

### 2.1 Phase 개념

한 걸음(Gait Cycle)은 크게 두 가지 phase로 구성됩니다:

```
[한 걸음 주기 (Gait Cycle)]

┌───────────────────────────────────────────┐
│                 한 걸음                    │
├─────────────────────┬─────────────────────┤
│     Swing Phase     │    Stance Phase     │
│   (발을 들고 이동)   │  (지면에서 밀기)     │
│      0% ~ 50%       │     50% ~ 100%      │
└─────────────────────┴─────────────────────┘

시간 →
```

#### Swing Phase (스윙 상)
- 발을 들어올려 앞으로 이동
- 발끝이 공중에 있음
- 일반적으로 타원 또는 반원 궤적

#### Stance Phase (지지 상)
- 발이 지면에 닿아 있음
- 몸체를 앞으로 밀어주는 역할
- 직선 궤적 (지면 위에서)

### 2.2 발끝 궤적 설계

```python
import numpy as np
import math

class FootTrajectory:
    """
    한 발의 발끝 궤적을 생성하는 클래스
    """
    def __init__(self, step_length=60, step_height=40, ground_height=-150):
        """
        Parameters:
        -----------
        step_length : float
            한 걸음의 보폭 (mm)
        step_height : float
            발을 들어올리는 높이 (mm)
        ground_height : float
            지면 높이 (몸체 기준, mm)
        """
        self.step_length = step_length
        self.step_height = step_height
        self.ground_height = ground_height

    def get_position(self, phase):
        """
        phase에 따른 발끝 위치 계산

        Parameters:
        -----------
        phase : float
            0.0 ~ 1.0 사이의 값
            0.0 ~ 0.5: Swing phase
            0.5 ~ 1.0: Stance phase

        Returns:
        --------
        x, z : float
            발끝의 X좌표(전후)와 Z좌표(상하)
        """
        Sl = self.step_length
        Sh = self.step_height
        ground_z = self.ground_height

        if phase < 0.5:
            # ===== Swing Phase =====
            # 반원 궤적으로 발을 앞으로 이동
            swing_phase = phase * 2  # 0 ~ 1로 정규화

            # X: 뒤에서 앞으로 이동
            x = -Sl/2 + Sl * swing_phase

            # Z: 반원 궤적 (sin 곡선)
            z = ground_z + Sh * math.sin(math.pi * swing_phase)
        else:
            # ===== Stance Phase =====
            # 지면에 닿아 직선으로 복귀
            stance_phase = (phase - 0.5) * 2  # 0 ~ 1로 정규화

            # X: 앞에서 뒤로 이동
            x = Sl/2 - Sl * stance_phase

            # Z: 지면 높이 유지
            z = ground_z

        return x, z
```

### 2.3 Swing/Stance 궤적 시각화

```python
import matplotlib.pyplot as plt

def visualize_foot_trajectory():
    """
    발끝 궤적을 시각화
    """
    traj = FootTrajectory(step_length=60, step_height=40, ground_height=-150)

    phases = np.linspace(0, 1, 100)
    positions = [traj.get_position(p) for p in phases]
    x_vals = [p[0] for p in positions]
    z_vals = [p[1] for p in positions]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Swing phase (파란색)
    swing_idx = 50
    ax.plot(x_vals[:swing_idx], z_vals[:swing_idx],
            'b-', linewidth=3, label='Swing Phase (발 들림)')

    # Stance phase (빨간색)
    ax.plot(x_vals[swing_idx:], z_vals[swing_idx:],
            'r-', linewidth=3, label='Stance Phase (지면 접촉)')

    # 시작/끝 점 표시
    ax.scatter(x_vals[0], z_vals[0], color='green', s=100, zorder=5, label='Start')
    ax.scatter(x_vals[-1], z_vals[-1], color='orange', s=100, zorder=5, label='End')

    # 화살표로 방향 표시
    ax.annotate('', xy=(x_vals[25], z_vals[25]),
                xytext=(x_vals[20], z_vals[20]),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    ax.annotate('', xy=(x_vals[75], z_vals[75]),
                xytext=(x_vals[70], z_vals[70]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.set_xlabel('X Position (mm)', fontsize=12)
    ax.set_ylabel('Z Position (mm)', fontsize=12)
    ax.set_title('Foot Trajectory - Swing/Stance Phase', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

# 실행
visualize_foot_trajectory()
```

---

## 3. TrottingGait 소스 코드 분석

### 3.1 kinematicMotion.py의 TrottingGait 클래스

SpotMicroAI 프로젝트의 `Kinematics/kinematicMotion.py`에 있는 TrottingGait 클래스를 분석합니다.

```python
"""
This class will define the trotting-gait function
A complete cycle is done in Tt
Each leg has the following "states"
0 - wait on ground for t0
1 - move on ground for steplength Sl for t1
2 - wait on ground for t2
3 - lift leg by Sh and Sl for t3 back to 0
"""
class TrottingGait:

    def __init__(self):
        self.step_gain = 0.8
        self.maxSl=2
        self.bodyPos=(0,100,0)
        self.bodyRot=(0,0,0)

        # 시간 파라미터 (밀리초 단위)
        self.t0=300   # State 0: 지면에서 대기 시간
        self.t1=1200  # State 1: 지면에서 이동 시간 (Stance 후반)
        self.t2=300   # State 2: 지면에서 대기 시간
        self.t3=200   # State 3: 발을 들어올리는 시간 (Swing)

        # 거리 파라미터
        self.Sl=0.0   # Step Length (보폭)
        self.Sw=0     # Step Width (좌우 이동)
        self.Sh=40    # Step Height (발 들어올리는 높이)
        self.Sa=0     # Step Alpha (회전)

        # 다리 오프셋
        self.Spf=87   # Front leg spur width
        self.Spr=77   # Rear leg spur width
        self.Fo=120   # Front leg X offset
        self.Ro=50    # Rear leg X offset

        self.Rc=[-50,0,0,1]  # rotation center
```

### 3.2 상태 다이어그램

TrottingGait는 4개의 상태(state)를 순환합니다:

```
[Gait State Machine]

    ┌─────────────────────────────────────────────────────────┐
    │               Total Time (Tt = t0+t1+t2+t3)             │
    ├──────────┬────────────────┬──────────┬─────────────────┤
    │  State 0 │    State 1     │  State 2 │     State 3     │
    │   (t0)   │     (t1)       │   (t2)   │      (t3)       │
    │   Wait   │  Ground Move   │   Wait   │   Lift & Swing  │
    │  on GND  │  (Stance)      │  on GND  │    (Swing)      │
    │  300ms   │   1200ms       │  300ms   │     200ms       │
    └──────────┴────────────────┴──────────┴─────────────────┘

    ←────────── Stance Phase ──────────→←── Swing Phase ──→
```

### 3.3 calcLeg() 함수 분석

```python
def calcLeg(self, t, x, y, z):
    """
    시간 t에서 다리의 발끝 위치를 계산

    Parameters:
    -----------
    t : float
        현재 시간 (밀리초, Tt로 나눈 나머지)
    x, y, z : float
        다리의 기본 위치

    Returns:
    --------
    curLp : numpy array
        현재 발끝 위치 [x, y, z, 1]
    """
    # 시작 위치 (보폭의 절반만큼 뒤)
    startLp = np.array([x - self.Sl/2.0, y, z - self.Sw, 1])

    # 끝 위치 (보폭의 절반만큼 앞)
    endY = 0
    endLp = np.array([x + self.Sl/2, y + endY, z + self.Sw, 1])

    if t < self.t0:
        # State 0: 시작 위치에서 대기
        return startLp

    elif t < self.t0 + self.t1:
        # State 1: 지면에서 startLp → endLp로 이동 (Stance)
        td = t - self.t0
        tp = 1 / (self.t1 / td)  # 진행률 (0~1)

        diffLp = endLp - startLp
        curLp = startLp + diffLp * tp

        # 회전 적용 (선택적)
        psi = -((math.pi/180*self.Sa)/2) + (math.pi/180*self.Sa)*tp
        Ry = np.array([[np.cos(psi), 0, np.sin(psi), 0],
                       [0, 1, 0, 0],
                       [-np.sin(psi), 0, np.cos(psi), 0],
                       [0, 0, 0, 1]])
        curLp = Ry.dot(curLp)
        return curLp

    elif t < self.t0 + self.t1 + self.t2:
        # State 2: 끝 위치에서 대기
        return endLp

    elif t < self.t0 + self.t1 + self.t2 + self.t3:
        # State 3: 발을 들어 endLp → startLp로 이동 (Swing)
        td = t - (self.t0 + self.t1 + self.t2)
        tp = 1 / (self.t3 / td)  # 진행률 (0~1)

        diffLp = startLp - endLp
        curLp = endLp + diffLp * tp

        # 발 들어올리기 (sin 곡선)
        curLp[1] += self.Sh * math.sin(math.pi * tp)
        return curLp
```

### 3.4 positions() 함수 분석

```python
def positions(self, t, kb_offset={}):
    """
    시간 t에서 4개 다리의 발끝 위치를 계산

    Parameters:
    -----------
    t : float
        현재 시간 (초)
    kb_offset : dict
        키보드 입력 오프셋 (IDstepLength, IDstepWidth, IDstepAlpha)

    Returns:
    --------
    r : numpy array
        4개 다리의 발끝 위치 [[FL], [FR], [RL], [RR]]
    """
    spf = self.Spf  # Front spur width
    spr = self.Spr  # Rear spur width

    # 키보드 입력으로 보폭 설정
    if list(kb_offset.values()) == [0.0, 0.0, 0.0]:
        self.Sl = 0.0  # 정지
        self.Sw = 0.0
        self.Sa = 0.0
    else:
        self.Sl = kb_offset['IDstepLength']
        self.Sw = kb_offset['IDstepWidth']
        self.Sa = kb_offset['IDstepAlpha']

    # 총 주기 시간
    Tt = self.t0 + self.t1 + self.t2 + self.t3  # 2000ms
    Tt2 = Tt / 2  # 1000ms (반 주기)

    # 시간을 밀리초로 변환하고 주기로 나눈 나머지
    td = (t * 1000) % Tt        # FL, RR용
    t2 = (t * 1000 - Tt2) % Tt  # FR, RL용 (반 주기 차이)

    # 다리 기본 위치
    Fx = self.Fo    # Front X offset (120)
    Rx = -self.Ro   # Rear X offset (-50)
    Fy = -100       # Front Y (높이)
    Ry = -100       # Rear Y (높이)

    # 4개 다리 위치 계산
    # FL과 RR은 같은 phase (td)
    # FR과 RL은 반대 phase (t2)
    r = np.array([
        self.calcLeg(td, Fx, Fy, spf),   # Front Left
        self.calcLeg(t2, Fx, Fy, -spf),  # Front Right
        self.calcLeg(t2, Rx, Ry, spr),   # Rear Left
        self.calcLeg(td, Rx, Ry, -spr)   # Rear Right
    ])

    return r
```

### 3.5 핵심 포인트 정리

1. **Phase 동기화**: FL+RR (td)과 FR+RL (t2)이 반 주기(Tt/2) 차이로 움직임
2. **4가지 State**: Wait → Ground Move → Wait → Lift (State 0~3)
3. **Sin 곡선**: Swing phase에서 발을 부드럽게 들어올림
4. **키보드 제어**: kb_offset으로 실시간 보폭 조절 가능

---

## 4. 제자리 걷기 시뮬레이션

### 4.1 제자리 걷기 구현

제자리 걷기는 `stepLength=0`으로 설정하여 발을 들었다 내리는 동작만 수행합니다.

```python
import sys
sys.path.append("..")

import numpy as np
import time
import math
from Kinematics.kinematicMotion import TrottingGait

class InPlaceWalking:
    """
    제자리 걷기 시뮬레이션 클래스
    """
    def __init__(self):
        self.trotting = TrottingGait()

        # 제자리 걷기 설정 (보폭 = 0)
        self.kb_offset = {
            'IDstepLength': 0.0,  # 전후 보폭 없음
            'IDstepWidth': 0.0,   # 좌우 이동 없음
            'IDstepAlpha': 0.0    # 회전 없음
        }

        # 시작 시간
        self.start_time = time.time()

    def get_foot_positions(self):
        """
        현재 시간에서 4개 다리의 발끝 위치 반환
        """
        current_time = time.time() - self.start_time
        positions = self.trotting.positions(current_time, self.kb_offset)
        return positions

    def print_positions(self):
        """
        발끝 위치를 콘솔에 출력
        """
        positions = self.get_foot_positions()
        leg_names = ['Front Left', 'Front Right', 'Rear Left', 'Rear Right']

        print("\n" + "="*60)
        print(f"Time: {time.time() - self.start_time:.2f}s")
        print("-"*60)
        for i, (name, pos) in enumerate(zip(leg_names, positions)):
            print(f"{name:12} | X: {pos[0]:7.2f} | Y: {pos[1]:7.2f} | Z: {pos[2]:7.2f}")
        print("="*60)


# 실행 예시
if __name__ == "__main__":
    walker = InPlaceWalking()

    try:
        for _ in range(100):
            walker.print_positions()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n시뮬레이션 종료")
```

### 4.2 PyBullet 시뮬레이션

```python
import pybullet as p
import pybullet_data
import time
import numpy as np
import math
import sys
sys.path.append("..")

from Kinematics.kinematicMotion import TrottingGait
from Kinematics.kinematics import Kinematic

class PyBulletInPlaceWalking:
    """
    PyBullet에서 제자리 걷기 시뮬레이션
    """
    def __init__(self):
        # PyBullet 초기화
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # 지면 및 로봇 로드
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("../urdf/spotmicroai.urdf", [0, 0, 0.3])

        # 보행 패턴 및 기구학
        self.trotting = TrottingGait()
        self.kinematics = Kinematic()

        # 관절 인덱스 (URDF 구조에 따라 다를 수 있음)
        self.joint_indices = {
            'FL': [0, 1, 2],   # Front Left: shoulder, leg, foot
            'FR': [3, 4, 5],   # Front Right
            'RL': [6, 7, 8],   # Rear Left
            'RR': [9, 10, 11]  # Rear Right
        }

        # PD 제어 파라미터
        self.kp = 0.02
        self.kd = 0.3
        self.max_force = 15

        # 시작 시간
        self.start_time = time.time()

        # GUI 슬라이더 추가
        self.height_slider = p.addUserDebugParameter("Height", -50, 50, 20)
        self.step_height_slider = p.addUserDebugParameter("Step Height", 0, 100, 40)

    def get_foot_positions(self, t):
        """
        시간 t에서 발끝 위치 반환 (제자리 걷기)
        """
        kb_offset = {
            'IDstepLength': 0.0,
            'IDstepWidth': 0.0,
            'IDstepAlpha': 0.0
        }
        return self.trotting.positions(t, kb_offset)

    def foot_to_angles(self, foot_pos):
        """
        발끝 위치를 관절 각도로 변환 (IK)
        """
        x, y, z, _ = foot_pos
        angles = self.kinematics.legIK([x, y, z, 1])
        return angles

    def set_leg_angles(self, leg_name, angles):
        """
        한 다리의 관절 각도 설정
        """
        joint_ids = self.joint_indices[leg_name]
        for joint_id, angle in zip(joint_ids, angles):
            p.setJointMotorControl2(
                self.robot,
                joint_id,
                p.POSITION_CONTROL,
                targetPosition=angle,
                force=self.max_force,
                positionGain=self.kp,
                velocityGain=self.kd
            )

    def step(self):
        """
        한 시뮬레이션 스텝 실행
        """
        t = time.time() - self.start_time

        # GUI에서 파라미터 읽기
        height = p.readUserDebugParameter(self.height_slider)
        step_height = p.readUserDebugParameter(self.step_height_slider)
        self.trotting.Sh = step_height

        # 발끝 위치 계산
        foot_positions = self.get_foot_positions(t)
        leg_names = ['FL', 'FR', 'RL', 'RR']

        # 각 다리에 IK 적용
        for leg_name, foot_pos in zip(leg_names, foot_positions):
            try:
                # 높이 보정
                foot_pos_adjusted = foot_pos.copy()
                foot_pos_adjusted[1] += height

                angles = self.foot_to_angles(foot_pos_adjusted)
                self.set_leg_angles(leg_name, angles)
            except Exception as e:
                print(f"IK error for {leg_name}: {e}")

        p.stepSimulation()

    def run(self, duration=60):
        """
        시뮬레이션 실행
        """
        print("="*50)
        print("제자리 걷기 시뮬레이션 시작")
        print("Ctrl+C로 종료")
        print("="*50)

        try:
            while True:
                self.step()
                time.sleep(1./240.)
        except KeyboardInterrupt:
            print("\n시뮬레이션 종료")
        finally:
            p.disconnect()


# 실행
if __name__ == "__main__":
    sim = PyBulletInPlaceWalking()
    sim.run()
```

### 4.3 Matplotlib 애니메이션

PyBullet 없이 Matplotlib으로 발끝 궤적을 애니메이션으로 시각화합니다.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import sys
sys.path.append("..")

from Kinematics.kinematicMotion import TrottingGait

def animate_in_place_walking():
    """
    제자리 걷기 발끝 궤적 애니메이션
    """
    trotting = TrottingGait()
    trotting.Sh = 40  # 발 들어올리는 높이

    kb_offset = {
        'IDstepLength': 0.0,
        'IDstepWidth': 0.0,
        'IDstepAlpha': 0.0
    }

    fig = plt.figure(figsize=(14, 6))

    # 측면도 (X-Y, 높이)
    ax1 = fig.add_subplot(121)
    ax1.set_xlim(-200, 200)
    ax1.set_ylim(-200, 0)
    ax1.set_xlabel('X Position (mm)')
    ax1.set_ylabel('Y Position (Height, mm)')
    ax1.set_title('Side View - Foot Height')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 상단도 (X-Z)
    ax2 = fig.add_subplot(122)
    ax2.set_xlim(-200, 200)
    ax2.set_ylim(-150, 150)
    ax2.set_xlabel('X Position (mm)')
    ax2.set_ylabel('Z Position (mm)')
    ax2.set_title('Top View - Foot Positions')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 다리 색상
    colors = ['blue', 'red', 'green', 'orange']
    leg_names = ['FL', 'FR', 'RL', 'RR']

    # 초기 점
    dots1 = [ax1.scatter([], [], c=c, s=100, label=n)
             for c, n in zip(colors, leg_names)]
    dots2 = [ax2.scatter([], [], c=c, s=100, label=n)
             for c, n in zip(colors, leg_names)]

    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

    def update(frame):
        t = frame / 30.0  # 30 FPS 가정

        positions = trotting.positions(t, kb_offset)

        for i, (pos, dot1, dot2) in enumerate(zip(positions, dots1, dots2)):
            # 측면도 (X, Y)
            dot1.set_offsets([[pos[0], pos[1]]])
            # 상단도 (X, Z)
            dot2.set_offsets([[pos[0], pos[2]]])

        ax1.set_title(f'Side View - Time: {t:.2f}s')
        ax2.set_title(f'Top View - Time: {t:.2f}s')

        return dots1 + dots2

    anim = FuncAnimation(fig, update, frames=200, interval=33, blit=True)
    plt.tight_layout()
    plt.show()

# 실행
animate_in_place_walking()
```

---

## 5. 실습 과제

### 5.1 기본 과제

1. **TrottingGait 파라미터 변경 실험**
   - `t0`, `t1`, `t2`, `t3` 값을 변경하고 결과 관찰
   - `Sh` (step height)를 변경하고 발 들어올리는 높이 확인

2. **Phase 시각화**
   - 각 다리의 Swing/Stance phase를 색상으로 구분하여 시각화
   - 대각선 쌍이 동시에 움직이는지 확인

### 5.2 심화 과제

1. **전진 보행 구현**
   - `IDstepLength`를 0이 아닌 값으로 설정
   - 로봇이 실제로 앞으로 이동하는지 확인

2. **회전 보행 구현**
   - `IDstepAlpha`를 조정하여 제자리 회전 구현

3. **보행 속도 조절**
   - Tt (총 주기 시간)을 변경하여 빠른/느린 보행 실험

---

## 6. 실제 소스 코드 참조 가이드

이 섹션에서는 SpotMicroJetson 프로젝트의 실제 구현 코드를 파일별로 정리합니다.

### 6.1 핵심 파일 구조

```
SpotMicroJetson/
├── Kinematics/
│   ├── kinematicMotion.py   ← TrottingGait 클래스 (보행 패턴)
│   └── kinematics.py        ← Kinematic 클래스 (IK/FK)
├── Simulation/
│   ├── pybullet_automatic_gait.py  ← PyBullet 시뮬레이션
│   └── spotmicroai.py       ← Robot 클래스 (로봇 제어)
├── Common/
│   └── multiprocess_kb.py   ← 키보드 입력 처리
└── JetsonNano/
    └── start_automatic_gait.py  ← 실제 로봇 실행
```

---

### 6.2 TrottingGait 클래스 상세 분석

📁 **파일**: `Kinematics/kinematicMotion.py` (Line 69-155)

#### 초기화 파라미터 (Line 71-89)

| 변수 | 값 | 설명 |
|------|-----|------|
| `t0` | 300ms | State 0: 지면 대기 시간 |
| `t1` | 1200ms | State 1: 지면 이동 시간 (Stance) |
| `t2` | 300ms | State 2: 지면 대기 시간 |
| `t3` | 200ms | State 3: 발 들어올림 (Swing) |
| `Sl` | 0.0 | 보폭 (Step Length) |
| `Sh` | 40mm | 발 들어올리는 높이 |
| `Spf` | 87mm | 앞다리 좌우 폭 |
| `Spr` | 77mm | 뒷다리 좌우 폭 |
| `Fo` | 120mm | 앞다리 X 오프셋 |
| `Ro` | 50mm | 뒷다리 X 오프셋 |

#### calcLeg() 함수 (Line 95-123) - 상태 전이

```python
# Line 100-101: State 0 - 시작 위치 대기
if(t<self.t0):
    return startLp

# Line 102-114: State 1 - Stance Phase (지면 이동)
elif(t<self.t0+self.t1):
    td=t-self.t0
    tp=1/(self.t1/td)  # 진행률 계산
    curLp=startLp+diffLp*tp  # 선형 보간

# Line 115-116: State 2 - 끝 위치 대기
elif(t<self.t0+self.t1+self.t2):
    return endLp

# Line 117-123: State 3 - Swing Phase (발 들어올림)
elif(t<self.t0+self.t1+self.t2+self.t3):
    curLp[1]+=self.Sh*math.sin(math.pi*tp)  # Y축으로 sin 곡선
```

#### positions() 함수 (Line 128-155) - 4다리 위치 계산

```python
# Line 142-148: 시간 계산
Tt=(self.t0+self.t1+self.t2+self.t3)  # 총 주기: 2000ms
Tt2=Tt/2  # 반주기: 1000ms

td=(t*1000)%Tt      # FL, RR용 시간
t2=(t*1000-Tt2)%Tt  # FR, RL용 시간 (반주기 차이)

# Line 153: 4다리 위치 배열 반환
r=np.array([
    self.calcLeg(td,Fx,Fy,spf),   # Front Left (FL)
    self.calcLeg(t2,Fx,Fy,-spf),  # Front Right (FR)
    self.calcLeg(rt2,Rx,Ry,spr),  # Rear Left (RL)
    self.calcLeg(rtd,Rx,Ry,-spr)  # Rear Right (RR)
])
```

---

### 6.3 Kinematic 클래스 (IK 구현)

📁 **파일**: `Kinematics/kinematics.py` (Line 24-173)

#### 링크 길이 정의 (Line 26-30)

```python
self.l1=50   # Shoulder offset
self.l2=20   # Shoulder length
self.l3=100  # Upper leg (대퇴)
self.l4=100  # Lower leg (하퇴)
```

#### legIK() - 역기구학 (Line 67-87)

```python
def legIK(self,point):
    (x,y,z)=(point[0],point[1],point[2])
    
    # Step 1: Shoulder 각도 계산
    F=sqrt(x**2+y**2-l1**2)
    theta1=-atan2(y,x)-atan2(F,-l1)
    
    # Step 2: Knee 각도 계산 (Cosine Law)
    G=F-l2  
    H=sqrt(G**2+z**2)
    D=(H**2-l3**2-l4**2)/(2*l3*l4)
    theta3=acos(D)
    
    # Step 3: Hip 각도 계산
    theta2=atan2(z,G)-atan2(l4*sin(theta3),l3+l4*cos(theta3))
    
    return(theta1,theta2,theta3)
```

#### calcIK() - 전체 IK (Line 163-173)

```python
def calcIK(self,Lp,angles,center):
    # Body IK 변환 행렬 계산
    (Tlf,Trf,Tlb,Trb)= self.bodyIK(omega,phi,psi,xm,ym,zm)
    
    # 4개 다리에 대해 IK 계산
    return np.array([
        self.legIK(np.linalg.inv(Tlf).dot(Lp[0])),  # FL
        self.legIK(Ix.dot(np.linalg.inv(Trf).dot(Lp[1]))),  # FR
        self.legIK(np.linalg.inv(Tlb).dot(Lp[2])),  # RL
        self.legIK(Ix.dot(np.linalg.inv(Trb).dot(Lp[3])))   # RR
    ])
```

---

### 6.4 Robot 클래스 (시뮬레이션)

📁 **파일**: `Simulation/spotmicroai.py` (Line 24-335)

#### 주요 파라미터 (Line 46-48)

```python
self.kp = 0.045   # Position gain
self.kd = 0.4     # Velocity gain  
self.maxForce = 25.0  # 최대 토크
```

#### feetPosition() - 발끝 위치 설정 (Line 255-256)

```python
def feetPosition(self,Lp):
    self.Lp=Lp  # TrottingGait.positions() 결과 저장
```

#### step() - 시뮬레이션 스텝 (Line 270-335)

```python
def step(self):
    # Line 295: IK 계산으로 관절 각도 얻기
    self.angles = self.kin.calcIK(self.Lp, self.rot, self.pos)
    
    # Line 301-310: 12개 관절에 각도 적용
    for lx, leg in enumerate(['front_left', 'front_right', 'rear_left', 'rear_right']):
        for px, part in enumerate(['shoulder', 'leg', 'foot']):
            j = self.jointNameToId[leg+"_"+part]
            p.setJointMotorControl2(
                bodyIndex=quadruped,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=self.angles[lx][px]*self.dirs[lx][px],
                positionGain=kp,
                velocityGain=kd,
                force=maxForce
            )
```

---

### 6.5 KeyInterrupt 클래스 (키보드 제어)

📁 **파일**: `Common/multiprocess_kb.py` (Line 17-99)

#### 제어 파라미터 (Line 30-32)

```python
self.X_STEP = 10.0   # 전후 보폭 증가량
self.Y_STEP = 5.0    # 좌우 보폭 증가량
self.YAW_STEP = 3.0  # 회전 각도 증가량
```

#### 키 매핑 (Line 49-51)

```python
# W/S: 전진/후진 보폭
command_dict['IDstepLength'] = self.X_STEP * result_dict['s'] - self.X_STEP * result_dict['w']

# A/D: 좌우 이동
command_dict['IDstepWidth'] = self.Y_STEP * result_dict['d'] - self.Y_STEP * result_dict['a']

# Q/E: 회전
command_dict['IDstepAlpha'] = self.YAW_STEP * result_dict['q'] - self.YAW_STEP * result_dict['e']
```

---

### 6.6 메인 실행 흐름

📁 **파일**: `Simulation/pybullet_automatic_gait.py` (Line 53-106)

```python
def main(id, command_status):
    # Line 55: 로봇 초기화
    robot = spotmicroai.Robot(False, True, reset)
    
    # Line 69: TrottingGait 인스턴스 생성
    trotting = TrottingGait()
    
    while True:
        # Line 87-88: 키보드 입력 읽기
        result_dict = command_status.get()
        command_status.put(result_dict)
        
        # Line 92-95: 보행 또는 정지
        if result_dict['StartStepping']:
            robot.feetPosition(trotting.positions(d-3, result_dict))
        else:
            robot.feetPosition(Lp)  # 기본 자세
        
        # Line 102: 시뮬레이션 스텝 실행
        robot.step()
```

---

### 6.7 데이터 흐름 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           데이터 흐름                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [KeyInterrupt]          [TrottingGait]           [Robot]               │
│       │                       │                      │                  │
│   키보드 입력             보행 패턴 생성          시뮬레이션 실행        │
│       │                       │                      │                  │
│       ▼                       ▼                      ▼                  │
│  ┌─────────┐            ┌──────────┐           ┌──────────┐            │
│  │ W/A/S/D │ ────────▶  │positions()│ ───────▶ │feetPosition│           │
│  │ Q/E키   │  kb_offset │  함수    │   Lp[4]  │  (Lp)    │            │
│  └─────────┘            └──────────┘           └──────────┘            │
│                               │                      │                  │
│                               ▼                      ▼                  │
│                         ┌──────────┐           ┌──────────┐            │
│                         │ calcLeg()│           │  calcIK()│            │
│                         │ 각 다리  │           │   IK     │            │
│                         └──────────┘           └──────────┘            │
│                                                      │                  │
│                                                      ▼                  │
│                                                ┌──────────┐            │
│                                                │  step()  │            │
│                                                │ 관절제어 │            │
│                                                └──────────┘            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 6.8 직접 실행하기

#### PyBullet 시뮬레이션 실행

```bash
cd Simulation
python pybullet_automatic_gait.py
```

**키보드 조작**:
- `W` / `S`: 전진 / 후진 보폭 증가
- `A` / `D`: 좌측 / 우측 이동
- `Q` / `E`: 좌회전 / 우회전
- `Space`: 보폭 초기화

#### 실제 로봇 실행 (JetsonNano)

```bash
cd JetsonNano
python start_automatic_gait.py
```

---

## 7. 정리

### 7.1 핵심 개념

| 개념 | 설명 |
|------|------|
| **Trot Gait** | 대각선 다리 쌍이 동시에 움직이는 보행 패턴 |
| **Swing Phase** | 발을 들어 앞으로 이동하는 구간 |
| **Stance Phase** | 지면에 닿아 몸을 밀어주는 구간 |
| **Gait Cycle** | 한 걸음의 완전한 주기 (Swing + Stance) |

### 7.2 소스 코드 위치

| 파일 | 설명 |
|------|------|
| `Kinematics/kinematicMotion.py` | TrottingGait 클래스 정의 |
| `Simulation/pybullet_automatic_gait.py` | PyBullet 시뮬레이션 예제 |
| `JetsonNano/start_automatic_gait.py` | 실제 로봇 제어 예제 |

### 7.3 다음 주 예고

7주차에서는 **전진 보행**과 **회전 보행**을 구현하고, 키보드 입력을 통해 실시간으로 로봇을 제어하는 방법을 학습합니다.

---

## 참고 자료

- [SpotMicroAI GitHub](https://github.com/FlorianWilk/SpotMicroAI)
- Understanding Quadruped Gaits: [MIT Cheetah Papers](https://biomimetics.mit.edu/)
- Trot Gait Analysis: Boston Dynamics Spot Robot
