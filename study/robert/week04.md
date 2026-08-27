# 4주차: Forward & Inverse Kinematics 구현

4주차는 SpotMicro의 다리 운동학을 이해하고 구현합니다. Forward Kinematics(FK)로 관절 각도에서 발끝 위치를 계산하고, Inverse Kinematics(IK)로 원하는 발끝 위치에 필요한 관절 각도를 계산합니다.

## FK와 IK 개념 비교

| 구분 | 정기구학 (Forward Kinematics, FK) ➡️ | 역기구학 (Inverse Kinematics, IK) ⬅️ |
|------|--------------------------------------|--------------------------------------|
| **핵심 질문** | "관절을 이만큼 움직이면, 손끝은 어디에 가 있을까?" | "손끝을 저 위치에 두려면, 관절을 각각 몇 도로 꺾어야 할까?" |
| **입력** | 각 관절의 각도 ($\theta_1, \theta_2, \dots$) | 엔드 이펙터(손끝)의 목표 위치와 방향 ($x, y, z$) |
| **출력** | 엔드 이펙터(손끝)의 위치와 방향 ($x, y, z$) | 그렇게 만들기 위한 각 관절의 각도 ($\theta_1, \theta_2, \dots$) |
| **계산 특징** | 명확하고 단순함. 각도를 정하면 손끝의 위치는 수학적으로 단 하나로 결정됨 | 복잡하고 어려움. 목표 지점에 손을 대는 방법은 여러 가지(다중 해)일 수도 있고, 아예 닿지 못할 수도 있음 |
| **비유** | 인형의 어깨를 30도, 팔꿈치를 45도 굽혔을 때 손의 위치를 재는 것 | 책상 위의 컵을 잡기 위해 뇌가 팔 관절의 각도를 계산하는 과정 |

## 4족 보행에서 FK/IK의 실전 활용

4족 로봇의 보행을 구현할 때 기구학은 **'다리의 움직임 디자인(IK)'**과 **'현재 상태 확인(FK)'**이라는 두 가지 핵심 역할을 수행합니다. 로봇이 땅을 딛고 앞으로 나아가는 과정은 생각보다 복잡한 계산의 연속입니다.

우리가 로봇에게 "10cm 앞으로 걸어가"라고 명령했을 때 벌어지는 일들을 단계별로 살펴볼까요?

### 1. 👣 역기구학(IK): 발끝의 궤적 그리기

로봇의 뇌(컴퓨터)는 먼저 다리 각 관절의 각도를 생각하는 것이 아니라, **'발끝이 공중에서 어떤 곡선을 그리며 이동해야 할지'**를 먼저 결정합니다. 이를 **궤적 계획(Trajectory Planning)**이라고 합니다.

**과정**: 발끝이 현재 좌표 $(x_0, y_0, z_0)$에서 목표 좌표 $(x_1, y_1, z_1)$까지 매끄러운 포물선을 그리며 움직이도록 설계합니다.

**사용**: 이때 **역기구학(IK)**이 필요합니다. 설계된 포물선 위의 수많은 점(좌표) 하나하나를 로봇이 실제로 구현하려면, 그 좌표에 맞는 **각 관절의 각도($\theta$)**를 순식간에 계산해 모터에 전달해야 하기 때문입니다.

### 2. ⚖️ 정기구학(FK): 몸체의 균형과 위치 파악

로봇이 걷는 동안 몸체가 너무 기울어지거나 뒤집히면 안 됩니다.

**과정**: 로봇은 각 다리의 모터에 달린 센서로부터 현재 관절 각도 데이터를 실시간으로 받습니다.

**사용**: 이 각도 데이터를 정기구학(FK) 식에 넣으면, 현재 네 다리의 끝단 위치가 몸체를 기준으로 어디에 있는지 정확히 알 수 있습니다. 이를 통해 **무게 중심(Center of Gravity)**이 안정적인 삼각형 안에 있는지 확인하고 균형을 잡습니다.

---

## Forward Kinematics (FK) 이해

Forward Kinematics는 관절 각도가 주어졌을 때 발끝(end-effector)의 위치를 계산하는 방법입니다.

```python
# SpotMicro 다리 구조
- 각 다리: 3-DOF (Abduction-Hip-Knee)
- Joint 1 (Abduction): 어깨 회전 (X축)
- Joint 2 (Hip): 고관절 (Y축)
- Joint 3 (Knee): 무릎 관절 (Y축)
```

### FK 변환 행렬

각 관절의 변환 행렬을 곱하여 발끝 위치를 계산합니다:

```python
def forward_kinematics(theta1, theta2, theta3, l1, l2, l3):
    """
    theta1: Abduction angle (rad)
    theta2: Hip angle (rad)
    theta3: Knee angle (rad)
    l1, l2, l3: Link lengths
    Returns: (x, y, z) position of foot
    """
    # Abduction joint (X-axis rotation)
    x = l1 * cos(theta1)

    # Hip and Knee joints (Y-axis plane)
    l_yz = l2 * cos(theta2) + l3 * cos(theta2 + theta3)
    y = l1 * sin(theta1) + l_yz * sin(theta1)
    z = -l2 * sin(theta2) - l3 * sin(theta2 + theta3)

    return x, y, z
```

**실습**: PyBullet에서 관절 각도를 변경하며 FK 계산 결과와 실제 발끝 위치(`getLinkState`) 비교

---

## Inverse Kinematics (IK) 수식 유도

Inverse Kinematics는 목표 발끝 위치 (x, y, z)가 주어졌을 때 필요한 관절 각도를 계산합니다.

### Geometric IK (해석적 해법)

SpotMicro의 다리 구조는 기하학적으로 IK를 풀 수 있습니다:

#### Step 1: Abduction Joint (θ₁)

```python
# Abduction angle
theta1 = atan2(y, x)
```

#### Step 2: Hip-Knee 평면 문제로 축소

abduction을 고려한 후, 2D 평면 문제로 단순화:

```python
# Distance from shoulder to target in YZ plane
r = sqrt(x² + y²) - l1
d = sqrt(r² + z²)  # 2D distance to target
```

#### Step 3: Hip Joint (θ₂)

Cosine law를 사용한 관절 각도 계산:

```python
# Cosine law for knee angle first
cos_theta3 = (d² - l2² - l3²) / (2 * l2 * l3)
theta3 = acos(cos_theta3)  # Knee angle

# Hip angle from geometry
alpha = atan2(-z, r)  # Angle to target
beta = acos((l2² + d² - l3²) / (2 * l2 * d))
theta2 = alpha - beta  # Hip angle
```

### IK 구현 코드

```python
import numpy as np

def inverse_kinematics(x, y, z, l1, l2, l3):
    """
    Analytical IK for 3-DOF leg
    x, y, z: Target foot position
    l1: Shoulder link length
    l2: Upper leg length
    l3: Lower leg length
    Returns: (theta1, theta2, theta3) in radians
    """
    # Joint 1: Abduction
    theta1 = np.arctan2(y, x)

    # Project to 2D plane
    r = np.sqrt(x**2 + y**2) - l1
    d = np.sqrt(r**2 + z**2)

    # Check reachability
    if d > (l2 + l3) or d < abs(l2 - l3):
        raise ValueError("Target unreachable")

    # Joint 3: Knee (elbow-down configuration)
    cos_theta3 = (d**2 - l2**2 - l3**2) / (2 * l2 * l3)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = -np.arccos(cos_theta3)  # Negative for elbow-down

    # Joint 2: Hip
    alpha = np.arctan2(-z, r)
    beta = np.arctan2(l3 * np.sin(theta3), l2 + l3 * np.cos(theta3))
    theta2 = alpha - beta

    return theta1, theta2, theta3
```

**실습**:
1. 4개의 발을 정사각형 패턴에 배치하는 IK 계산
2. PyBullet에서 계산된 관절 각도로 제어하여 검증
3. 도달 불가능한 위치에 대한 오류 처리 확인

---

## SpotMicro 다리 구조 분석

### 링크 길이 측정

URDF 파일에서 실제 링크 길이를 추출합니다:

```python
# PyBullet에서 링크 정보 추출
import pybullet as p

robot = p.loadURDF("spotmicro.urdf")

# 각 다리의 링크 길이
link_lengths = {}
for i in range(p.getNumJoints(robot)):
    joint_info = p.getJointInfo(robot, i)
    link_name = joint_info[12].decode('utf-8')
    # URDF에서 length 파라미터 확인
    print(f"Link {i}: {link_name}")
```

### 전형적인 SpotMicro 치수

```python
# 예시 링크 길이 (실제 모델에 따라 다름)
L_SHOULDER = 0.055  # l1: Shoulder offset (m)
L_UPPER_LEG = 0.110  # l2: Upper leg (m)
L_LOWER_LEG = 0.130  # l3: Lower leg (m)

# Workspace 계산
MAX_REACH = L_SHOULDER + L_UPPER_LEG + L_LOWER_LEG  # ~0.295m
MIN_REACH = abs(L_UPPER_LEG - L_LOWER_LEG)  # ~0.020m
```

### 다리 위치 (Home Stance)

```python
# Body frame 기준 4개 다리의 초기 위치
HOME_POSITIONS = {
    'front_left':  [0.15, 0.10, -0.20],
    'front_right': [0.15, -0.10, -0.20],
    'rear_left':   [-0.15, 0.10, -0.20],
    'rear_right':  [-0.15, -0.10, -0.20]
}
```

**실습**:
1. URDF 파싱하여 정확한 링크 길이 추출
2. 4개 다리 각각의 workspace 시각화
3. Home position에서 각 다리의 관절 각도 계산 (IK 사용)

---

## 통합 실습: IK 기반 보행 준비

이번 주 마지막 실습으로 IK를 활용한 간단한 발 궤적을 생성합니다:

```python
import numpy as np
import pybullet as p
import time

# 초기화
p.connect(p.GUI)
robot = p.loadURDF("spotmicro.urdf")

# 링크 길이 (실제 값으로 교체 필요)
L1, L2, L3 = 0.055, 0.110, 0.130

# 한 발을 원형 궤적으로 이동
t = 0
while True:
    # Circular trajectory
    x = 0.15 + 0.05 * np.cos(t)
    y = 0.10
    z = -0.20 + 0.03 * np.sin(t)

    # IK 계산
    theta1, theta2, theta3 = inverse_kinematics(x, y, z, L1, L2, L3)

    # PyBullet 제어 (front_left leg indices: 0, 1, 2)
    p.setJointMotorControl2(robot, 0, p.POSITION_CONTROL, theta1)
    p.setJointMotorControl2(robot, 1, p.POSITION_CONTROL, theta2)
    p.setJointMotorControl2(robot, 2, p.POSITION_CONTROL, theta3)

    p.stepSimulation()
    time.sleep(1/240)
    t += 0.01
```

**목표**:
- FK/IK 수식을 완전히 이해
- PyBullet과 MuJoCo 양쪽에서 IK 검증
- 다음 주 CPG 보행 생성을 위한 기반 마련

---

## 실제 소스 코드에서 IK/FK 활용 사례

SpotMicroJetson 프로젝트에서 IK와 FK가 어떻게 구현되고 사용되는지 살펴봅시다.

### 1. IK 구현: `Kinematics/kinematics.py`

#### `legIK()` 함수 - 역기구학 계산

[kinematics.py:67-87](d:\git\SpotMicroJetson\Kinematics\kinematics.py#L67-L87)

```python
def legIK(self, point):
    """
    한 다리의 역기구학 계산
    입력: 목표 발끝 위치 (x, y, z)
    출력: 3개 관절 각도 (theta1, theta2, theta3)
    """
    (x, y, z) = (point[0], point[1], point[2])
    (l1, l2, l3, l4) = (self.l1, self.l2, self.l3, self.l4)

    # Step 1: Abduction 각도 계산
    F = sqrt(x**2 + y**2 - l1**2)
    theta1 = -atan2(y, x) - atan2(F, -l1)

    # Step 2: Hip-Knee 평면 문제
    G = F - l2
    H = sqrt(G**2 + z**2)

    # Step 3: Knee 각도 (Cosine Law)
    D = (H**2 - l3**2 - l4**2) / (2 * l3 * l4)
    theta3 = acos(D)

    # Step 4: Hip 각도
    theta2 = atan2(z, G) - atan2(l4*sin(theta3), l3 + l4*cos(theta3))

    return (theta1, theta2, theta3)
```

**사용 위치**:
- [kinematics.py:115-122](d:\git\SpotMicroJetson\Kinematics\kinematics.py#L115-L122): `drawLegPair()` 함수에서 목표 발끝 위치를 받아 관절 각도를 계산
- [kinematics.py:163-173](d:\git\SpotMicroJetson\Kinematics\kinematics.py#L163-L173): `calcIK()` 함수에서 4개 다리 모두의 IK를 한번에 계산

#### `initIK()` 함수 - IK 기반 로봇 초기화

[kinematics.py:176-181](d:\git\SpotMicroJetson\Kinematics\kinematics.py#L176-L181)

```python
def initIK(Lp):
    """
    목표 발끝 위치 배열 Lp를 받아서 IK 계산
    Lp: 4x4 배열 [front_left, front_right, rear_left, rear_right]
    각 행: [x, y, z, 1] (homogeneous coordinates)
    """
    setupView(200).view_init(elev=12., azim=28)
    moduleKinematics = Kinematic()
    moduleKinematics.drawRobot(Lp, (0,0,0), (0,0,0))

    return moduleKinematics.thetas  # 계산된 관절 각도 반환
```

**실전 사용 예시**:

```python
# 4개 발의 목표 위치 설정 (Home Stance)
Lp = np.array([
    [100, -100, 100, 1],   # Front Left
    [100, -100, -100, 1],  # Front Right
    [-100, -100, 100, 1],  # Rear Left
    [-100, -100, -100, 1]  # Rear Right
])

# IK 계산으로 필요한 관절 각도 구하기
thetas = initIK(Lp)
# thetas는 4x3 배열: 각 다리의 [theta1, theta2, theta3]
```

### 2. FK 구현: `calcLegPoints()` 함수

#### 정기구학으로 다리 전체 링크 위치 계산

[kinematics.py:89-100](d:\git\SpotMicroJetson\Kinematics\kinematics.py#L89-L100)

```python
def calcLegPoints(self, angles):
    """
    정기구학: 관절 각도로부터 각 링크의 3D 위치 계산
    입력: (theta1, theta2, theta3)
    출력: 5개 점의 위치 [어깨, 관절1, 관절2, 관절3, 발끝]
    """
    (l1, l2, l3, l4) = (self.l1, self.l2, self.l3, self.l4)
    (theta1, theta2, theta3) = angles
    theta23 = theta2 + theta3

    T0 = np.array([0, 0, 0, 1])  # 어깨(시작점)
    T1 = T0 + np.array([-l1*cos(theta1), l1*sin(theta1), 0, 0])
    T2 = T1 + np.array([-l2*sin(theta1), -l2*cos(theta1), 0, 0])
    T3 = T2 + np.array([-l3*sin(theta1)*cos(theta2),
                        -l3*cos(theta1)*cos(theta2),
                        l3*sin(theta2), 0])
    T4 = T3 + np.array([-l4*sin(theta1)*cos(theta23),
                        -l4*cos(theta1)*cos(theta23),
                        l4*sin(theta23), 0])

    return np.array([T0, T1, T2, T3, T4])
```

**사용 위치**: 시각화 및 검증

- [kinematics.py:115-116](d:\git\SpotMicroJetson\Kinematics\kinematics.py#L115-L116): IK 계산 후 결과를 시각화할 때 FK로 다리 위치 재계산
- [kinematics.py:156-161](d:\git\SpotMicroJetson\Kinematics\kinematics.py#L156-L161): `drawRobotbyAngles()`에서 관절 각도만으로 로봇 전체를 그릴 때

#### `initFK()` 함수 - FK 기반 로봇 초기화

[kinematics.py:184-189](d:\git\SpotMicroJetson\Kinematics\kinematics.py#L184-L189)

```python
def initFK(La):
    """
    관절 각도 배열 La를 받아서 FK로 로봇 그리기
    La: 4x3 배열 [[theta1, theta2, theta3], ...] (4개 다리)
    """
    setupView(200).view_init(elev=12., azim=28)
    moduleKinematics = Kinematic()
    moduleKinematics.drawRobotbyAngles(La, (0,0,0), (0,0,0))

    return moduleKinematics.thetas
```

### 3. 보행 제어에서의 활용: `kinematicMotion.py`

#### Trotting Gait에서 IK 사용

[kinematicMotion.py:69-155](d:\git\SpotMicroJetson\Kinematics\kinematicMotion.py#L69-L155)

```python
class TrottingGait:
    def positions(self, t, kb_offset={}):
        """
        시간 t에 따른 4개 발의 목표 위치 계산 (궤적 계획)

        반환값: 4개 발의 목표 좌표 배열
        이 좌표들이 legIK()로 전달되어 관절 각도로 변환됨
        """
        # 시간에 따라 발끝 궤적 계산
        td = (t * 1000) % Tt  # Front-Left, Rear-Right 시간
        t2 = (t * 1000 - Tt2) % Tt  # Front-Right, Rear-Left 시간

        # 각 발의 목표 위치 계산 (IK의 입력이 됨)
        r = np.array([
            self.calcLeg(td, Fx, Fy, spf),    # Front-Left
            self.calcLeg(t2, Fx, Fy, -spf),   # Front-Right
            self.calcLeg(rt2, Rx, Ry, spr),   # Rear-Left
            self.calcLeg(rtd, Rx, Ry, -spr)   # Rear-Right
        ])
        return r  # -> 이 배열이 legIK()의 입력으로 사용됨
```

### 실제 동작 흐름

```
1. 보행 패턴 생성 (Trajectory Planning)
   TrottingGait.positions(t)
   └─> 시간 t에서 4개 발의 목표 좌표 [x, y, z] 계산

2. 역기구학 적용 (IK)
   legIK(x, y, z)
   └─> 목표 좌표를 관절 각도 [θ1, θ2, θ3]로 변환

3. 모터 제어
   setJointMotorControl2(joint_id, target_angle)
   └─> 실제 하드웨어에 관절 각도 명령

4. 상태 확인 (FK)
   getJointState() -> 현재 관절 각도 읽기
   calcLegPoints(angles) -> 현재 발끝 위치 계산
   └─> 균형 및 안정성 체크
```

---

## FK 없이는 로봇이 '눈을 감고' 걷는 것과 같습니다

로봇이 FK(정기구학)를 통해 자신의 상태를 '확인'하지 않는다면, 마치 **눈을 감고 귀를 막은 채 지시받은 대로만 몸을 움직이는 것**과 같습니다.

모터에게 "30도 움직여"라고 명령을 내렸더라도, 실제 세상에서는 여러 가지 이유로 오차가 발생합니다. FK로 현재 상태를 확인하지 않았을 때 발생할 수 있는 구체적인 문제들을 살펴봅시다.

### 1. 👣 내가 지금 어디를 딛고 있는지 모릅니다

로봇 다리가 바위 위를 밟았다고 가정해 봅시다. 모터는 명령받은 각도까지 가려고 하지만 바위 때문에 더 이상 움직이지 못할 수 있습니다.

**확인을 안 하면**: 로봇은 발이 평지에 있다고 착각하고 다음 걸음을 내딛다가 무게 중심이 무너져 넘어집니다. 📉

**FK를 사용하면**: 현재 멈춰선 모터 각도들을 FK 식에 대입해 "아, 내 발끝이 지금 예상보다 높은 위치($z$값)에서 멈췄구나! 여기가 바위 위구나!"라고 알아차릴 수 있습니다.

```python
# FK를 통한 지형 감지 예시
target_foot_z = -100  # 목표 발끝 높이
actual_angles = getJointState()  # 센서로 실제 각도 읽기
actual_foot_pos = calcLegPoints(actual_angles)  # FK 계산
actual_foot_z = actual_foot_pos[4][2]  # 발끝의 실제 z 좌표

if abs(actual_foot_z - target_foot_z) > 10:  # 오차 10mm 이상
    print(f"장애물 감지! 예상: {target_foot_z}, 실제: {actual_foot_z}")
    # 보행 패턴 수정 또는 높이 보정
```

### 2. 🎛️ 모터의 오차가 누적됩니다

모터는 완벽하지 않습니다. 10도를 움직이라고 해도 실제로는 9.9도만 움직일 수도 있고, 외부 하중 때문에 조금씩 밀릴 수도 있습니다.

**확인을 안 하면**: 이런 미세한 오차가 걸음마다 쌓여서, 로봇은 똑바로 걷고 있다고 생각하지만 실제로는 엉뚱한 방향으로 가고 있게 됩니다.

**FK를 사용하면**: 각 모터의 실제 각도 센서값을 읽어 현재 발의 정확한 위치를 계산하고, 목표 지점에서 벗어났다면 이를 수정할 수 있습니다.

```python
# 피드백 제어 루프
while walking:
    # 목표 위치로 IK 계산
    target_angles = legIK(target_pos)
    setJointAngles(target_angles)

    # 실제 도달한 위치를 FK로 확인
    actual_angles = getJointState()
    actual_pos = calcLegPoints(actual_angles)[4]  # 발끝 위치

    # 오차 계산 및 보정
    error = target_pos - actual_pos
    if np.linalg.norm(error) > threshold:
        # PID 제어로 오차 보정
        correction = pid_controller.update(error)
        target_angles += correction
```

### 3. ⚖️ 몸체의 실제 자세를 모릅니다

4족 로봇의 몸체는 네 다리의 끝점이 만드는 평면 위에 떠 있습니다.

**확인을 안 하면**: 한쪽 다리가 미끄러져서 각도가 변했는데도 모르면, 로봇은 몸체가 수평인 줄 알고 가다가 전복될 위험이 큽니다.

**FK를 사용하면**: 네 다리의 관절 각도를 종합해 "현재 내 몸체가 지면으로부터 몇 cm 높이에 있고, 어느 방향으로 기울어져 있는지"를 실시간으로 계산해 냅니다.

```python
# 4개 다리의 FK 계산으로 몸체 자세 추정
foot_positions = []
for leg_id in range(4):
    angles = getJointState(leg_id)
    foot_pos = calcLegPoints(angles)[4]  # 각 다리 발끝 위치
    foot_positions.append(foot_pos)

# 4개 발끝으로 평면 방정식 계산
body_plane = fit_plane(foot_positions)
body_height = body_plane.z_offset
body_roll = body_plane.roll_angle
body_pitch = body_plane.pitch_angle

# 안정성 체크
if abs(body_roll) > 15 or abs(body_pitch) > 15:
    print("경고: 몸체 기울기 과다! 자세 보정 필요")
```

### FK는 피드백(Feedback)의 핵심 도구

결국 FK는 **"이론상의 명령"**과 **"실제 로봇의 상태"** 사이의 간극을 메워주는 **피드백(Feedback)**의 핵심 도구입니다.

#### 실전 예시: 얼음 위에서 걷기

만약 로봇이 아주 매끄러운 얼음 위를 걷고 있다고 상상해 보세요. 발이 계속 미끄러지는 상황입니다.

```python
# FK + IMU 센서 융합으로 미끄러짐 감지
class SlipDetection:
    def detect_slip(self):
        # 1. FK로 계산한 몸체 위치
        fk_body_pos = self.calc_body_from_legs()

        # 2. IMU(자이로/가속도계)로 측정한 실제 위치
        imu_body_pos = self.imu.get_position()

        # 3. 차이 계산
        position_error = imu_body_pos - fk_body_pos

        # 4. 미끄러짐 판단
        if np.linalg.norm(position_error) > SLIP_THRESHOLD:
            return True, position_error
        return False, None

    def handle_slip(self, slip_vector):
        # 미끄러짐 감지 시 대응
        self.reduce_step_length()      # 보폭 줄이기
        self.increase_foot_friction()  # 발 압력 증가
        self.adjust_trajectory(slip_vector)  # 미끄러진 방향 보정
```

**FK와 센서 융합의 이점**:

1. **미끄러짐 감지**: 모터 각도로 계산한 위치(FK)와 실제 몸의 움직임(IMU)이 다름을 인지
2. **보행 속도 조절**: 미끄러운 것을 알고 보폭을 좁히거나 천천히 걸음
3. **위치 보정**: 미끄러진 만큼 다시 계산해서 원래 경로로 복귀

### 비유: 사람도 감각 없이는 못 걷습니다

사람이 눈을 감고 걸으면 어떻게 될까요? 처음 몇 걸음은 괜찮지만, 곧 비틀거리고 균형을 잃게 됩니다. 왜냐하면:

- 시각(Vision): 주변 환경과 발의 위치 확인 → 로봇의 **FK**에 해당
- 전정기관(Inner Ear): 몸의 기울기와 가속도 감지 → 로봇의 **IMU**에 해당
- 고유수용감각(Proprioception): 근육과 관절의 위치 인식 → 로봇의 **관절 센서**에 해당

로봇도 마찬가지입니다. FK 없이는 자신의 몸 상태를 모른 채 명령만 따르다가 넘어지거나 길을 잃게 됩니다.

---

## 주요 개념 정리

| 개념 | 설명 | 수식 |
|------|------|------|
| **FK** | 관절 각도 → 발끝 위치 | p = f(θ₁, θ₂, θ₃) |
| **IK** | 발끝 위치 → 관절 각도 | θ = f⁻¹(x, y, z) |
| **Workspace** | 발이 도달 가능한 공간 | l₂-l₃ ≤ d ≤ l₂+l₃ |
| **Singularity** | IK 해가 불안정한 자세 | 완전히 펴진/접힌 상태 |

**다음 주 예고**: 5주차에는 이번 주 구현한 IK를 활용하여 CPG(Central Pattern Generator) 기반 Trot 보행을 생성합니다!
