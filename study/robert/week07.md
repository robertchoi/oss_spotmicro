# 7주차: Body 자세 제어와 전진 보행 구현

7주차는 로봇 몸체(Body)의 자세 제어와 전진 보행을 학습합니다. Height/Pitch/Roll 제어의 원리, CoM(Center of Mass) 추적 개념, 그리고 실제 전진 보행 구현까지 단계별로 진행합니다.

## 학습 목표

1. Body Height/Pitch/Roll 제어 원리 이해
2. CoM(Center of Mass) 추적 개념 학습
3. 전진 보행(Forward Walking) 구현
4. 좌우 회전 보행 구현

---

## 1. Body Height/Pitch/Roll 제어

### 1.1 Body 자세 파라미터

로봇 몸체의 자세는 6자유도(6-DOF)로 표현됩니다:

```
[Body 6-DOF Parameters]

Position (위치)              Orientation (회전)
├── X: 전후 이동              ├── Roll (φ): X축 회전 (좌우 기울기)
├── Y: 상하 이동 (Height)     ├── Pitch (θ): Y축 회전 (전후 기울기)
└── Z: 좌우 이동              └── Yaw (ψ): Z축 회전 (방향 전환)
```

```
          [Roll - 좌우 기울기]          [Pitch - 전후 기울기]

            ↺ Roll (+)                    ↺ Pitch (+)
               ─────                         ─────
              /     \                       /     \
             │   ●   │                     │   ●   │
              \_____/                       \_____/
            ↙       ↘                        ↑↓
         Left      Right                  Front/Back
```

### 1.2 Height 제어 원리

Height(높이) 제어는 4개 다리의 Y좌표를 동시에 조정하여 몸체 높이를 변경합니다.

```python
import numpy as np
import math

class BodyController:
    """
    Body 자세 제어 클래스
    """
    def __init__(self):
        # 기본 높이 (mm)
        self.default_height = -100
        
        # 다리 기본 위치 (x, y, z)
        # FL, FR, RL, RR 순서
        self.default_foot_positions = np.array([
            [120, -100, 87],    # Front Left
            [120, -100, -87],   # Front Right
            [-50, -100, 77],    # Rear Left
            [-50, -100, -77]    # Rear Right
        ])
        
    def set_height(self, height_offset):
        """
        몸체 높이 조정
        
        Parameters:
        -----------
        height_offset : float
            높이 오프셋 (양수: 위로, 음수: 아래로)
        
        Returns:
        --------
        new_positions : numpy array
            조정된 발끝 위치
        """
        new_positions = self.default_foot_positions.copy()
        
        # 모든 다리의 Y좌표에 height_offset 적용
        # 몸체를 올리려면 -> 발끝을 상대적으로 아래로 (-offset)
        new_positions[:, 1] -= height_offset
        
        return new_positions
```

### 1.3 Pitch 제어 원리

Pitch 제어는 앞다리와 뒷다리의 높이를 다르게 조정하여 전후 기울기를 만듭니다.

```python
def set_pitch(self, pitch_angle_deg):
    """
    Pitch (전후 기울기) 조정
    
    Parameters:
    -----------
    pitch_angle_deg : float
        Pitch 각도 (도 단위, 양수: 앞으로 기울임)
    """
    pitch_rad = math.radians(pitch_angle_deg)
    new_positions = self.default_foot_positions.copy()
    
    # 회전 중심에서 각 다리까지의 X 거리
    # 앞다리: +120mm, 뒷다리: -50mm
    for i, pos in enumerate(new_positions):
        x_dist = pos[0]  # 회전 중심으로부터 X 거리
        
        # Y 오프셋 계산: Δy = x * tan(pitch)
        y_offset = x_dist * math.tan(pitch_rad)
        new_positions[i, 1] -= y_offset
    
    return new_positions
```

```
[Pitch 동작 원리]

    Pitch = 0° (수평)           Pitch = +10° (앞으로 기울임)
    
         FL ●───────● FR             FL ●
            │       │                   ╲
            │ Body  │                    ╲ Body
            │       │                     ╲
         RL ●───────● RR                   ●───────● FR
                                        RL         RR

    → 앞다리(FL, FR)는 높이고, 뒷다리(RL, RR)는 낮춤
```

### 1.4 Roll 제어 원리

Roll 제어는 왼쪽 다리와 오른쪽 다리의 높이를 다르게 조정합니다.

```python
def set_roll(self, roll_angle_deg):
    """
    Roll (좌우 기울기) 조정
    
    Parameters:
    -----------
    roll_angle_deg : float
        Roll 각도 (도 단위, 양수: 왼쪽으로 기울임)
    """
    roll_rad = math.radians(roll_angle_deg)
    new_positions = self.default_foot_positions.copy()
    
    # 회전 중심에서 각 다리까지의 Z 거리
    for i, pos in enumerate(new_positions):
        z_dist = pos[2]  # 회전 중심으로부터 Z 거리
        
        # Y 오프셋 계산: Δy = z * tan(roll)
        y_offset = z_dist * math.tan(roll_rad)
        new_positions[i, 1] -= y_offset
    
    return new_positions
```

### 1.5 통합 자세 제어

Height, Pitch, Roll을 동시에 적용하는 함수입니다.

```python
def set_body_pose(self, height_offset=0, pitch_deg=0, roll_deg=0, yaw_deg=0):
    """
    Body 자세 통합 제어
    
    Parameters:
    -----------
    height_offset : float
        높이 오프셋 (mm)
    pitch_deg : float
        Pitch 각도 (도)
    roll_deg : float
        Roll 각도 (도)
    yaw_deg : float
        Yaw 각도 (도)
    
    Returns:
    --------
    new_positions : numpy array
        조정된 발끝 위치 [FL, FR, RL, RR]
    """
    pitch_rad = math.radians(pitch_deg)
    roll_rad = math.radians(roll_deg)
    yaw_rad = math.radians(yaw_deg)
    
    new_positions = self.default_foot_positions.copy().astype(float)
    
    for i, pos in enumerate(new_positions):
        x_dist = pos[0]
        z_dist = pos[2]
        
        # 1. Height 적용
        y_offset = height_offset
        
        # 2. Pitch 적용 (X축 기준 회전)
        y_offset += x_dist * math.tan(pitch_rad)
        
        # 3. Roll 적용 (Z축 기준 회전)
        y_offset += z_dist * math.tan(roll_rad)
        
        new_positions[i, 1] -= y_offset
        
        # 4. Yaw 적용 (Y축 기준 회전) - 발끝 X, Z 좌표 변경
        if yaw_deg != 0:
            new_x = pos[0] * math.cos(yaw_rad) - pos[2] * math.sin(yaw_rad)
            new_z = pos[0] * math.sin(yaw_rad) + pos[2] * math.cos(yaw_rad)
            new_positions[i, 0] = new_x
            new_positions[i, 2] = new_z
    
    return new_positions
```

### 1.6 PyBullet에서 Body 자세 제어 테스트

```python
import pybullet as p
import pybullet_data
import time
import numpy as np
import math
import sys
sys.path.append("..")

from Kinematics.kinematics import Kinematic

class BodyPoseTest:
    """
    PyBullet에서 Body 자세 제어 테스트
    """
    def __init__(self):
        # PyBullet 초기화
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # 지면 및 로봇 로드
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("../urdf/spotmicroai.urdf", [0, 0, 0.3])
        
        # 기구학
        self.kinematics = Kinematic()
        
        # GUI 슬라이더 추가
        self.height_slider = p.addUserDebugParameter("Height", -50, 50, 0)
        self.pitch_slider = p.addUserDebugParameter("Pitch", -30, 30, 0)
        self.roll_slider = p.addUserDebugParameter("Roll", -30, 30, 0)
        
        # Body Controller
        self.body_ctrl = BodyController()
        
    def run(self):
        """
        메인 루프
        """
        while True:
            # 슬라이더 값 읽기
            height = p.readUserDebugParameter(self.height_slider)
            pitch = p.readUserDebugParameter(self.pitch_slider)
            roll = p.readUserDebugParameter(self.roll_slider)
            
            # Body 자세 계산
            foot_positions = self.body_ctrl.set_body_pose(
                height_offset=height,
                pitch_deg=pitch,
                roll_deg=roll
            )
            
            # 각 다리에 IK 적용
            self.apply_ik(foot_positions)
            
            p.stepSimulation()
            time.sleep(1./240.)
    
    def apply_ik(self, foot_positions):
        """
        발끝 위치로 IK 계산 후 관절에 적용
        """
        leg_names = ['FL', 'FR', 'RL', 'RR']
        joint_indices = {
            'FL': [0, 1, 2],
            'FR': [3, 4, 5],
            'RL': [6, 7, 8],
            'RR': [9, 10, 11]
        }
        
        for i, (leg_name, pos) in enumerate(zip(leg_names, foot_positions)):
            try:
                # IK 계산
                angles = self.kinematics.legIK([pos[0], pos[1], pos[2], 1])
                
                # 관절 제어
                for joint_id, angle in zip(joint_indices[leg_name], angles):
                    p.setJointMotorControl2(
                        self.robot,
                        joint_id,
                        p.POSITION_CONTROL,
                        targetPosition=angle,
                        force=15
                    )
            except Exception as e:
                print(f"IK error for {leg_name}: {e}")


# 실행
if __name__ == "__main__":
    test = BodyPoseTest()
    test.run()
```

---

## 2. CoM(Center of Mass) 추적

### 2.1 CoM 개념

**CoM (Center of Mass)**: 로봇 전체 질량의 중심점입니다. 로봇이 안정적으로 서있거나 걸으려면 CoM이 **지지 다각형(Support Polygon)** 내에 있어야 합니다.

```
[Support Polygon과 CoM]

     FL ●─────────────● FR
        │╲           ╱│
        │  ╲  CoM  ╱  │      ← CoM이 Support Polygon 내부에 있으면 안정
        │    ╲ ● ╱    │
        │      ╳      │
        │    ╱   ╲    │
        │  ╱       ╲  │
     RL ●───────────● RR

     ■ Support Polygon: 지면에 닿은 발끝을 연결한 다각형
     ● CoM: 질량 중심점
```

### 2.2 Trot Gait에서의 CoM

Trot Gait에서는 대각선 다리 쌍이 번갈아 움직이므로, Support Polygon이 삼각형으로 줄어듭니다.

```
[Trot Gait 시 Support Polygon 변화]

Phase 1: FL + RR이 Swing              Phase 2: FR + RL이 Swing
(FR + RL이 지면에 닿음)                (FL + RR이 지면에 닿음)

        ○─────────────● FR                  FL ●─────────────○
        │╲           ╱ │                    │ ╲           ╱│
        │  ╲       ╱   │                    │   ╲  CoM  ╱  │
        │    ╲   ╱     │                    │     ╲ ● ╱    │
        │      ╲       │                    │       ╳      │
        │     ╱  CoM   │                    │      ╱ ╲     │
        │   ╱   ●      │                    │    ╱     ╲   │
     RL ●─────────────○                     ○─────────────● RR

     ● = 지면 접촉 (Stance)
     ○ = 공중 (Swing)
```

### 2.3 CoM 계산

```python
class CoMCalculator:
    """
    Center of Mass 계산 클래스
    """
    def __init__(self):
        # 각 부분의 질량 (예시 값, 실제 로봇에 맞게 조정 필요)
        self.body_mass = 1.5      # kg, 몸체
        self.shoulder_mass = 0.1  # kg, 어깨
        self.upper_leg_mass = 0.1 # kg, 상단 다리
        self.lower_leg_mass = 0.05# kg, 하단 다리
        
        # 총 질량
        self.total_mass = (self.body_mass + 
                          4 * (self.shoulder_mass + 
                               self.upper_leg_mass + 
                               self.lower_leg_mass))
        
    def calculate_com(self, body_pos, body_rot, foot_positions):
        """
        전체 로봇의 CoM 계산
        
        Parameters:
        -----------
        body_pos : tuple
            몸체 중심 위치 (x, y, z)
        body_rot : tuple
            몸체 회전 (roll, pitch, yaw)
        foot_positions : numpy array
            4개 다리의 발끝 위치
        
        Returns:
        --------
        com : numpy array
            Center of Mass 위치 [x, y, z]
        """
        # 간단화된 계산: 몸체 질량이 대부분이므로 몸체 중심을 CoM으로 근사
        # 더 정확한 계산을 위해서는 각 링크의 질량과 위치를 고려해야 함
        
        com_x = body_pos[0]
        com_y = body_pos[1]
        com_z = body_pos[2]
        
        # 다리 위치에 따른 CoM 보정 (간단화)
        for foot_pos in foot_positions:
            leg_mass_ratio = (self.shoulder_mass + 
                             self.upper_leg_mass + 
                             self.lower_leg_mass) / self.total_mass
            
            com_x += foot_pos[0] * leg_mass_ratio / 4
            com_z += foot_pos[2] * leg_mass_ratio / 4
        
        return np.array([com_x, com_y, com_z])
    
    def get_support_polygon(self, foot_positions, stance_mask):
        """
        지지 다각형 계산
        
        Parameters:
        -----------
        foot_positions : numpy array
            4개 다리의 발끝 위치
        stance_mask : list
            각 다리의 Stance 여부 [FL, FR, RL, RR]
            True = Stance (지면 접촉)
            False = Swing (공중)
        
        Returns:
        --------
        polygon : list
            지지 다각형의 꼭짓점 좌표
        """
        polygon = []
        for i, (pos, is_stance) in enumerate(zip(foot_positions, stance_mask)):
            if is_stance:
                polygon.append([pos[0], pos[2]])  # X, Z 좌표 (상단 뷰)
        
        return polygon
    
    def is_stable(self, com, support_polygon):
        """
        CoM이 지지 다각형 내에 있는지 확인
        
        Parameters:
        -----------
        com : numpy array
            CoM 위치
        support_polygon : list
            지지 다각형 꼭짓점
        
        Returns:
        --------
        stable : bool
            안정성 여부
        """
        from matplotlib.path import Path
        
        if len(support_polygon) < 3:
            return False  # 최소 3점 필요
        
        polygon_path = Path(support_polygon)
        com_2d = [com[0], com[2]]  # X, Z 좌표
        
        return polygon_path.contains_point(com_2d)
```

### 2.4 CoM 추적 시각화

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation

def visualize_com_tracking():
    """
    CoM 추적 시각화
    """
    com_calc = CoMCalculator()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    ax.set_xlabel('X Position (mm)')
    ax.set_ylabel('Z Position (mm)')
    ax.set_title('CoM Tracking - Top View')
    ax.set_aspect('equal')
    ax.grid(True)
    
    # 다리 위치 (상단 뷰: X, Z)
    foot_positions = np.array([
        [120, -100, 87],    # FL
        [120, -100, -87],   # FR
        [-50, -100, 77],    # RL
        [-50, -100, -77]    # RR
    ])
    
    colors = ['blue', 'red', 'green', 'orange']
    leg_names = ['FL', 'FR', 'RL', 'RR']
    
    def update(frame):
        ax.clear()
        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('Z Position (mm)')
        ax.grid(True)
        
        # Trot Gait Phase 시뮬레이션
        t = frame / 30.0
        phase = (t * 1000 / 2000) % 1.0
        
        # FL+RR 또는 FR+RL이 Stance
        if phase < 0.5:
            stance_mask = [False, True, True, False]  # FR+RL Stance
            title_phase = "Phase 1: FR+RL Stance"
        else:
            stance_mask = [True, False, False, True]  # FL+RR Stance
            title_phase = "Phase 2: FL+RR Stance"
        
        ax.set_title(f'CoM Tracking - {title_phase}')
        
        # 발끝 위치 그리기
        for i, (pos, color, name) in enumerate(zip(foot_positions, colors, leg_names)):
            marker = 'o' if stance_mask[i] else 'x'
            ax.scatter(pos[0], pos[2], c=color, s=200, marker=marker, 
                      label=f'{name} ({"Stance" if stance_mask[i] else "Swing"})')
        
        # Support Polygon 그리기
        support_poly = com_calc.get_support_polygon(foot_positions, stance_mask)
        if len(support_poly) >= 3:
            poly = Polygon(support_poly, fill=True, alpha=0.3, color='cyan')
            ax.add_patch(poly)
        
        # CoM 그리기
        com = com_calc.calculate_com((0, 100, 0), (0, 0, 0), foot_positions)
        ax.scatter(com[0], com[2], c='black', s=300, marker='*', 
                  label=f'CoM ({com[0]:.1f}, {com[2]:.1f})')
        
        ax.legend(loc='upper right')
        
    anim = FuncAnimation(fig, update, frames=120, interval=50)
    plt.tight_layout()
    plt.show()

# 실행
visualize_com_tracking()
```

---

## 3. 전진 보행 구현

### 3.1 전진 보행 원리

전진 보행은 `stepLength (Sl)` 파라미터를 사용하여 발끝이 전후로 이동하게 합니다.

```
[전진 보행 발끝 궤적]

                    ┌─────── Swing Phase ───────┐
                   ╱                             ╲
                  ╱    (발을 들고 앞으로 이동)     ╲
                 ╱                                 ╲
    ●───────────●                                   ●───────────●
    │                                                           │
    │    Sl/2    │←──── Step Length (Sl) ────→│    Sl/2    │
    │                                                           │
    ●───────────●─────────────────────────────────●───────────●
                        (Stance: 뒤로 밀기)
                 ←─────── Stance Phase ───────→
```

### 3.2 TrottingGait에서 stepLength 설정

```python
import sys
sys.path.append("..")

from Kinematics.kinematicMotion import TrottingGait
import numpy as np
import time

class ForwardWalking:
    """
    전진 보행 구현 클래스
    """
    def __init__(self):
        self.trotting = TrottingGait()
        
        # 전진 보행 설정
        self.step_length = 60.0  # 보폭 (mm)
        self.step_width = 0.0    # 좌우 이동 없음
        self.step_alpha = 0.0    # 회전 없음
        
        self.kb_offset = {
            'IDstepLength': self.step_length,
            'IDstepWidth': self.step_width,
            'IDstepAlpha': self.step_alpha
        }
        
        self.start_time = time.time()
        
    def get_foot_positions(self):
        """
        현재 시간의 발끝 위치 반환
        """
        t = time.time() - self.start_time
        return self.trotting.positions(t, self.kb_offset)
    
    def set_step_length(self, length):
        """
        보폭 설정
        """
        self.step_length = length
        self.kb_offset['IDstepLength'] = length
    
    def stop(self):
        """
        정지
        """
        self.kb_offset = {
            'IDstepLength': 0.0,
            'IDstepWidth': 0.0,
            'IDstepAlpha': 0.0
        }
```

### 3.3 PyBullet 전진 보행 시뮬레이션

```python
import pybullet as p
import pybullet_data
import time
import numpy as np
import sys
sys.path.append("..")

from Kinematics.kinematicMotion import TrottingGait
from Kinematics.kinematics import Kinematic

class ForwardWalkingSimulation:
    """
    PyBullet 전진 보행 시뮬레이션
    """
    def __init__(self):
        # PyBullet 초기화
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # 지면 및 로봇 로드
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("../urdf/spotmicroai.urdf", [0, 0, 0.3])
        
        # 카메라 설정
        p.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0]
        )
        
        # 보행 및 기구학
        self.trotting = TrottingGait()
        self.kinematics = Kinematic()
        
        # GUI 슬라이더
        self.step_length_slider = p.addUserDebugParameter("Step Length", 0, 100, 60)
        self.step_height_slider = p.addUserDebugParameter("Step Height", 20, 80, 40)
        
        # 관절 인덱스
        self.joint_indices = {
            'FL': [0, 1, 2],
            'FR': [3, 4, 5],
            'RL': [6, 7, 8],
            'RR': [9, 10, 11]
        }
        
        self.start_time = time.time()
        
    def run(self):
        """
        메인 시뮬레이션 루프
        """
        print("="*50)
        print("전진 보행 시뮬레이션 시작")
        print("Step Length 슬라이더로 보폭 조절")
        print("Ctrl+C로 종료")
        print("="*50)
        
        try:
            while True:
                t = time.time() - self.start_time
                
                # 슬라이더 값 읽기
                step_length = p.readUserDebugParameter(self.step_length_slider)
                step_height = p.readUserDebugParameter(self.step_height_slider)
                
                self.trotting.Sh = step_height
                
                kb_offset = {
                    'IDstepLength': step_length,
                    'IDstepWidth': 0.0,
                    'IDstepAlpha': 0.0
                }
                
                # 발끝 위치 계산
                foot_positions = self.trotting.positions(t, kb_offset)
                
                # IK 적용
                self.apply_ik(foot_positions)
                
                # 카메라 추적
                robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
                p.resetDebugVisualizerCamera(
                    cameraDistance=1.0,
                    cameraYaw=45,
                    cameraPitch=-30,
                    cameraTargetPosition=robot_pos
                )
                
                p.stepSimulation()
                time.sleep(1./240.)
                
        except KeyboardInterrupt:
            print("\n시뮬레이션 종료")
        finally:
            p.disconnect()
            
    def apply_ik(self, foot_positions):
        """
        IK 적용
        """
        leg_names = ['FL', 'FR', 'RL', 'RR']
        
        for i, (leg_name, pos) in enumerate(zip(leg_names, foot_positions)):
            try:
                angles = self.kinematics.legIK([pos[0], pos[1], pos[2], 1])
                
                for joint_id, angle in zip(self.joint_indices[leg_name], angles):
                    p.setJointMotorControl2(
                        self.robot,
                        joint_id,
                        p.POSITION_CONTROL,
                        targetPosition=angle,
                        force=15
                    )
            except Exception as e:
                pass


# 실행
if __name__ == "__main__":
    sim = ForwardWalkingSimulation()
    sim.run()
```

### 3.4 전진 보행 파라미터 분석

| 파라미터 | 기본값 | 설명 | 영향 |
|----------|--------|------|------|
| `stepLength (Sl)` | 60mm | 보폭 | ↑ 빠른 전진, 불안정해질 수 있음 |
| `stepHeight (Sh)` | 40mm | 발 들어올리는 높이 | ↑ 장애물 넘기 쉬움, 에너지 소모 ↑ |
| `t1` | 1200ms | Stance 시간 | ↑ 느린 보행, 안정성 ↑ |
| `t3` | 200ms | Swing 시간 | ↓ 빠른 발 이동 |

---

## 4. 좌우 회전 보행 구현

### 4.1 회전 원리

회전 보행은 `stepAlpha (Sa)` 파라미터를 사용하여 각 다리가 회전하면서 이동하게 합니다.

```
[제자리 회전]

        FL ●───────● FR           FL ●───────● FR
            │     │                   ╲     ╱
            │  ↻  │   ───────→         ╲ ↻ ╱
            │     │    회전             ╱   ╲
        RL ●───────● RR           RL ●───────● RR
        
        stepAlpha > 0: 시계 방향 회전 (오른쪽)
        stepAlpha < 0: 반시계 방향 회전 (왼쪽)
```

### 4.2 회전 보행 구현

```python
class TurningWalk:
    """
    좌우 회전 보행 구현
    """
    def __init__(self):
        self.trotting = TrottingGait()
        self.start_time = time.time()
        
    def turn_left(self, angle=15):
        """
        왼쪽 회전 (반시계 방향)
        """
        return {
            'IDstepLength': 0.0,   # 전진 없음
            'IDstepWidth': 0.0,
            'IDstepAlpha': -angle  # 음수: 왼쪽 회전
        }
    
    def turn_right(self, angle=15):
        """
        오른쪽 회전 (시계 방향)
        """
        return {
            'IDstepLength': 0.0,
            'IDstepWidth': 0.0,
            'IDstepAlpha': angle   # 양수: 오른쪽 회전
        }
    
    def forward_left(self, step_length=60, angle=10):
        """
        전진하면서 왼쪽으로 회전 (커브)
        """
        return {
            'IDstepLength': step_length,
            'IDstepWidth': 0.0,
            'IDstepAlpha': -angle
        }
    
    def forward_right(self, step_length=60, angle=10):
        """
        전진하면서 오른쪽으로 회전 (커브)
        """
        return {
            'IDstepLength': step_length,
            'IDstepWidth': 0.0,
            'IDstepAlpha': angle
        }
    
    def get_positions(self, command):
        """
        명령에 따른 발끝 위치 반환
        """
        t = time.time() - self.start_time
        return self.trotting.positions(t, command)
```

### 4.3 키보드 제어 통합

```python
import pybullet as p
import pybullet_data
import time
import sys
sys.path.append("..")

from Kinematics.kinematicMotion import TrottingGait
from Kinematics.kinematics import Kinematic

class KeyboardControlledWalking:
    """
    키보드로 제어하는 보행 시뮬레이션
    """
    def __init__(self):
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("../urdf/spotmicroai.urdf", [0, 0, 0.3])
        
        self.trotting = TrottingGait()
        self.kinematics = Kinematic()
        
        self.joint_indices = {
            'FL': [0, 1, 2],
            'FR': [3, 4, 5],
            'RL': [6, 7, 8],
            'RR': [9, 10, 11]
        }
        
        # 현재 명령 상태
        self.step_length = 0.0
        self.step_alpha = 0.0
        
        self.start_time = time.time()
        
    def check_keys(self):
        """
        키보드 입력 확인
        """
        keys = p.getKeyboardEvents()
        
        # 기본값 (정지)
        self.step_length = 0.0
        self.step_alpha = 0.0
        
        # W: 전진
        if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
            self.step_length = 60.0
        
        # S: 후진
        if ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN:
            self.step_length = -60.0
        
        # A: 왼쪽 회전
        if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
            self.step_alpha = -15.0
        
        # D: 오른쪽 회전
        if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
            self.step_alpha = 15.0
        
        # Q: 전진 + 왼쪽 회전
        if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN:
            self.step_length = 60.0
            self.step_alpha = -10.0
        
        # E: 전진 + 오른쪽 회전
        if ord('e') in keys and keys[ord('e')] & p.KEY_IS_DOWN:
            self.step_length = 60.0
            self.step_alpha = 10.0
            
    def run(self):
        """
        메인 루프
        """
        print("="*50)
        print("키보드 제어 보행 시뮬레이션")
        print("-"*50)
        print("W: 전진 | S: 후진")
        print("A: 왼쪽 회전 | D: 오른쪽 회전")
        print("Q: 전진+왼쪽 | E: 전진+오른쪽")
        print("Ctrl+C로 종료")
        print("="*50)
        
        try:
            while True:
                t = time.time() - self.start_time
                
                # 키보드 입력 확인
                self.check_keys()
                
                kb_offset = {
                    'IDstepLength': self.step_length,
                    'IDstepWidth': 0.0,
                    'IDstepAlpha': self.step_alpha
                }
                
                # 발끝 위치 계산
                foot_positions = self.trotting.positions(t, kb_offset)
                
                # IK 적용
                self.apply_ik(foot_positions)
                
                # 카메라 추적
                robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
                p.resetDebugVisualizerCamera(
                    cameraDistance=1.0,
                    cameraYaw=45,
                    cameraPitch=-30,
                    cameraTargetPosition=robot_pos
                )
                
                p.stepSimulation()
                time.sleep(1./240.)
                
        except KeyboardInterrupt:
            print("\n시뮬레이션 종료")
        finally:
            p.disconnect()
            
    def apply_ik(self, foot_positions):
        """
        IK 적용
        """
        leg_names = ['FL', 'FR', 'RL', 'RR']
        
        for i, (leg_name, pos) in enumerate(zip(leg_names, foot_positions)):
            try:
                angles = self.kinematics.legIK([pos[0], pos[1], pos[2], 1])
                
                for joint_id, angle in zip(self.joint_indices[leg_name], angles):
                    p.setJointMotorControl2(
                        self.robot,
                        joint_id,
                        p.POSITION_CONTROL,
                        targetPosition=angle,
                        force=15
                    )
            except Exception as e:
                pass


# 실행
if __name__ == "__main__":
    sim = KeyboardControlledWalking()
    sim.run()
```

---

## 5. 주차 과제

### 5.1 과제 내용

**전진 보행 + 좌우 회전 구현 및 파라미터 튜닝 보고서 작성**

#### 필수 구현 항목

1. **전진 보행 구현**
   - stepLength를 사용한 전진 보행
   - PyBullet 시뮬레이션에서 로봇이 실제로 앞으로 이동하는지 확인

2. **좌우 회전 구현**
   - stepAlpha를 사용한 제자리 회전
   - 전진하면서 회전 (커브 주행)

3. **키보드 제어**
   - W/A/S/D 키로 전진/회전 제어
   - Q/E 키로 커브 주행

#### 파라미터 튜닝 보고서

다음 파라미터들을 변경하며 실험하고 결과를 보고서에 정리:

| 실험 항목 | 테스트 값 범위 | 관찰 내용 |
|-----------|---------------|-----------|
| `stepLength (Sl)` | 30, 60, 90, 120 mm | 보행 속도, 안정성 변화 |
| `stepHeight (Sh)` | 20, 40, 60, 80 mm | 발 들어올림, 에너지 효율 |
| `t1` (Stance time) | 600, 900, 1200, 1500 ms | 보행 주기, 안정성 |
| `t3` (Swing time) | 100, 200, 300, 400 ms | 발 이동 속도 |
| `stepAlpha (Sa)` | 5, 10, 15, 20 deg | 회전 각도, 반경 |

#### 보고서 형식

```markdown
# 7주차 과제: 전진 보행 + 좌우 회전 파라미터 튜닝 보고서

## 1. 실험 환경
- OS: 
- Python 버전: 
- PyBullet 버전: 

## 2. 전진 보행 실험

### 2.1 stepLength 변화
| Sl (mm) | 이동 거리 (5초) | 안정성 | 비고 |
|---------|----------------|--------|------|
| 30      |                |        |      |
| 60      |                |        |      |
| 90      |                |        |      |
| 120     |                |        |      |

### 2.2 stepHeight 변화
| Sh (mm) | 최대 높이 | 부드러움 | 비고 |
|---------|----------|---------|------|
| 20      |          |         |      |
| 40      |          |         |      |
| 60      |          |         |      |
| 80      |          |         |      |

## 3. 회전 보행 실험

### 3.1 stepAlpha 변화 (제자리 회전)
| Sa (deg) | 회전 각도 (5초) | 안정성 | 비고 |
|----------|----------------|--------|------|
| 5        |                |        |      |
| 10       |                |        |      |
| 15       |                |        |      |
| 20       |                |        |      |

### 3.2 전진 + 회전 조합
| Sl (mm) | Sa (deg) | 커브 반경 | 비고 |
|---------|----------|----------|------|
| 60      | 5        |          |      |
| 60      | 10       |          |      |
| 60      | 15       |          |      |

## 4. 최적 파라미터 도출
- 안정적인 전진 보행: Sl = ?, Sh = ?, t1 = ?, t3 = ?
- 안정적인 회전: Sa = ?
- 추천 조합: 

## 5. 결론 및 개선점
...
```

### 5.2 제출물

1. **소스 코드**: 전진/회전 보행이 구현된 Python 파일
2. **보고서**: 파라미터 튜닝 결과 보고서 (Markdown 또는 PDF)
3. **시뮬레이션 영상** (선택): 보행 시뮬레이션 녹화 영상

---

## 6. 참고 자료

### 6.1 핵심 파일

| 파일 | 경로 | 설명 |
|------|------|------|
| kinematicMotion.py | `Kinematics/kinematicMotion.py` | TrottingGait 클래스 |
| kinematics.py | `Kinematics/kinematics.py` | Kinematic 클래스 (IK/FK) |
| spotmicroai.py | `Simulation/spotmicroai.py` | Robot 클래스 |

### 6.2 관련 개념

- **IK (Inverse Kinematics)**: 발끝 위치 → 관절 각도 변환
- **FK (Forward Kinematics)**: 관절 각도 → 발끝 위치 변환
- **Gait Cycle**: 한 걸음의 전체 주기
- **Duty Factor**: Stance 시간 / 전체 주기 시간

### 6.3 추가 학습 자료

1. Introduction to Quadruped Locomotion
2. Static vs Dynamic Balance in Legged Robots
3. CoM Control Strategies for Quadruped Robots
