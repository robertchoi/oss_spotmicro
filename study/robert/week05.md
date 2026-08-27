# 5주차: IK 시각화 및 4족 보행 제어

5주차는 4주차에서 학습한 역기구학(IK)을 Matplotlib로 시각화하고, 4개 다리를 독립적으로 제어하는 방법을 배웁니다. 발끝 궤적을 설계하고 이를 PyBullet 시뮬레이션에서 실제로 구현합니다.

## 학습 목표

1. Matplotlib를 활용한 3D 로봇 다리 시각화
2. 역기구학(IK)으로 발끝 위치를 관절 각도로 변환
3. 4개 다리의 독립적인 제어 구현
4. 발끝 궤적 설계 및 시뮬레이션 적용

## 1. Matplotlib로 로봇 다리 시각화

### 1.1 3D 플로팅 기초

Matplotlib의 3D 플로팅 기능을 사용하여 로봇 다리를 시각화합니다.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D 플롯 생성
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 축 설정
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_xlim([-200, 200])
ax.set_ylim([-200, 200])
ax.set_zlim([-200, 100])

plt.show()
```

### 1.2 다리 링크 그리기

Forward Kinematics로 계산된 관절 위치(T0~T4)를 연결하여 다리를 그립니다.

```python
from math import cos, sin, sqrt, atan2, acos

class LegVisualizer:
    def __init__(self):
        # 링크 길이 (mm)
        self.l1 = 50   # Shoulder offset
        self.l2 = 20   # Shoulder length
        self.l3 = 100  # Upper leg
        self.l4 = 115  # Lower leg

    def calc_leg_points(self, theta1, theta2, theta3):
        """
        정기구학으로 각 관절의 3D 위치 계산
        """
        l1, l2, l3, l4 = self.l1, self.l2, self.l3, self.l4
        theta23 = theta2 + theta3

        # T0: 기준점
        T0 = np.array([0, 0, 0])

        # T1: Shoulder 관절
        T1 = T0 + np.array([-l1*cos(theta1), l1*sin(theta1), 0])

        # T2: Leg 관절
        T2 = T1 + np.array([-l2*sin(theta1), -l2*cos(theta1), 0])

        # T3: Foot 관절
        T3 = T2 + np.array([
            -l3*sin(theta1)*cos(theta2),
            -l3*cos(theta1)*cos(theta2),
            l3*sin(theta2)
        ])

        # T4: 발끝
        T4 = T3 + np.array([
            -l4*sin(theta1)*cos(theta23),
            -l4*cos(theta1)*cos(theta23),
            l4*sin(theta23)
        ])

        return np.array([T0, T1, T2, T3, T4])

    def draw_leg(self, ax, theta1, theta2, theta3, color='blue', label='Leg'):
        """
        3D 플롯에 다리 그리기
        """
        points = self.calc_leg_points(theta1, theta2, theta3)

        # 링크 그리기
        ax.plot3D(points[:, 0], points[:, 1], points[:, 2],
                  color=color, linewidth=3, marker='o',
                  markersize=6, label=label)

        # 관절 표시
        for i, point in enumerate(points):
            ax.text(point[0], point[1], point[2],
                   f'T{i}', fontsize=10)

        return points

# 사용 예시
visualizer = LegVisualizer()

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Home position 각도
theta1, theta2, theta3 = 0.0, -0.8, 1.6

# 다리 그리기
visualizer.draw_leg(ax, theta1, theta2, theta3, color='blue', label='Home Position')

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_xlim([-200, 200])
ax.set_ylim([-200, 200])
ax.set_zlim([-200, 100])
ax.legend()

plt.show()
```

### 1.3 4개 다리 동시 시각화

```python
class QuadrupedVisualizer:
    def __init__(self):
        self.leg_viz = LegVisualizer()

        # 몸통 크기
        self.body_length = 200  # mm
        self.body_width = 100   # mm

        # 각 다리의 기준점 위치 (몸통 기준)
        self.leg_positions = {
            'front_left':  np.array([self.body_length/2, self.body_width/2, 0]),
            'front_right': np.array([self.body_length/2, -self.body_width/2, 0]),
            'rear_left':   np.array([-self.body_length/2, self.body_width/2, 0]),
            'rear_right':  np.array([-self.body_length/2, -self.body_width/2, 0])
        }

        self.colors = {
            'front_left': 'blue',
            'front_right': 'red',
            'rear_left': 'green',
            'rear_right': 'orange'
        }

    def draw_body(self, ax):
        """몸통 그리기"""
        L = self.body_length / 2
        W = self.body_width / 2

        # 몸통 4개 모서리
        corners = np.array([
            [L, W, 0], [L, -W, 0], [-L, -W, 0], [-L, W, 0], [L, W, 0]
        ])

        ax.plot3D(corners[:, 0], corners[:, 1], corners[:, 2],
                 color='black', linewidth=2)

    def draw_robot(self, ax, leg_angles):
        """
        전체 로봇 그리기
        leg_angles: 딕셔너리 {'front_left': [θ1, θ2, θ3], ...}
        """
        # 몸통 그리기
        self.draw_body(ax)

        # 각 다리 그리기
        for leg_name, base_pos in self.leg_positions.items():
            theta1, theta2, theta3 = leg_angles[leg_name]

            # 다리 관절 위치 계산
            points = self.leg_viz.calc_leg_points(theta1, theta2, theta3)

            # 몸통 기준점으로 이동
            points_global = points + base_pos

            # 그리기
            ax.plot3D(points_global[:, 0], points_global[:, 1], points_global[:, 2],
                     color=self.colors[leg_name], linewidth=2,
                     marker='o', markersize=4, label=leg_name)

# 사용 예시
quad_viz = QuadrupedVisualizer()

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 모든 다리를 home position으로 설정
leg_angles = {
    'front_left':  [0.0, -0.8, 1.6],
    'front_right': [0.0, -0.8, 1.6],
    'rear_left':   [0.0, -0.8, 1.6],
    'rear_right':  [0.0, -0.8, 1.6]
}

quad_viz.draw_robot(ax, leg_angles)

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_xlim([-200, 200])
ax.set_ylim([-200, 200])
ax.set_zlim([-300, 100])
ax.legend()
ax.set_title('SpotMicro - Home Position')

plt.show()
```

---

## 2. 발끝 위치 → 관절 각도 변환 (IK)

### 2.1 역기구학 함수 구현

4주차에서 유도한 IK 수식을 실제로 구현합니다.

```python
class LegIK:
    def __init__(self, l1=50, l2=20, l3=100, l4=115):
        """
        l1: Shoulder offset (mm)
        l2: Shoulder length (mm)
        l3: Upper leg length (mm)
        l4: Lower leg length (mm)
        """
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4

    def inverse_kinematics(self, x, y, z):
        """
        역기구학: 발끝 위치 (x, y, z) → 관절 각도 (θ1, θ2, θ3)

        Parameters:
        -----------
        x, y, z : float
            목표 발끝 위치 (mm)

        Returns:
        --------
        theta1, theta2, theta3 : float
            관절 각도 (radians)
        """
        l1, l2, l3, l4 = self.l1, self.l2, self.l3, self.l4

        # Step 1: Abduction 각도 (θ1)
        F = sqrt(x**2 + y**2 - l1**2)
        theta1 = -atan2(y, x) - atan2(F, -l1)

        # Step 2: Hip-Knee 평면으로 축소
        G = F - l2
        H = sqrt(G**2 + z**2)

        # 도달 가능성 확인
        if H > (l3 + l4) or H < abs(l3 - l4):
            raise ValueError(f"Target ({x}, {y}, {z}) is unreachable. H={H:.2f}")

        # Step 3: Knee 각도 (θ3) - Cosine Law
        cos_theta3 = (H**2 - l3**2 - l4**2) / (2 * l3 * l4)
        cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)  # 수치 오차 방지
        theta3 = acos(cos_theta3)

        # Step 4: Hip 각도 (θ2)
        alpha = atan2(z, G)
        beta = atan2(l4 * sin(theta3), l3 + l4 * cos(theta3))
        theta2 = alpha - beta

        return theta1, theta2, theta3

    def forward_kinematics(self, theta1, theta2, theta3):
        """
        정기구학: 관절 각도 → 발끝 위치 (검증용)
        """
        l1, l2, l3, l4 = self.l1, self.l2, self.l3, self.l4
        theta23 = theta2 + theta3

        x = -l1*cos(theta1) - l2*sin(theta1) - l3*sin(theta1)*cos(theta2) - l4*sin(theta1)*cos(theta23)
        y = l1*sin(theta1) - l2*cos(theta1) - l3*cos(theta1)*cos(theta2) - l4*cos(theta1)*cos(theta23)
        z = l3*sin(theta2) + l4*sin(theta23)

        return x, y, z

# 검증 예시
leg_ik = LegIK()

# 목표 발끝 위치
target_pos = (100, -100, -150)

# IK 계산
theta1, theta2, theta3 = leg_ik.inverse_kinematics(*target_pos)
print(f"관절 각도: θ1={np.rad2deg(theta1):.2f}°, θ2={np.rad2deg(theta2):.2f}°, θ3={np.rad2deg(theta3):.2f}°")

# FK로 검증
actual_pos = leg_ik.forward_kinematics(theta1, theta2, theta3)
print(f"목표 위치: {target_pos}")
print(f"실제 위치: ({actual_pos[0]:.2f}, {actual_pos[1]:.2f}, {actual_pos[2]:.2f})")
print(f"오차: {np.linalg.norm(np.array(target_pos) - np.array(actual_pos)):.4f} mm")
```

### 2.2 IK 시각화

IK 계산 결과를 Matplotlib로 시각화하여 검증합니다.

```python
def visualize_ik_result(target_pos, leg_ik, leg_viz):
    """
    IK 계산 결과를 시각화
    """
    fig = plt.figure(figsize=(14, 6))

    # 왼쪽: 목표 위치 표시
    ax1 = fig.add_subplot(121, projection='3d')

    # IK 계산
    theta1, theta2, theta3 = leg_ik.inverse_kinematics(*target_pos)

    # 다리 그리기
    points = leg_viz.draw_leg(ax1, theta1, theta2, theta3, color='blue', label='Leg')

    # 목표 위치 표시
    ax1.scatter(*target_pos, color='red', s=100, marker='x', label='Target')

    # 실제 발끝 위치
    actual_pos = points[4]
    ax1.scatter(*actual_pos, color='green', s=100, marker='o', label='Actual Foot')

    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_xlim([-200, 200])
    ax1.set_ylim([-200, 200])
    ax1.set_zlim([-200, 100])
    ax1.legend()
    ax1.set_title('IK Result - 3D View')

    # 오른쪽: 측면도
    ax2 = fig.add_subplot(122)
    ax2.plot(points[:, 0], points[:, 2], 'b-o', linewidth=2, markersize=6, label='Leg')
    ax2.scatter(target_pos[0], target_pos[2], color='red', s=100, marker='x', label='Target')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Z (mm)')
    ax2.grid(True)
    ax2.legend()
    ax2.set_title('Side View (X-Z Plane)')
    ax2.axis('equal')

    plt.tight_layout()
    plt.show()

# 사용 예시
visualize_ik_result((100, -100, -150), leg_ik, LegVisualizer())
```

---

## 3. 4개 다리 독립 제어

### 3.1 4개 다리의 목표 위치 설정

```python
class QuadrupedController:
    def __init__(self):
        self.leg_ik = LegIK()
        self.quad_viz = QuadrupedVisualizer()

    def set_foot_positions(self, foot_positions):
        """
        4개 발의 목표 위치로부터 관절 각도 계산

        Parameters:
        -----------
        foot_positions : dict
            {'front_left': [x, y, z], 'front_right': [x, y, z], ...}
            (몸통 기준 로컬 좌표)

        Returns:
        --------
        leg_angles : dict
            {'front_left': [θ1, θ2, θ3], ...}
        """
        leg_angles = {}

        for leg_name, target_pos in foot_positions.items():
            # 각 다리의 IK 계산
            theta1, theta2, theta3 = self.leg_ik.inverse_kinematics(*target_pos)
            leg_angles[leg_name] = [theta1, theta2, theta3]

        return leg_angles

    def visualize_stance(self, foot_positions):
        """
        주어진 발 위치로 로봇 자세 시각화
        """
        # IK 계산
        leg_angles = self.set_foot_positions(foot_positions)

        # 시각화
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        self.quad_viz.draw_robot(ax, leg_angles)

        # 목표 발 위치 표시
        for leg_name, pos in foot_positions.items():
            base_pos = self.quad_viz.leg_positions[leg_name]
            global_pos = np.array(pos) + base_pos
            ax.scatter(*global_pos, color='red', s=50, marker='x')

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_xlim([-200, 200])
        ax.set_ylim([-200, 200])
        ax.set_zlim([-300, 100])
        ax.legend()

        plt.show()

# 사용 예시 - Home Stance
controller = QuadrupedController()

# 4개 발의 목표 위치 (몸통 기준 로컬 좌표)
home_stance = {
    'front_left':  [50, 100, -150],
    'front_right': [50, -100, -150],
    'rear_left':   [-50, 100, -150],
    'rear_right':  [-50, -100, -150]
}

controller.visualize_stance(home_stance)
```

### 3.2 다양한 자세 실험

```python
# 실험 1: 앉은 자세 (Sitting)
sitting_stance = {
    'front_left':  [80, 100, -100],   # 앞다리는 높게
    'front_right': [80, -100, -100],
    'rear_left':   [-50, 100, -200],  # 뒷다리는 낮게
    'rear_right':  [-50, -100, -200]
}

controller.visualize_stance(sitting_stance)

# 실험 2: 한쪽으로 기울기
tilted_stance = {
    'front_left':  [50, 100, -120],   # 왼쪽 다리는 짧게
    'front_right': [50, -100, -180],  # 오른쪽 다리는 길게
    'rear_left':   [-50, 100, -120],
    'rear_right':  [-50, -100, -180]
}

controller.visualize_stance(tilted_stance)

# 실험 3: 한 발 들기
tripod_stance = {
    'front_left':  [50, 100, -80],    # 왼쪽 앞발 들어올림
    'front_right': [50, -100, -150],
    'rear_left':   [-50, 100, -150],
    'rear_right':  [-50, -100, -150]
}

controller.visualize_stance(tripod_stance)
```

---

## 4. 발끝 궤적 설계

### 4.1 원형 궤적 (Circular Trajectory)

한 발을 원형 궤적으로 움직이는 패턴을 설계합니다.

```python
class TrajectoryGenerator:
    def __init__(self, center, radius, height_offset=0):
        """
        center : tuple (x, y, z)
            원의 중심점
        radius : float
            원의 반지름
        height_offset : float
            수직 방향 오프셋
        """
        self.center = np.array(center)
        self.radius = radius
        self.height_offset = height_offset

    def circular_trajectory(self, t, period=2.0):
        """
        원형 궤적 생성

        Parameters:
        -----------
        t : float
            시간 (초)
        period : float
            한 바퀴 도는 시간 (초)

        Returns:
        --------
        x, y, z : float
            시간 t에서의 발끝 위치
        """
        angle = 2 * np.pi * t / period

        x = self.center[0] + self.radius * np.cos(angle)
        y = self.center[1] + self.radius * np.sin(angle)
        z = self.center[2] + self.height_offset * np.sin(2 * angle)

        return x, y, z

    def plot_trajectory(self, duration=2.0, dt=0.01):
        """
        궤적을 3D로 시각화
        """
        times = np.arange(0, duration, dt)
        trajectory = np.array([self.circular_trajectory(t) for t in times])

        fig = plt.figure(figsize=(12, 5))

        # 3D 뷰
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                  'b-', linewidth=2, label='Trajectory')
        ax1.scatter(*self.center, color='red', s=100, marker='x', label='Center')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        ax1.legend()
        ax1.set_title('3D Trajectory')

        # 2D 투영 (XY 평면)
        ax2 = fig.add_subplot(122)
        ax2.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2)
        ax2.scatter(self.center[0], self.center[1], color='red', s=100, marker='x')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.grid(True)
        ax2.axis('equal')
        ax2.set_title('Top View (XY Plane)')

        plt.tight_layout()
        plt.show()

# 사용 예시
traj_gen = TrajectoryGenerator(center=(100, -100, -150), radius=30, height_offset=20)
traj_gen.plot_trajectory(duration=2.0)
```

### 4.2 타원형 궤적 (Elliptical Trajectory)

보행에 더 적합한 타원형 궤적을 생성합니다.

```python
class EllipticalTrajectory:
    def __init__(self, center, a, b, height=50):
        """
        center : tuple (x, y, z)
            타원의 중심점
        a : float
            장축 반지름 (전후 방향)
        b : float
            단축 반지름 (좌우 방향)
        height : float
            발을 들어올리는 최대 높이
        """
        self.center = np.array(center)
        self.a = a
        self.b = b
        self.height = height

    def generate_step(self, phase):
        """
        한 걸음의 궤적 생성

        Parameters:
        -----------
        phase : float
            0.0 ~ 1.0 사이의 값 (0: 시작, 1: 끝)

        Returns:
        --------
        x, y, z : float
            발끝 위치
        """
        # 0.0 ~ 0.5: Swing phase (발을 들어올림)
        # 0.5 ~ 1.0: Stance phase (발이 지면에 닿음)

        if phase < 0.5:
            # Swing phase - 타원 궤적
            swing_phase = phase * 2  # 0 ~ 1로 정규화
            angle = np.pi * swing_phase

            x = self.center[0] + self.a * np.cos(angle)
            y = self.center[1]
            z = self.center[2] + self.height * np.sin(angle)
        else:
            # Stance phase - 직선 복귀
            stance_phase = (phase - 0.5) * 2  # 0 ~ 1로 정규화

            x = self.center[0] - self.a + 2 * self.a * stance_phase
            y = self.center[1]
            z = self.center[2]

        return x, y, z

    def plot_step(self):
        """
        한 걸음 궤적 시각화
        """
        phases = np.linspace(0, 1, 100)
        trajectory = np.array([self.generate_step(p) for p in phases])

        fig = plt.figure(figsize=(12, 5))

        # 측면도 (X-Z)
        ax1 = fig.add_subplot(121)

        # Swing phase
        swing_idx = 50
        ax1.plot(trajectory[:swing_idx, 0], trajectory[:swing_idx, 2],
                'b-', linewidth=3, label='Swing (발 들림)')

        # Stance phase
        ax1.plot(trajectory[swing_idx:, 0], trajectory[swing_idx:, 2],
                'r-', linewidth=3, label='Stance (발 닿음)')

        ax1.scatter(*self.center[[0, 2]], color='green', s=100, marker='x', label='Center')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Z (mm)')
        ax1.grid(True)
        ax1.legend()
        ax1.set_title('Side View - One Step')
        ax1.axis('equal')

        # 3D 뷰
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot3D(trajectory[:swing_idx, 0], trajectory[:swing_idx, 1], trajectory[:swing_idx, 2],
                  'b-', linewidth=3, label='Swing')
        ax2.plot3D(trajectory[swing_idx:, 0], trajectory[swing_idx:, 1], trajectory[swing_idx:, 2],
                  'r-', linewidth=3, label='Stance')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_zlabel('Z (mm)')
        ax2.legend()
        ax2.set_title('3D View')

        plt.tight_layout()
        plt.show()

# 사용 예시
ellipse_traj = EllipticalTrajectory(center=(100, -100, -150), a=40, b=20, height=60)
ellipse_traj.plot_step()
```

---

## 5. PyBullet에서 궤적 재현

### 5.1 한 발 원형 궤적 구현

Matplotlib에서 설계한 궤적을 PyBullet 시뮬레이션에 적용합니다.

```python
import pybullet as p
import pybullet_data
import time
import numpy as np

class PyBulletLegController:
    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.leg_ik = LegIK()

        # 관절 인덱스 (URDF 구조에 따라 조정 필요)
        self.joint_indices = {
            'front_left': [0, 1, 2],    # shoulder, hip, knee
            'front_right': [3, 4, 5],
            'rear_left': [6, 7, 8],
            'rear_right': [9, 10, 11]
        }

        # PD 게인
        self.kp = 0.015
        self.kd = 0.5
        self.max_force = 15.0

    def set_leg_position(self, leg_name, target_pos):
        """
        한 다리의 발끝 위치를 설정

        Parameters:
        -----------
        leg_name : str
            다리 이름 ('front_left', 'front_right', 'rear_left', 'rear_right')
        target_pos : tuple
            목표 발끝 위치 (x, y, z) - 몸통 기준 로컬 좌표
        """
        # IK 계산
        theta1, theta2, theta3 = self.leg_ik.inverse_kinematics(*target_pos)
        angles = [theta1, theta2, theta3]

        # 관절 제어
        joint_ids = self.joint_indices[leg_name]
        for joint_id, angle in zip(joint_ids, angles):
            p.setJointMotorControl2(
                self.robot_id,
                joint_id,
                p.POSITION_CONTROL,
                targetPosition=angle,
                force=self.max_force,
                positionGain=self.kp,
                velocityGain=self.kd
            )

def circular_trajectory_pybullet():
    """
    PyBullet에서 한 발 원형 궤적 구현
    """
    # PyBullet 초기화
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    plane = p.loadURDF("plane.urdf")
    robot = p.loadURDF("spot_micro.urdf", [0, 0, 0.3])

    # 컨트롤러 생성
    controller = PyBulletLegController(robot)

    # 궤적 생성기
    traj_gen = TrajectoryGenerator(
        center=(100, -100, -150),
        radius=30,
        height_offset=20
    )

    # 시뮬레이션 루프
    t = 0
    dt = 1/240

    try:
        while True:
            # 원형 궤적 계산
            x, y, z = traj_gen.circular_trajectory(t, period=3.0)

            # 앞왼쪽 다리만 움직임
            controller.set_leg_position('front_left', (x, y, z))

            # 나머지 다리는 고정
            for leg_name in ['front_right', 'rear_left', 'rear_right']:
                controller.set_leg_position(leg_name, (50, -100 if 'right' in leg_name else 100, -150))

            p.stepSimulation()
            time.sleep(dt)
            t += dt

    except KeyboardInterrupt:
        print("시뮬레이션 종료")
    finally:
        p.disconnect()

# 실행
circular_trajectory_pybullet()
```

### 5.2 4개 다리 Trot 보행 패턴

대각선 다리를 쌍으로 움직이는 Trot 보행 패턴을 구현합니다.

```python
class TrotGait:
    def __init__(self):
        self.ellipse = EllipticalTrajectory(
            center=(0, 0, -150),
            a=30,    # 보폭
            b=10,
            height=50
        )

    def get_leg_positions(self, t, period=2.0):
        """
        시간 t에서 4개 다리의 목표 위치 계산

        Trot 보행: 대각선 다리가 같이 움직임
        - Pair 1: front_left + rear_right
        - Pair 2: front_right + rear_left

        Parameters:
        -----------
        t : float
            시간 (초)
        period : float
            한 걸음 주기 (초)

        Returns:
        --------
        foot_positions : dict
            {'front_left': (x, y, z), ...}
        """
        # Phase 계산 (0 ~ 1)
        phase = (t % period) / period

        # Pair 1: front_left, rear_right (phase = 0에서 시작)
        phase1 = phase
        x1, y1, z1 = self.ellipse.generate_step(phase1)

        # Pair 2: front_right, rear_left (phase = 0.5에서 시작, 반대)
        phase2 = (phase + 0.5) % 1.0
        x2, y2, z2 = self.ellipse.generate_step(phase2)

        return {
            'front_left':  (x1 + 100, 100, z1),
            'front_right': (x2 + 100, -100, z2),
            'rear_left':   (x2 - 100, 100, z2),
            'rear_right':  (x1 - 100, -100, z1)
        }

def trot_gait_pybullet():
    """
    PyBullet에서 Trot 보행 패턴 구현
    """
    # PyBullet 초기화
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    plane = p.loadURDF("plane.urdf")
    robot = p.loadURDF("spot_micro.urdf", [0, 0, 0.3])

    # 컨트롤러 및 보행 패턴
    controller = PyBulletLegController(robot)
    gait = TrotGait()

    # 시뮬레이션 루프
    t = 0
    dt = 1/240

    try:
        while True:
            # 현재 시간의 발 위치 계산
            foot_positions = gait.get_leg_positions(t, period=2.0)

            # 각 다리 제어
            for leg_name, target_pos in foot_positions.items():
                controller.set_leg_position(leg_name, target_pos)

            p.stepSimulation()
            time.sleep(dt)
            t += dt

            # 10초마다 진행 상황 출력
            if int(t * 10) % 100 == 0:
                print(f"Time: {t:.2f}s")

    except KeyboardInterrupt:
        print("시뮬레이션 종료")
    finally:
        p.disconnect()

# 실행
trot_gait_pybullet()
```

---

## 6. 실습 예제

### 6.1 Matplotlib 애니메이션

궤적을 애니메이션으로 시각화합니다.

```python
from matplotlib.animation import FuncAnimation

def animate_trajectory():
    """
    발끝 궤적을 애니메이션으로 시각화
    """
    traj_gen = TrajectoryGenerator(center=(100, -100, -150), radius=30, height_offset=20)
    leg_viz = LegVisualizer()
    leg_ik = LegIK()

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    # 궤적 계산
    duration = 3.0
    dt = 0.05
    times = np.arange(0, duration, dt)
    trajectory = np.array([traj_gen.circular_trajectory(t, period=duration) for t in times])

    # 궤적 그리기 (전체)
    ax1.plot3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
              'gray', linewidth=1, alpha=0.3, label='Full Trajectory')

    # 초기 설정
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_xlim([50, 150])
    ax1.set_ylim([-150, -50])
    ax1.set_zlim([-200, -100])
    ax1.legend()

    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Z (mm)')
    ax2.grid(True)
    ax2.axis('equal')

    def update(frame):
        ax1.cla()
        ax2.cla()

        # 현재 시간
        t = times[frame]

        # 현재 목표 위치
        target_pos = trajectory[frame]

        # IK 계산
        theta1, theta2, theta3 = leg_ik.inverse_kinematics(*target_pos)

        # 3D 뷰
        ax1.plot3D(trajectory[:frame+1, 0], trajectory[:frame+1, 1], trajectory[:frame+1, 2],
                  'b-', linewidth=2, label='Trajectory')
        leg_viz.draw_leg(ax1, theta1, theta2, theta3, color='green', label='Leg')
        ax1.scatter(*target_pos, color='red', s=100, marker='x', label='Target')

        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        ax1.set_xlim([50, 150])
        ax1.set_ylim([-150, -50])
        ax1.set_zlim([-200, -100])
        ax1.legend()
        ax1.set_title(f'Time: {t:.2f}s')

        # 측면도
        points = leg_viz.calc_leg_points(theta1, theta2, theta3)
        ax2.plot(trajectory[:frame+1, 0], trajectory[:frame+1, 2], 'b-', linewidth=2)
        ax2.plot(points[:, 0], points[:, 2], 'go-', linewidth=2, markersize=6)
        ax2.scatter(target_pos[0], target_pos[2], color='red', s=100, marker='x')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Z (mm)')
        ax2.grid(True)
        ax2.set_title('Side View')

    anim = FuncAnimation(fig, update, frames=len(times), interval=50, repeat=True)
    plt.tight_layout()
    plt.show()

# 실행
animate_trajectory()
```

### 6.2 IK 오차 분석

```python
def analyze_ik_accuracy():
    """
    IK의 정확도를 분석
    """
    leg_ik = LegIK()

    # 테스트할 목표 위치들
    test_positions = []
    for x in np.linspace(50, 150, 10):
        for y in np.linspace(-150, -50, 10):
            for z in np.linspace(-200, -100, 10):
                test_positions.append((x, y, z))

    errors = []
    valid_count = 0

    for target in test_positions:
        try:
            # IK 계산
            theta1, theta2, theta3 = leg_ik.inverse_kinematics(*target)

            # FK로 검증
            actual = leg_ik.forward_kinematics(theta1, theta2, theta3)

            # 오차 계산
            error = np.linalg.norm(np.array(target) - np.array(actual))
            errors.append(error)
            valid_count += 1

        except ValueError:
            # 도달 불가능한 위치
            pass

    # 결과 출력
    print(f"=== IK 정확도 분석 ===")
    print(f"테스트 위치 수: {len(test_positions)}")
    print(f"도달 가능한 위치: {valid_count} ({valid_count/len(test_positions)*100:.1f}%)")
    print(f"평균 오차: {np.mean(errors):.6f} mm")
    print(f"최대 오차: {np.max(errors):.6f} mm")
    print(f"표준 편차: {np.std(errors):.6f} mm")

    # 오차 히스토그램
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor='black')
    plt.xlabel('Error (mm)')
    plt.ylabel('Count')
    plt.title('IK Positioning Error Distribution')
    plt.axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.6f} mm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 실행
analyze_ik_accuracy()
```

---

## 7. 과제

### 과제 1: 원형 궤적 시각화 및 PyBullet 재현

1. Matplotlib로 한 발의 원형 궤적을 그리세요
   - 중심: (100, -100, -150)
   - 반지름: 40mm
   - 수직 변화: ±30mm

2. 위 궤적을 PyBullet에서 재현하세요
   - 한 발만 움직이고 나머지는 고정
   - 3초에 한 바퀴 회전
   - 부드러운 움직임 확인

**소스 코드 참고:**

#### Matplotlib 시각화 예제
- [Kinematics/kinematics.py](../../../Kinematics/kinematics.py) - `animationKinecatics()` 함수 참고
  - `FuncAnimation`을 사용한 애니메이션 구현 예제
  - 3D 플롯에서 로봇 다리 시각화
  - 참고 라인: 192-239

#### PyBullet 시뮬레이션 예제
- [Simulation/pybullet_automatic_gait.py](../../../Simulation/pybullet_automatic_gait.py)
  - PyBullet 환경 설정 및 로봇 로드: 라인 55
  - 시뮬레이션 루프 구조: 라인 73-100
  - 발 위치 제어: `robot.feetPosition(Lp)` - 라인 97
  - GUI 파라미터 슬라이더: 라인 63

```python
# 예제 코드 구조
robot = spotmicroai.Robot(False, True, reset)
IDheight = p.addUserDebugParameter("height", -40, 90, 20)

while True:
    # 시간에 따른 궤적 계산
    x, y, z = circular_trajectory(time.time())

    # 발 위치 업데이트
    Lp[0] = [x, y, z, 1]  # 첫 번째 다리
    robot.feetPosition(Lp)

    robot.step()
```

---

### 과제 2: 타원형 보행 궤적

1. 타원형 궤적을 설계하세요
   - 장축: 60mm (전후 방향)
   - 단축: 20mm (좌우 방향)
   - 최대 높이: 70mm

2. Matplotlib 애니메이션으로 시각화하세요

3. PyBullet에서 4개 다리 Trot 보행으로 구현하세요

**소스 코드 참고:**

#### Trot 보행 구현
- [Kinematics/kinematicMotion.py](../../../Kinematics/kinematicMotion.py) - `TrottingGait` 클래스
  - 클래스 정의: 라인 69-156
  - `calcLeg()` 함수: 라인 75-115
    - 4단계 보행 사이클 구현 (대기, 드래그, 대기, 리프트)
    - 사인 함수를 사용한 부드러운 발 들어올리기
  - `positions()` 함수: 라인 117-155
    - 4개 다리의 좌표 동시 계산
    - 대각선 다리 쌍 동기화 (front_left + rear_right, front_right + rear_left)

```python
# TrottingGait 사용 예제
class TrottingGait:
    def calcLeg(self, t, x, y, z):
        # 4단계 사이클
        if t < Tt1:  # t0: 대기
            return [x, y, z, 1]
        elif t < Tt2:  # t1: 드래그 (전진)
            return [x + Sl * p, y, z, 1]
        elif t < Tt3:  # t2: 대기
            return [x + Sl, y, z, 1]
        else:  # t3: 리프트 (발 들어올리기)
            return [x + Sl * (1 - p), y + Sh * sin(pi * p), z, 1]
```

#### 실제 시뮬레이션 통합
- [Simulation/pybullet_automatic_gait.py](../../../Simulation/pybullet_automatic_gait.py)
  - Trotting 객체 생성: 라인 69
  - 보행 패턴 적용: 라인 95
  - 키보드 입력으로 보행 파라미터 제어: 라인 88-90

```python
# 실제 사용 예제
trotting = TrottingGait()

while True:
    d = time.time() - start_time

    if stepping:
        # Trot 보행 적용
        robot.feetPosition(trotting.positions(d - 3, command_dict))
    else:
        # 정지 자세
        robot.feetPosition(Lp)
```

#### 궤적 파라미터 조정
프로젝트의 기본 파라미터 ([Kinematics/kinematicMotion.py:71-74](../../../Kinematics/kinematicMotion.py#L71-L74)):
```python
Sl = 0      # Step Length (보폭)
Sw = 0      # Step Width (좌우 이동)
Sh = 72     # Step Height (발 들어올림 높이)
Sa = 0      # Step Rotation (회전 각도)
```

과제에서는 이 값들을 다음과 같이 조정:
```python
Sl = 60     # 장축 60mm
Sw = 20     # 단축 20mm
Sh = 70     # 최대 높이 70mm
```

---

### 과제 3: 다양한 자세 실험

다음 자세들을 Matplotlib와 PyBullet 양쪽에서 구현하세요:

1. **앉은 자세**: 뒷다리를 접고 앞다리를 편 자세
2. **옆으로 기울기**: 한쪽 다리를 짧게, 반대쪽을 길게
3. **한 발 들기**: 3개 다리로 균형 잡기

**소스 코드 참고:**

#### 역기구학 (IK) 구현
- [Kinematics/kinematics.py](../../../Kinematics/kinematics.py) - `Kinematic` 클래스
  - `legIK(point)` 함수: 라인 67-87
    - 발끝 위치 (x, y, z) → 관절 각도 (θ1, θ2, θ3) 변환
    - 기하학적 해법 사용

```python
# IK 함수 구조 (kinematics.py:67-87)
def legIK(self, point):
    (x, y, z) = (point[0], point[1], point[2])
    (l1, l2, l3, l4) = (self.l1, self.l2, self.l3, self.l4)

    # Step 1: Abduction 각도
    F = sqrt(x**2 + y**2 - l1**2)
    theta1 = -atan2(y, x) - atan2(F, -l1)

    # Step 2: Hip-Knee 평면 문제
    G = F - l2
    H = sqrt(G**2 + z**2)

    # Step 3: Knee 각도
    D = (H**2 - l3**2 - l4**2) / (2 * l3 * l4)
    theta3 = acos(D)

    # Step 4: Hip 각도
    theta2 = atan2(z, G) - atan2(l4*sin(theta3), l3 + l4*cos(theta3))

    return (theta1, theta2, theta3)
```

#### 정기구학 (FK) 검증
- [Kinematics/kinematics.py:89-100](../../../Kinematics/kinematics.py#L89-L100) - `calcLegPoints()` 함수
  - 관절 각도 → 각 링크의 3D 위치 계산
  - IK 결과 검증용

#### 자세 시각화
- [Kinematics/kinematics.py:163-173](../../../Kinematics/kinematics.py#L163-L173) - `calcIK()` 함수
  - 4개 다리 모두의 IK 한번에 계산
  - 전체 로봇 자세 설정

```python
# 자세 예제 코드
moduleKinematics = kn.Kinematic()

# 앉은 자세
sitting_pose = np.array([
    [80, -100, 100, 1],    # Front Left (높게)
    [80, -100, -100, 1],   # Front Right (높게)
    [-50, -200, 100, 1],   # Rear Left (낮게)
    [-50, -200, -100, 1]   # Rear Right (낮게)
])

# 기울어진 자세
tilted_pose = np.array([
    [50, -120, 100, 1],    # Front Left (짧게)
    [50, -180, -100, 1],   # Front Right (길게)
    [-50, -120, 100, 1],   # Rear Left (짧게)
    [-50, -180, -100, 1]   # Rear Right (길게)
])

# IK 계산
thetas = moduleKinematics.calcIK(sitting_pose, (0,0,0), (0,0,0))

# Matplotlib로 시각화
moduleKinematics.drawRobot(sitting_pose, (0,0,0), (0,0,0))
```

#### PyBullet에서 자세 적용
- [Simulation/spotmicroai.py](../../../Simulation/spotmicroai.py) - `Robot` 클래스
  - `feetPosition(Lp)` 함수: 발 위치 설정
  - `bodyRotation(rot)` 함수: 몸통 회전 설정
  - `step()` 함수: 시뮬레이션 한 스텝 진행

```python
# PyBullet에서 자세 적용 예제
robot = spotmicroai.Robot(False, True, reset)

# 앉은 자세 적용
sitting_Lp = np.array([
    [80, -100, 100, 1],
    [80, -100, -100, 1],
    [-50, -200, 100, 1],
    [-50, -200, -100, 1]
])

robot.feetPosition(sitting_Lp)
robot.step()

# 시각화 업데이트
for _ in range(240):  # 1초 동안 유지
    robot.step()
    time.sleep(1/240)
```

#### 링크 길이 참고
프로젝트에서 사용하는 실제 링크 길이 ([Kinematics/kinematics.py:59-62](../../../Kinematics/kinematics.py#L59-L62)):
```python
l1 = 50   # Shoulder offset
l2 = 20   # Shoulder length
l3 = 100  # Upper leg (대퇴)
l4 = 115  # Lower leg (하퇴)
```

이 값들을 사용하여 도달 가능한 작업 공간(workspace) 계산:
- 최대 도달 거리: l1 + l2 + l3 + l4 = 285mm
- 최소 도달 거리: |l3 - l4| = 15mm

---

## 8. 주요 개념 정리

| 개념 | 설명 | 활용 |
|------|------|------|
| **Matplotlib 3D 플롯** | 3차원 공간에서 로봇 시각화 | 디버깅, 궤적 검증 |
| **IK (Inverse Kinematics)** | 발끝 위치 → 관절 각도 | 보행 제어의 핵심 |
| **원형 궤적** | 일정한 반지름의 원형 경로 | 기초 궤적 실험 |
| **타원형 궤적** | 보행에 적합한 타원 경로 | 실제 보행 구현 |
| **Trot 보행** | 대각선 다리 쌍이 동기화 | 4족 보행의 기본 |
| **독립 제어** | 각 다리를 개별적으로 제어 | 복잡한 동작 구현 |

---

## 9. 다음 주 예고

6주차에는 **CPG (Central Pattern Generator)** 기반 보행 패턴 생성을 학습합니다:

- CPG 이론 및 수식
- Coupled Oscillator 구현
- Trot, Walk, Bound 보행 생성
- 실시간 파라미터 조정

---

## 10. 참고 자료

- Matplotlib 3D Plotting: [공식 문서](https://matplotlib.org/stable/tutorials/toolkits/mplot3d.html)
- 로봇 기구학 기초
- 보행 패턴 이론
- PyBullet IK vs Custom IK 비교
