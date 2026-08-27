import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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