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