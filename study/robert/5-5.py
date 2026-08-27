"""
Week 5-5: Matplotlib 애니메이션으로 발끝 궤적 시각화
- 원형 궤적을 따라 움직이는 로봇 다리
- 3D 뷰와 측면도 동시 표시
- 실시간 관절 각도 정보
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from math import cos, sin, sqrt, atan2, acos


# ==================== IK 구현 ====================

class LegIK:
    def __init__(self, l1=50, l2=20, l3=100, l4=115):
        """
        SpotMicro 다리 역기구학

        Parameters:
        -----------
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


# ==================== 다리 시각화 ====================

class LegVisualizer:
    def __init__(self, l1=50, l2=20, l3=100, l4=115):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4

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

    def draw_leg(self, ax, theta1, theta2, theta3, color='green', label='Leg'):
        """
        3D 플롯에 다리 그리기
        """
        points = self.calc_leg_points(theta1, theta2, theta3)

        # 링크 그리기
        ax.plot3D(points[:, 0], points[:, 1], points[:, 2],
                  color=color, linewidth=3, marker='o',
                  markersize=6, label=label)

        return points


# ==================== 궤적 생성기 ====================

class TrajectoryGenerator:
    def __init__(self, center, radius, height_offset=0):
        """
        원형 궤적 생성

        Parameters:
        -----------
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


# ==================== 애니메이션 ====================

def animate_trajectory():
    """
    발끝 궤적을 애니메이션으로 시각화
    """
    # 객체 생성
    traj_gen = TrajectoryGenerator(center=(100, -100, -150), radius=30, height_offset=20)
    leg_viz = LegVisualizer()
    leg_ik = LegIK()

    # Figure 생성
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
        """애니메이션 업데이트 함수"""
        ax1.cla()
        ax2.cla()

        # 현재 시간
        t = times[frame]

        # 현재 목표 위치
        target_pos = trajectory[frame]

        # IK 계산
        try:
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
            ax1.set_title(f'3D View - Time: {t:.2f}s')

            # 측면도 (X-Z)
            points = leg_viz.calc_leg_points(theta1, theta2, theta3)
            ax2.plot(trajectory[:frame+1, 0], trajectory[:frame+1, 2], 'b-', linewidth=2)
            ax2.plot(points[:, 0], points[:, 2], 'go-', linewidth=2, markersize=6)
            ax2.scatter(target_pos[0], target_pos[2], color='red', s=100, marker='x')
            ax2.set_xlabel('X (mm)')
            ax2.set_ylabel('Z (mm)')
            ax2.grid(True)
            ax2.set_title('Side View (X-Z Plane)')
            ax2.axis('equal')

            # 관절 각도 정보 표시
            info_text = f'Joint Angles:\nθ1 = {np.rad2deg(theta1):6.2f}°\nθ2 = {np.rad2deg(theta2):6.2f}°\nθ3 = {np.rad2deg(theta3):6.2f}°'
            ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        except ValueError as e:
            print(f"Frame {frame}: {e}")

    # 애니메이션 생성
    anim = FuncAnimation(fig, update, frames=len(times), interval=50, repeat=True)

    # 안내 메시지
    fig.suptitle('창을 닫으면 프로그램이 종료됩니다', fontsize=10, y=0.02)

    plt.tight_layout()
    plt.show()

    return anim


# ==================== 메인 실행 ====================

def main():
    """
    메인 함수
    """
    print("=" * 60)
    print("Week 5-5: Matplotlib 애니메이션으로 궤적 시각화")
    print("=" * 60)
    print()
    print("궤적 파라미터:")
    print("  중심: (100, -100, -150) mm")
    print("  반지름: 30 mm")
    print("  수직 변화: ±20 mm")
    print("  주기: 3.0 초")
    print()
    print("애니메이션을 시작합니다...")
    print("(창을 닫으면 프로그램이 종료됩니다)")
    print()

    # 애니메이션 실행
    animate_trajectory()

    print("\n프로그램 종료")


if __name__ == "__main__":
    main()
