"""
Week 5-4: 원형 궤적 시각화 및 PyBullet 재현
- Matplotlib로 한 발의 원형 궤적 그리기
- PyBullet에서 실제 로봇으로 재현
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from math import cos, sin, sqrt, atan2, acos
import time


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

    def draw_leg(self, ax, theta1, theta2, theta3, color='blue', label='Leg'):
        """
        3D 플롯에 다리 그리기
        """
        points = self.calc_leg_points(theta1, theta2, theta3)

        # 링크 그리기
        ax.plot3D(points[:, 0], points[:, 1], points[:, 2],
                  color=color, linewidth=3, marker='o',
                  markersize=6, label=label)

        return points


# ==================== 원형 궤적 생성기 ====================

class CircularTrajectory:
    def __init__(self, center, radius, height_offset=0):
        """
        원형 궤적 생성

        Parameters:
        -----------
        center : tuple (x, y, z)
            원의 중심점
        radius : float
            원의 반지름 (mm)
        height_offset : float
            수직 방향 오프셋 (mm)
        """
        self.center = np.array(center)
        self.radius = radius
        self.height_offset = height_offset

    def get_position(self, t, period=3.0):
        """
        시간 t에서의 발끝 위치 계산

        Parameters:
        -----------
        t : float
            시간 (초)
        period : float
            한 바퀴 도는 시간 (초)

        Returns:
        --------
        x, y, z : float
            발끝 위치 (mm)
        """
        angle = 2 * np.pi * t / period

        x = self.center[0] + self.radius * np.cos(angle)
        y = self.center[1] + self.radius * np.sin(angle)
        z = self.center[2] + self.height_offset * np.sin(2 * angle)

        return x, y, z

    def plot_trajectory_3d(self, duration=3.0, dt=0.01):
        """
        3D 궤적 시각화
        """
        times = np.arange(0, duration, dt)
        trajectory = np.array([self.get_position(t, duration) for t in times])

        fig = plt.figure(figsize=(14, 6))

        # 3D 뷰
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                   'b-', linewidth=2, label='Circular Trajectory')
        ax1.scatter(*self.center, color='red', s=100, marker='x', label='Center')

        # 시작점과 끝점 표시
        ax1.scatter(*trajectory[0], color='green', s=150, marker='o', label='Start')
        ax1.scatter(*trajectory[-1], color='orange', s=150, marker='s', label='End')

        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        ax1.set_title('3D Circular Trajectory')
        ax1.legend()
        ax1.grid(True)

        # XY 평면 투영
        ax2 = fig.add_subplot(122)
        ax2.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='XY Projection')
        ax2.scatter(self.center[0], self.center[1], color='red', s=100, marker='x', label='Center')
        ax2.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=150, marker='o', label='Start')

        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_title('Top View (XY Plane)')
        ax2.grid(True)
        ax2.axis('equal')
        ax2.legend()

        # 안내 메시지 추가
        fig.suptitle('창을 닫거나 아무 키나 누르면 다음으로 진행합니다', fontsize=10, y=0.02)

        plt.tight_layout()
        plt.show()

    def animate_with_leg(self, duration=3.0, dt=0.05):
        """
        다리 움직임과 함께 애니메이션
        """
        leg_viz = LegVisualizer()
        leg_ik = LegIK()

        times = np.arange(0, duration, dt)
        trajectory = np.array([self.get_position(t, duration) for t in times])

        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        def update(frame):
            ax1.cla()
            ax2.cla()
            ax3.cla()

            # 현재 시간
            t = times[frame]

            # 현재 목표 위치
            target_pos = trajectory[frame]

            try:
                # IK 계산
                theta1, theta2, theta3 = leg_ik.inverse_kinematics(*target_pos)

                # 3D 뷰 - 다리와 궤적
                ax1.plot3D(trajectory[:frame+1, 0], trajectory[:frame+1, 1], trajectory[:frame+1, 2],
                          'b-', linewidth=2, alpha=0.5, label='Trajectory')
                points = leg_viz.draw_leg(ax1, theta1, theta2, theta3, color='green', label='Leg')
                ax1.scatter(*target_pos, color='red', s=100, marker='x', label='Target')

                ax1.set_xlabel('X (mm)')
                ax1.set_ylabel('Y (mm)')
                ax1.set_zlabel('Z (mm)')
                ax1.set_xlim([self.center[0]-self.radius-50, self.center[0]+self.radius+50])
                ax1.set_ylim([self.center[1]-self.radius-50, self.center[1]+self.radius+50])
                ax1.set_zlim([self.center[2]-self.height_offset-50, self.center[2]+self.height_offset+50])
                ax1.legend()
                ax1.set_title(f'3D View - Time: {t:.2f}s')

                # 측면도 (X-Z)
                ax2.plot(trajectory[:frame+1, 0], trajectory[:frame+1, 2], 'b-', linewidth=2)
                ax2.plot(points[:, 0], points[:, 2], 'go-', linewidth=2, markersize=6)
                ax2.scatter(target_pos[0], target_pos[2], color='red', s=100, marker='x')
                ax2.set_xlabel('X (mm)')
                ax2.set_ylabel('Z (mm)')
                ax2.grid(True)
                ax2.set_title('Side View (X-Z)')
                ax2.axis('equal')

                # 평면도 (X-Y)
                ax3.plot(trajectory[:frame+1, 0], trajectory[:frame+1, 1], 'b-', linewidth=2)
                ax3.scatter(target_pos[0], target_pos[1], color='red', s=100, marker='x')
                ax3.scatter(self.center[0], self.center[1], color='orange', s=100, marker='+')
                ax3.set_xlabel('X (mm)')
                ax3.set_ylabel('Y (mm)')
                ax3.grid(True)
                ax3.set_title('Top View (X-Y)')
                ax3.axis('equal')

                # 관절 각도 정보 표시
                info_text = f'Joint Angles:\nθ1 = {np.rad2deg(theta1):6.2f}°\nθ2 = {np.rad2deg(theta2):6.2f}°\nθ3 = {np.rad2deg(theta3):6.2f}°'
                ax3.text(0.02, 0.98, info_text, transform=ax3.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            except ValueError as e:
                print(f"Frame {frame}: {e}")

        anim = FuncAnimation(fig, update, frames=len(times), interval=50, repeat=True)
        plt.tight_layout()
        plt.show()

        return anim


# ==================== PyBullet 시뮬레이션 ====================

def run_pybullet_simulation():
    """
    PyBullet에서 원형 궤적 재현

    주의: 이 함수를 실행하려면 PyBullet과 SpotMicro URDF가 필요합니다.
    """
    try:
        import pybullet as p
        import pybullet_data
        import os
    except ImportError:
        print("PyBullet이 설치되지 않았습니다.")
        print("설치: pip install pybullet")
        return

    # PyBullet 초기화
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # 환경 로드
    plane = p.loadURDF("plane.urdf")

    # SpotMicro 로봇 로드
    # 현재 파일 위치에서 상대 경로로 URDF 찾기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "..", "..", "urdf", "spotmicroai_gen.urdf.xml")

    try:
        if os.path.exists(urdf_path):
            robot = p.loadURDF(urdf_path, [0, 0, 0.3], useFixedBase=False)
            print(f"URDF 로드 성공: {urdf_path}")
        else:
            print(f"URDF 파일을 찾을 수 없습니다: {urdf_path}")
            print("\n간단한 시각화를 위해 Matplotlib 애니메이션을 사용하세요 (옵션 2).")
            p.disconnect()
            return
    except Exception as e:
        print(f"URDF 로드 중 오류 발생: {e}")
        print("\n간단한 시각화를 위해 Matplotlib 애니메이션을 사용하세요 (옵션 2).")
        p.disconnect()
        return

    # 관절 이름 매핑 가져오기
    nJoints = p.getNumJoints(robot)
    jointNameToId = {}
    for i in range(nJoints):
        jointInfo = p.getJointInfo(robot, i)
        jointNameToId[jointInfo[1].decode('UTF-8')] = jointInfo[0]

    print(f"\n로봇 관절 정보:")
    print(f"총 관절 수: {nJoints}")

    # IK 및 궤적 생성기
    leg_ik = LegIK()
    trajectory = CircularTrajectory(
        center=(100, -100, -150),
        radius=40,
        height_offset=30
    )

    # 초기 발 위치 (Home Stance) - 각 다리의 로컬 좌표
    initial_positions = {
        'front_left':  [100, 100, -150],
        'front_right': [100, -100, -150],
        'rear_left':   [-100, 100, -150],
        'rear_right':  [-100, -100, -150]
    }

    # PD 게인
    Kp = 0.015
    Kd = 0.5
    max_force = 15.0

    # 시뮬레이션 루프
    t = 0
    dt = 1/240
    period = 3.0

    print("\nPyBullet 시뮬레이션 시작...")
    print("원형 궤적으로 앞왼쪽 다리를 움직입니다.")
    print("Ctrl+C를 눌러 종료하세요.\n")

    try:
        while True:
            # 원형 궤적 계산 (앞왼쪽 다리만)
            x, y, z = trajectory.get_position(t, period)

            # 현재 프레임의 발 위치
            foot_positions = initial_positions.copy()
            foot_positions['front_left'] = [x, y, z]

            # 각 다리의 IK 계산 및 제어
            for leg_name, target_pos in foot_positions.items():
                try:
                    # IK 계산
                    theta1, theta2, theta3 = leg_ik.inverse_kinematics(*target_pos)

                    # 관절 이름으로 제어
                    parts = ['shoulder', 'leg', 'foot']
                    angles = [theta1, theta2, theta3]

                    for part, angle in zip(parts, angles):
                        joint_name = f"{leg_name}_{part}"
                        if joint_name in jointNameToId:
                            joint_id = jointNameToId[joint_name]
                            p.setJointMotorControl2(
                                robot, joint_id, p.POSITION_CONTROL,
                                targetPosition=angle, force=max_force,
                                positionGain=Kp, velocityGain=Kd
                            )
                except ValueError as e:
                    # 도달 불가능한 위치
                    pass

            p.stepSimulation()
            time.sleep(dt)
            t += dt

            # 진행 상황 출력 (1초마다)
            if int(t * 10) % 10 == 0:
                print(f"Time: {t:.1f}s / {period:.1f}s", end='\r')

            # 한 바퀴 완료 후 종료 (선택사항)
            if t >= period * 3:  # 3바퀴 후 종료
                print(f"\n\n3바퀴 완료! 시뮬레이션을 계속 보려면 Ctrl+C로 종료하세요.")
                # break를 주석 처리하면 계속 반복됩니다

    except KeyboardInterrupt:
        print("\n\n시뮬레이션 종료")
    finally:
        p.disconnect()


# ==================== 메인 실행 ====================

def main():
    """
    메인 함수 - 원형 궤적 시각화
    """
    print("=" * 60)
    print("Week 5-4: 원형 궤적 시각화 및 PyBullet 재현")
    print("=" * 60)
    print()

    # 과제 파라미터
    center = (100, -100, -150)  # 중심점 (mm)
    radius = 40                  # 반지름 (mm)
    height_offset = 30           # 수직 변화 ±30mm
    period = 3.0                 # 주기 (초)

    print(f"궤적 파라미터:")
    print(f"  중심: {center}")
    print(f"  반지름: {radius} mm")
    print(f"  수직 변화: ±{height_offset} mm")
    print(f"  주기: {period} 초")
    print()

    # 궤적 생성기 생성
    trajectory = CircularTrajectory(center, radius, height_offset)

    # 선택 메뉴
    print("선택하세요:")
    print("1. 3D 궤적 플롯 보기")
    print("2. 다리 움직임 애니메이션 보기")
    print("3. PyBullet 시뮬레이션 실행")
    print("4. 모두 실행")
    print()

    choice = input("선택 (1-4): ").strip()

    if choice == '1':
        print("\n3D 궤적 플롯을 생성합니다...")
        trajectory.plot_trajectory_3d(duration=period)

    elif choice == '2':
        print("\n다리 움직임 애니메이션을 생성합니다...")
        print("(창을 닫으면 프로그램이 종료됩니다)")
        trajectory.animate_with_leg(duration=period)

    elif choice == '3':
        print("\nPyBullet 시뮬레이션을 시작합니다...")
        run_pybullet_simulation()

    elif choice == '4':
        print("\n1. 3D 궤적 플롯")
        trajectory.plot_trajectory_3d(duration=period)

        print("\n2. 다리 움직임 애니메이션")
        trajectory.animate_with_leg(duration=period)

        print("\n3. PyBullet 시뮬레이션")
        run_pybullet_simulation()

    else:
        print("잘못된 선택입니다.")

    print("\n프로그램 종료")


if __name__ == "__main__":
    main()
