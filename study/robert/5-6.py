"""
Week 5-6: 다양한 자세 실험
- 앉은 자세 (Sitting)
- 옆으로 기울기 (Tilting)
- 한 발 들기 (Tripod)
- Matplotlib 시각화 및 PyBullet 시뮬레이션
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
        cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
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
        """정기구학으로 각 관절의 3D 위치 계산"""
        l1, l2, l3, l4 = self.l1, self.l2, self.l3, self.l4
        theta23 = theta2 + theta3

        T0 = np.array([0, 0, 0])
        T1 = T0 + np.array([-l1*cos(theta1), l1*sin(theta1), 0])
        T2 = T1 + np.array([-l2*sin(theta1), -l2*cos(theta1), 0])
        T3 = T2 + np.array([
            -l3*sin(theta1)*cos(theta2),
            -l3*cos(theta1)*cos(theta2),
            l3*sin(theta2)
        ])
        T4 = T3 + np.array([
            -l4*sin(theta1)*cos(theta23),
            -l4*cos(theta1)*cos(theta23),
            l4*sin(theta23)
        ])

        return np.array([T0, T1, T2, T3, T4])


# ==================== 4족 로봇 시각화 ====================

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

        corners = np.array([
            [L, W, 0], [L, -W, 0], [-L, -W, 0], [-L, W, 0], [L, W, 0]
        ])

        ax.plot3D(corners[:, 0], corners[:, 1], corners[:, 2],
                 color='black', linewidth=2)

    def draw_robot(self, ax, leg_angles):
        """전체 로봇 그리기"""
        self.draw_body(ax)

        for leg_name, base_pos in self.leg_positions.items():
            theta1, theta2, theta3 = leg_angles[leg_name]
            points = self.leg_viz.calc_leg_points(theta1, theta2, theta3)
            points_global = points + base_pos

            ax.plot3D(points_global[:, 0], points_global[:, 1], points_global[:, 2],
                     color=self.colors[leg_name], linewidth=2,
                     marker='o', markersize=4, label=leg_name)


# ==================== 4족 로봇 제어 ====================

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
            {'front_left': [x, y, z], ...} (몸통 기준 로컬 좌표)

        Returns:
        --------
        leg_angles : dict
            {'front_left': [θ1, θ2, θ3], ...}
        """
        leg_angles = {}

        for leg_name, target_pos in foot_positions.items():
            try:
                theta1, theta2, theta3 = self.leg_ik.inverse_kinematics(*target_pos)
                leg_angles[leg_name] = [theta1, theta2, theta3]
            except ValueError as e:
                print(f"Warning: {leg_name} - {e}")
                # 기본 각도 사용
                leg_angles[leg_name] = [0.0, -0.8, 1.6]

        return leg_angles

    def visualize_stance(self, foot_positions, title="Robot Stance"):
        """주어진 발 위치로 로봇 자세 시각화"""
        leg_angles = self.set_foot_positions(foot_positions)

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
        ax.legend(loc='upper right')
        ax.set_title(title)

        plt.tight_layout()
        plt.show()


# ==================== PyBullet 시뮬레이션 ====================

def run_pybullet_poses():
    """
    PyBullet에서 다양한 자세 시뮬레이션
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

    # 중력 비활성화 (공중에 떠 있도록)
    p.setGravity(0, 0, 0)

    # 환경 로드 (참고용 - 실제로는 로봇이 공중에 떠 있음)
    plane = p.loadURDF("plane.urdf")

    # SpotMicro 로봇 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "..", "..", "urdf", "spotmicroai_gen.urdf.xml")

    try:
        if os.path.exists(urdf_path):
            # useFixedBase=True로 설정하여 몸통을 공중에 고정
            robot = p.loadURDF(urdf_path, [0, 0, 0.3], useFixedBase=True)
            print(f"URDF 로드 성공: {urdf_path}")
            print("로봇이 공중에 고정되어 자세 변화만 시뮬레이션됩니다.")
        else:
            print(f"URDF 파일을 찾을 수 없습니다: {urdf_path}")
            print("\nMatplotlib 시각화를 사용하세요 (메인 메뉴에서 선택).")
            p.disconnect()
            return
    except Exception as e:
        print(f"URDF 로드 중 오류 발생: {e}")
        p.disconnect()
        return

    # 관절 이름 매핑
    nJoints = p.getNumJoints(robot)
    jointNameToId = {}
    for i in range(nJoints):
        jointInfo = p.getJointInfo(robot, i)
        jointNameToId[jointInfo[1].decode('UTF-8')] = jointInfo[0]

    print(f"\n로봇 관절 수: {nJoints}")

    # IK 객체
    leg_ik = LegIK()

    # PD 게인
    Kp = 0.015
    Kd = 0.5
    max_force = 15.0

    # 자세 정의
    poses = {
        '1': {
            'name': 'Home Stance',
            'positions': {
                'front_left':  [50, 100, -150],
                'front_right': [50, -100, -150],
                'rear_left':   [-50, 100, -150],
                'rear_right':  [-50, -100, -150]
            },
            'duration': 3.0
        },
        '2': {
            'name': 'Sitting',
            'positions': {
                'front_left':  [80, 100, -100],
                'front_right': [80, -100, -100],
                'rear_left':   [-50, 100, -200],
                'rear_right':  [-50, -100, -200]
            },
            'duration': 3.0
        },
        '3': {
            'name': 'Tilted (Right)',
            'positions': {
                'front_left':  [50, 100, -120],
                'front_right': [50, -100, -180],
                'rear_left':   [-50, 100, -120],
                'rear_right':  [-50, -100, -180]
            },
            'duration': 3.0
        },
        '4': {
            'name': 'Tripod (Front Left Up)',
            'positions': {
                'front_left':  [50, 100, -80],
                'front_right': [50, -100, -150],
                'rear_left':   [-50, 100, -150],
                'rear_right':  [-50, -100, -150]
            },
            'duration': 3.0
        }
    }

    def apply_pose(foot_positions):
        """관절 각도 계산 및 적용"""
        for leg_name, target_pos in foot_positions.items():
            try:
                theta1, theta2, theta3 = leg_ik.inverse_kinematics(*target_pos)
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
            except ValueError:
                pass

    print("\n" + "=" * 60)
    print("PyBullet 자세 시뮬레이션")
    print("=" * 60)
    print("\n사용 가능한 자세:")
    for key, pose in poses.items():
        print(f"  {key}: {pose['name']}")
    print("\n자세를 순차적으로 전환합니다...")
    print("Ctrl+C를 눌러 종료하세요.\n")

    try:
        pose_order = ['1', '2', '3', '4']

        while True:
            for pose_key in pose_order:
                pose = poses[pose_key]
                print(f"\n현재 자세: {pose['name']}")

                # 자세 적용
                apply_pose(pose['positions'])

                # 해당 자세 유지
                start_time = time.time()
                while time.time() - start_time < pose['duration']:
                    p.stepSimulation()
                    time.sleep(1/240)

                print(f"{pose['name']} 완료 ({pose['duration']}초 유지)")

    except KeyboardInterrupt:
        print("\n\n시뮬레이션 종료")
    finally:
        p.disconnect()


# ==================== 메인 실행 ====================

def main():
    """
    메인 함수
    """
    print("=" * 60)
    print("Week 5-6: 다양한 자세 실험")
    print("=" * 60)
    print()

    controller = QuadrupedController()

    # 자세 정의
    poses = {
        '1': {
            'name': 'Home Stance',
            'positions': {
                'front_left':  [50, 100, -150],
                'front_right': [50, -100, -150],
                'rear_left':   [-50, 100, -150],
                'rear_right':  [-50, -100, -150]
            }
        },
        '2': {
            'name': 'Sitting (앉은 자세)',
            'positions': {
                'front_left':  [80, 100, -100],   # 앞다리는 높게
                'front_right': [80, -100, -100],
                'rear_left':   [-50, 100, -200],  # 뒷다리는 낮게
                'rear_right':  [-50, -100, -200]
            }
        },
        '3': {
            'name': 'Tilted Right (오른쪽으로 기울기)',
            'positions': {
                'front_left':  [50, 100, -120],   # 왼쪽 다리는 짧게
                'front_right': [50, -100, -180],  # 오른쪽 다리는 길게
                'rear_left':   [-50, 100, -120],
                'rear_right':  [-50, -100, -180]
            }
        },
        '4': {
            'name': 'Tilted Left (왼쪽으로 기울기)',
            'positions': {
                'front_left':  [50, 100, -180],   # 왼쪽 다리는 길게
                'front_right': [50, -100, -120],  # 오른쪽 다리는 짧게
                'rear_left':   [-50, 100, -180],
                'rear_right':  [-50, -100, -120]
            }
        },
        '5': {
            'name': 'Tripod - Front Left Up (왼쪽 앞발 들기)',
            'positions': {
                'front_left':  [50, 100, -80],    # 왼쪽 앞발 들어올림
                'front_right': [50, -100, -150],
                'rear_left':   [-50, 100, -150],
                'rear_right':  [-50, -100, -150]
            }
        },
        '6': {
            'name': 'Tripod - Rear Right Up (오른쪽 뒷발 들기)',
            'positions': {
                'front_left':  [50, 100, -150],
                'front_right': [50, -100, -150],
                'rear_left':   [-50, 100, -150],
                'rear_right':  [-50, -100, -80]   # 오른쪽 뒷발 들어올림
            }
        },
        '7': {
            'name': 'Stretching (스트레칭)',
            'positions': {
                'front_left':  [100, 120, -120],  # 앞다리 앞으로
                'front_right': [100, -120, -120],
                'rear_left':   [-100, 120, -180], # 뒷다리 뒤로
                'rear_right':  [-100, -120, -180]
            }
        }
    }

    print("선택하세요:")
    print("1. Matplotlib로 모든 자세 시각화")
    print("2. PyBullet으로 자세 시뮬레이션")
    print("3. 특정 자세만 Matplotlib로 보기")
    print()

    choice = input("선택 (1-3): ").strip()

    if choice == '1':
        print("\n모든 자세를 Matplotlib로 시각화합니다...")
        for pose_key, pose in poses.items():
            print(f"\n{pose_key}. {pose['name']}")
            controller.visualize_stance(pose['positions'], title=pose['name'])

    elif choice == '2':
        print("\nPyBullet 시뮬레이션을 시작합니다...")
        run_pybullet_poses()

    elif choice == '3':
        print("\n사용 가능한 자세:")
        for key, pose in poses.items():
            print(f"  {key}: {pose['name']}")
        print()

        pose_choice = input("자세 선택 (1-7): ").strip()

        if pose_choice in poses:
            pose = poses[pose_choice]
            print(f"\n{pose['name']}을(를) 시각화합니다...")
            controller.visualize_stance(pose['positions'], title=pose['name'])
        else:
            print("잘못된 선택입니다.")

    else:
        print("잘못된 선택입니다.")

    print("\n프로그램 종료")


if __name__ == "__main__":
    main()
