"""
7-4.py: Keyboard Controlled Walking Simulation
Week 07 - 전진 보행 + 좌우 회전 통합 키보드 제어

이 시뮬레이션은 키보드로 로봇의 전진, 후진, 좌우 회전을 제어합니다.
TrottingGait를 사용하여 Trot 보행 패턴을 구현합니다.

조작 방법:
- W: 전진
- S: 후진
- A: 왼쪽 회전
- D: 오른쪽 회전
- Q: 전진 + 왼쪽 회전 (커브)
- E: 전진 + 오른쪽 회전 (커브)
- R: 로봇 위치 리셋
- ESC: 종료
"""

import sys
import os

# 상위 디렉토리를 경로에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..", "..")
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "Kinematics"))
sys.path.insert(0, os.path.join(project_root, "Simulation"))

import pybullet as p
import pybullet_data
import time
import math
import numpy as np

from Kinematics.kinematics import Kinematic
from Kinematics.kinematicMotion import TrottingGait


class KeyboardControlledWalking:
    """
    키보드로 제어하는 보행 시뮬레이션
    전진, 후진, 좌우 회전, 커브 주행을 지원합니다.
    """
    
    def __init__(self):
        # PyBullet 초기화
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # 시뮬레이션 파라미터
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setRealTimeSimulation(True)
        
        # 지면 로드
        self.plane = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane, -1, lateralFriction=1.0)
        
        # SpotMicro 로봇 로드
        urdf_path = os.path.join(project_root, "urdf", "spotmicroai_gen.urdf.xml")
        self.init_position = [0, 0, 0.25]
        self.init_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        self.robot = p.loadURDF(
            urdf_path,
            self.init_position,
            self.init_orientation,
            useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION
        )
        
        # 기구학 및 보행 패턴
        self.kinematics = Kinematic()
        self.trotting = TrottingGait()
        
        # 관절 매핑
        self.joint_name_to_id = self._get_joint_names()
        
        # 다리 방향 보정
        self.dirs = [[-1, 1, 1], [1, 1, 1], [-1, 1, 1], [1, 1, 1]]
        
        # PD 제어 파라미터
        self.kp = 0.045
        self.kd = 0.4
        self.max_force = 25.0
        
        # 보행 파라미터
        self.step_length = 60.0   # 보폭 (mm)
        self.step_alpha = 15.0    # 회전 각도 (deg)
        self.step_height = 40.0   # 발 들어올리는 높이 (mm)
        self.body_height = 20.0   # 몸체 높이 (mm)
        
        # 현재 명령 상태
        self.current_step_length = 0.0
        self.current_step_alpha = 0.0
        self.current_command = "STOP"
        
        # 시작 시간
        self.start_time = time.time()
        
        # 실행 상태
        self.running = True
        
        # GUI 슬라이더 생성
        self._create_sliders()
        
        # 카메라 설정
        p.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.15]
        )
        
        # 조작 안내 텍스트
        self._add_help_text()
        
        print("=" * 60)
        print("Keyboard Controlled Walking Simulation")
        print("-" * 60)
        print("Controls:")
        print("  W: Forward    S: Backward")
        print("  A: Turn Left  D: Turn Right")
        print("  Q: Curve Left E: Curve Right")
        print("  R: Reset      ESC: Exit")
        print("-" * 60)
        print("Use sliders to adjust parameters")
        print("=" * 60)
    
    def _get_joint_names(self):
        """관절 이름과 ID 매핑"""
        joint_name_to_id = {}
        n_joints = p.getNumJoints(self.robot)
        
        for i in range(n_joints):
            joint_info = p.getJointInfo(self.robot, i)
            joint_name = joint_info[1].decode('UTF-8')
            joint_name_to_id[joint_name] = joint_info[0]
            
        return joint_name_to_id
    
    def _create_sliders(self):
        """GUI 슬라이더 생성"""
        self.step_length_slider = p.addUserDebugParameter("Step Length", 20, 100, self.step_length)
        self.step_alpha_slider = p.addUserDebugParameter("Turn Angle", 5, 30, self.step_alpha)
        self.step_height_slider = p.addUserDebugParameter("Step Height", 20, 80, self.step_height)
        self.height_slider = p.addUserDebugParameter("Body Height", -30, 50, self.body_height)
    
    def _add_help_text(self):
        """조작 안내 텍스트 추가"""
        p.addUserDebugText(
            "W:Fwd S:Back A:Left D:Right Q/E:Curve R:Reset",
            [0, 0, 0.5],
            textColorRGB=[1, 1, 0],
            textSize=1.2
        )
    
    def reset_robot(self):
        """로봇 위치 리셋"""
        p.resetBasePositionAndOrientation(
            self.robot,
            self.init_position,
            self.init_orientation
        )
        p.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0])
        self.start_time = time.time()
        self.current_step_length = 0.0
        self.current_step_alpha = 0.0
        self.current_command = "STOP"
        print("[INFO] Robot position reset")
    
    def check_keyboard(self):
        """
        키보드 입력 확인 및 명령 설정
        """
        keys = p.getKeyboardEvents()
        
        # 슬라이더에서 파라미터 읽기
        self.step_length = p.readUserDebugParameter(self.step_length_slider)
        self.step_alpha = p.readUserDebugParameter(self.step_alpha_slider)
        self.step_height = p.readUserDebugParameter(self.step_height_slider)
        self.body_height = p.readUserDebugParameter(self.height_slider)
        
        # 기본값 (정지)
        self.current_step_length = 0.0
        self.current_step_alpha = 0.0
        self.current_command = "STOP"
        
        # W: 전진
        if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
            self.current_step_length = self.step_length
            self.current_command = "FORWARD"
        
        # S: 후진
        if ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN:
            self.current_step_length = -self.step_length
            self.current_command = "BACKWARD"
        
        # A: 왼쪽 회전
        if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
            self.current_step_alpha = -self.step_alpha
            self.current_command = "TURN_LEFT"
        
        # D: 오른쪽 회전
        if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
            self.current_step_alpha = self.step_alpha
            self.current_command = "TURN_RIGHT"
        
        # Q: 전진 + 왼쪽 회전 (커브)
        if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN:
            self.current_step_length = self.step_length
            self.current_step_alpha = -self.step_alpha * 0.7  # 커브는 회전 각도 줄임
            self.current_command = "CURVE_LEFT"
        
        # E: 전진 + 오른쪽 회전 (커브)
        if ord('e') in keys and keys[ord('e')] & p.KEY_IS_DOWN:
            self.current_step_length = self.step_length
            self.current_step_alpha = self.step_alpha * 0.7
            self.current_command = "CURVE_RIGHT"
        
        # R: 리셋
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            self.reset_robot()
        
        # ESC: 종료 (ESC key ASCII code = 27)
        if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
            self.running = False
            print("[INFO] Exit requested")
    
    def check_stability(self):
        """안정성 확인 - 넘어지면 리셋"""
        _, body_orn = p.getBasePositionAndOrientation(self.robot)
        euler = p.getEulerFromQuaternion(body_orn)
        
        if abs(euler[0]) > math.pi/4 or abs(euler[1]) > math.pi/4:
            print("[WARNING] Robot fell over - Resetting...")
            self.reset_robot()
            return False
        return True
    
    def get_foot_positions(self, t):
        """
        시간 t에서의 4개 다리 발끝 위치 계산
        """
        # TrottingGait 파라미터 설정
        if self.current_step_length != 0.0 or self.current_step_alpha != 0.0:
            self.trotting.Sh = self.step_height
        else:
            self.trotting.Sh = 0.0  # 정지 시 발을 들지 않음
        
        kb_offset = {
            'IDstepLength': self.current_step_length,
            'IDstepWidth': 0.0,
            'IDstepAlpha': self.current_step_alpha
        }
        
        # TrottingGait로 발끝 위치 계산
        foot_positions = self.trotting.positions(t, kb_offset)
        
        return foot_positions
    
    def apply_ik(self, foot_positions):
        """
        발끝 위치로 IK 계산 후 관절에 적용
        """
        body_rot = (0, 0, 0)
        body_pos = (0, self.body_height, 0)
        
        # IK 계산
        angles = self.kinematics.calcIK(foot_positions, body_rot, body_pos)
        
        # 관절에 적용
        leg_names = ['front_left', 'front_right', 'rear_left', 'rear_right']
        part_names = ['shoulder', 'leg', 'foot']
        
        for leg_idx, leg_name in enumerate(leg_names):
            for part_idx, part_name in enumerate(part_names):
                joint_name = f"{leg_name}_{part_name}"
                
                if joint_name in self.joint_name_to_id:
                    joint_id = self.joint_name_to_id[joint_name]
                    target_angle = angles[leg_idx][part_idx] * self.dirs[leg_idx][part_idx]
                    
                    p.setJointMotorControl2(
                        bodyIndex=self.robot,
                        jointIndex=joint_id,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=target_angle,
                        positionGain=self.kp,
                        velocityGain=self.kd,
                        force=self.max_force
                    )
    
    def update_camera(self):
        """카메라가 로봇을 따라가도록 업데이트"""
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot)
        
        # 로봇의 yaw 각도 얻기
        euler = p.getEulerFromQuaternion(robot_orn)
        yaw_deg = math.degrees(euler[2])
        
        p.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=45 + yaw_deg,  # 로봇 회전에 따라 카메라도 회전
            cameraPitch=-30,
            cameraTargetPosition=robot_pos
        )
    
    def run(self):
        """메인 시뮬레이션 루프"""
        frame_count = 0
        last_command = ""
        
        try:
            while self.running:
                # 시간 계산
                t = time.time() - self.start_time
                
                # 키보드 입력 확인
                self.check_keyboard()
                
                # 안정성 확인
                self.check_stability()
                
                # 발끝 위치 계산
                foot_positions = self.get_foot_positions(t)
                
                # IK 적용
                self.apply_ik(foot_positions)
                
                # 카메라 업데이트
                self.update_camera()
                
                # 상태 표시 (명령 변경 시)
                if self.current_command != last_command:
                    robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot)
                    euler = p.getEulerFromQuaternion(robot_orn)
                    yaw_deg = math.degrees(euler[2])
                    
                    print(f"[{self.current_command}] StepLen: {self.current_step_length:.0f}mm | "
                          f"Alpha: {self.current_step_alpha:.1f}° | Yaw: {yaw_deg:.1f}°")
                    last_command = self.current_command
                
                # 주기적 상태 출력 (2초마다)
                frame_count += 1
                if frame_count % 480 == 0:
                    robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
                    distance = math.sqrt(robot_pos[0]**2 + robot_pos[1]**2)
                    print(f"  Distance: {distance:.2f}m | Pos: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})")
                
                # 짧은 대기
                time.sleep(1./240.)
                
        except KeyboardInterrupt:
            print("\n[INFO] Simulation terminated by user (Ctrl+C)")
        except p.error as e:
            # PyBullet 연결이 끊어진 경우 (창 닫기 등)
            print(f"[INFO] PyBullet connection closed: {e}")
        finally:
            try:
                p.disconnect()
                print("[INFO] PyBullet disconnected")
            except:
                pass


def main():
    """메인 함수"""
    sim = KeyboardControlledWalking()
    sim.run()


if __name__ == "__main__":
    main()
