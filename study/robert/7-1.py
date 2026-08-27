"""
7-1.py: Body Height/Pitch/Roll Control Simulation
Week 07 - Body Pose Control with PyBullet

이 시뮬레이션은 GUI 슬라이더를 사용하여 로봇 몸체의 
Height, Pitch, Roll을 실시간으로 조정할 수 있습니다.

조작 방법:
- Height 슬라이더: 몸체 높이 조정
- Pitch 슬라이더: 전후 기울기 조정
- Roll 슬라이더: 좌우 기울기 조정
- Yaw 슬라이더: 방향 회전 조정
- R 키: 로봇 위치 리셋
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


class BodyController:
    """
    Body 자세 제어 클래스
    Height, Pitch, Roll, Yaw를 통해 로봇 몸체 자세를 조정합니다.
    """
    
    def __init__(self):
        # 다리 기본 위치 (x, y, z, 1) - FL, FR, RL, RR 순서
        # x: 전후, y: 상하(높이), z: 좌우
        self.default_positions = np.array([
            [120, -100, 87, 1],    # Front Left
            [120, -100, -87, 1],   # Front Right
            [-50, -100, 77, 1],    # Rear Left
            [-50, -100, -77, 1]    # Rear Right
        ], dtype=np.float64)
        
    def set_body_pose(self, height_offset=0, pitch_deg=0, roll_deg=0, yaw_deg=0):
        """
        Body 자세 통합 제어
        
        Parameters:
        -----------
        height_offset : float
            높이 오프셋 (mm), 양수: 몸체 UP, 음수: 몸체 DOWN
        pitch_deg : float
            Pitch 각도 (도), 양수: 앞으로 기울임
        roll_deg : float
            Roll 각도 (도), 양수: 왼쪽으로 기울임
        yaw_deg : float
            Yaw 각도 (도), 양수: 시계 방향 회전
        
        Returns:
        --------
        new_positions : numpy array
            조정된 발끝 위치 [FL, FR, RL, RR]
        """
        pitch_rad = math.radians(pitch_deg)
        roll_rad = math.radians(roll_deg)
        yaw_rad = math.radians(yaw_deg)
        
        new_positions = self.default_positions.copy()
        
        for i, pos in enumerate(new_positions):
            x_dist = pos[0]  # X 거리 (전후)
            z_dist = pos[2]  # Z 거리 (좌우)
            
            # 1. Height 적용 (몸체를 올리면 발끝은 상대적으로 아래로)
            y_offset = height_offset
            
            # 2. Pitch 적용 (전후 기울기)
            # 앞다리(x>0): pitch가 양수면 발끝이 올라가야 함 (y 감소)
            y_offset += x_dist * math.tan(pitch_rad)
            
            # 3. Roll 적용 (좌우 기울기)
            # 왼쪽 다리(z>0): roll이 양수(왼쪽 기울임)면 발끝이 올라가야 함
            y_offset += z_dist * math.tan(roll_rad)
            
            new_positions[i][1] = self.default_positions[i][1] - y_offset
            
            # 4. Yaw 적용 (방향 회전)
            if abs(yaw_deg) > 0.001:
                orig_x = self.default_positions[i][0]
                orig_z = self.default_positions[i][2]
                new_positions[i][0] = orig_x * math.cos(yaw_rad) - orig_z * math.sin(yaw_rad)
                new_positions[i][2] = orig_x * math.sin(yaw_rad) + orig_z * math.cos(yaw_rad)
        
        return new_positions


class BodyPoseSimulation:
    """
    PyBullet에서 Body 자세 제어 시뮬레이션
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
        
        # 기구학 및 컨트롤러
        self.kinematics = Kinematic()
        self.body_ctrl = BodyController()
        
        # 관절 매핑
        self.joint_name_to_id = self._get_joint_names()
        
        # 다리 방향 보정 (URDF에 따라 다를 수 있음)
        self.dirs = [[-1, 1, 1], [1, 1, 1], [-1, 1, 1], [1, 1, 1]]
        
        # PD 제어 파라미터
        self.kp = 0.045
        self.kd = 0.4
        self.max_force = 25.0
        
        # GUI 슬라이더 생성
        self._create_sliders()
        
        # 카메라 설정
        p.resetDebugVisualizerCamera(
            cameraDistance=0.8,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.15]
        )
        
        # 안내 텍스트
        self._add_info_text()
        
        print("=" * 60)
        print("Body Pose Control Simulation")
        print("-" * 60)
        print("Controls:")
        print("  - Height slider: Adjust body height")
        print("  - Pitch slider:  Tilt forward/backward")
        print("  - Roll slider:   Tilt left/right")
        print("  - Yaw slider:    Rotate body")
        print("  - R key:         Reset robot position")
        print("  - Ctrl+C:        Exit simulation")
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
        self.height_slider = p.addUserDebugParameter("Height", -50, 50, 0)
        self.pitch_slider = p.addUserDebugParameter("Pitch (deg)", -25, 25, 0)
        self.roll_slider = p.addUserDebugParameter("Roll (deg)", -25, 25, 0)
        self.yaw_slider = p.addUserDebugParameter("Yaw (deg)", -30, 30, 0)
        
        # PD 게인 슬라이더
        self.kp_slider = p.addUserDebugParameter("Kp", 0, 0.1, self.kp)
        self.kd_slider = p.addUserDebugParameter("Kd", 0, 1.0, self.kd)
        self.force_slider = p.addUserDebugParameter("Max Force", 0, 50, self.max_force)
    
    def _add_info_text(self):
        """정보 텍스트 추가"""
        p.addUserDebugText(
            "Body Pose Control",
            [0, 0, 0.5],
            textColorRGB=[1, 1, 1],
            textSize=1.5
        )
    
    def reset_robot(self):
        """로봇 위치 리셋"""
        p.resetBasePositionAndOrientation(
            self.robot,
            self.init_position,
            self.init_orientation
        )
        p.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0])
        print("[INFO] Robot position reset")
    
    def check_keyboard(self):
        """키보드 입력 확인"""
        keys = p.getKeyboardEvents()
        
        # R 키: 리셋
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            self.reset_robot()
    
    def apply_ik(self, foot_positions, body_rot=(0, 0, 0), body_pos=(0, 0, 0)):
        """
        발끝 위치로 IK 계산 후 관절에 적용
        
        Parameters:
        -----------
        foot_positions : numpy array
            4개 다리의 발끝 위치
        body_rot : tuple
            Body rotation (roll, pitch, yaw) - 추가적인 body IK용
        body_pos : tuple
            Body position (x, y, z)
        """
        # IK 계산
        angles = self.kinematics.calcIK(foot_positions, body_rot, body_pos)
        
        # 관절에 적용
        leg_names = ['front_left', 'front_right', 'rear_left', 'rear_right']
        part_names = ['shoulder', 'leg', 'foot']
        
        # PD 게인 읽기
        kp = p.readUserDebugParameter(self.kp_slider)
        kd = p.readUserDebugParameter(self.kd_slider)
        max_force = p.readUserDebugParameter(self.force_slider)
        
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
                        positionGain=kp,
                        velocityGain=kd,
                        force=max_force
                    )
    
    def check_stability(self):
        """안정성 확인 - 넘어지면 리셋"""
        _, body_orn = p.getBasePositionAndOrientation(self.robot)
        euler = p.getEulerFromQuaternion(body_orn)
        
        # Roll 또는 Pitch가 45도 이상이면 리셋
        if abs(euler[0]) > math.pi/4 or abs(euler[1]) > math.pi/4:
            print("[WARNING] Robot fell over - Resetting...")
            self.reset_robot()
            return False
        return True
    
    def run(self):
        """메인 시뮬레이션 루프"""
        try:
            while True:
                # 키보드 입력 확인
                self.check_keyboard()
                
                # 안정성 확인
                self.check_stability()
                
                # 슬라이더 값 읽기
                height = p.readUserDebugParameter(self.height_slider)
                pitch = p.readUserDebugParameter(self.pitch_slider)
                roll = p.readUserDebugParameter(self.roll_slider)
                yaw = p.readUserDebugParameter(self.yaw_slider)
                
                # Body 자세 계산
                foot_positions = self.body_ctrl.set_body_pose(
                    height_offset=height,
                    pitch_deg=pitch,
                    roll_deg=roll,
                    yaw_deg=yaw
                )
                
                # Body IK를 위한 rotation 변환
                body_rot = (
                    math.radians(roll),   # omega (roll)
                    math.radians(pitch),  # phi (pitch)  
                    math.radians(yaw)     # psi (yaw)
                )
                body_pos = (0, height, 0)  # (x, y, z)
                
                # IK 적용
                self.apply_ik(foot_positions, body_rot=(0, 0, 0), body_pos=(0, 0, 0))
                
                # 짧은 대기 (CPU 사용량 감소)
                time.sleep(1./240.)
                
        except KeyboardInterrupt:
            print("\n[INFO] Simulation terminated by user")
        finally:
            p.disconnect()
            print("[INFO] PyBullet disconnected")


def main():
    """메인 함수"""
    sim = BodyPoseSimulation()
    sim.run()


if __name__ == "__main__":
    main()
