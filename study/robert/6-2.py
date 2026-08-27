"""
6-2.py: PyBullet 제자리 걷기 시뮬레이션

Week 06 - 4.1, 4.2절 코드를 통합한 완전한 실행 파일
TrottingGait 클래스를 사용하여 PyBullet에서 제자리 걷기를 시뮬레이션합니다.

실행 방법:
    cd study/robert
    python 6-2.py

키보드 조작:
    - GUI 슬라이더로 Height, Step Height 조절
    - Ctrl+C로 종료
"""

import sys
import os

# 상위 디렉토리의 모듈을 import하기 위한 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import pybullet as p
import pybullet_data
import time
import numpy as np
import math


class SimpleTrottingGait:
    """
    TrottingGait의 간소화된 독립 구현
    """
    def __init__(self):
        # 시간 파라미터 (밀리초)
        self.t0 = 300   # State 0: 지면 대기
        self.t1 = 1200  # State 1: 지면 이동 (Stance)
        self.t2 = 300   # State 2: 지면 대기
        self.t3 = 200   # State 3: 발 들어올림 (Swing)
        
        # 거리 파라미터
        self.Sl = 0.0   # Step Length (보폭)
        self.Sw = 0     # Step Width
        self.Sh = 40    # Step Height (발 들어올리는 높이)
        self.Sa = 0     # Step Alpha (회전)
        
        # 다리 오프셋
        self.Spf = 87   # Front leg spur width
        self.Spr = 77   # Rear leg spur width
        self.Fo = 120   # Front leg X offset
        self.Ro = 50    # Rear leg X offset

    def calcLeg(self, t, x, y, z):
        """단일 다리의 발끝 위치 계산"""
        startLp = np.array([x - self.Sl/2.0, y, z - self.Sw, 1])
        endLp = np.array([x + self.Sl/2, y, z + self.Sw, 1])
        
        if t < self.t0:
            return startLp
        elif t < self.t0 + self.t1:
            td = t - self.t0
            tp = td / self.t1 if self.t1 > 0 else 0
            diffLp = endLp - startLp
            curLp = startLp + diffLp * tp
            return curLp
        elif t < self.t0 + self.t1 + self.t2:
            return endLp
        elif t < self.t0 + self.t1 + self.t2 + self.t3:
            td = t - (self.t0 + self.t1 + self.t2)
            tp = td / self.t3 if self.t3 > 0 else 0
            diffLp = startLp - endLp
            curLp = endLp + diffLp * tp
            curLp[1] += self.Sh * math.sin(math.pi * tp)
            return curLp
        return startLp

    def positions(self, t, kb_offset={}):
        """4개 다리의 발끝 위치 계산"""
        spf = self.Spf
        spr = self.Spr
        
        if kb_offset:
            self.Sl = kb_offset.get('IDstepLength', 0.0)
            self.Sw = kb_offset.get('IDstepWidth', 0.0)
            self.Sa = kb_offset.get('IDstepAlpha', 0.0)
        
        Tt = self.t0 + self.t1 + self.t2 + self.t3
        Tt2 = Tt / 2
        
        td = (t * 1000) % Tt
        t2 = (t * 1000 - Tt2) % Tt
        
        Fx = self.Fo
        Rx = -self.Ro
        Fy = -100
        Ry = -100
        
        return np.array([
            self.calcLeg(td, Fx, Fy, spf),
            self.calcLeg(t2, Fx, Fy, -spf),
            self.calcLeg(t2, Rx, Ry, spr),
            self.calcLeg(td, Rx, Ry, -spr)
        ])


class SimpleKinematic:
    """
    역기구학의 간소화된 독립 구현
    """
    def __init__(self):
        self.l1 = 50   # Shoulder offset
        self.l2 = 20   # Shoulder length
        self.l3 = 100  # Upper leg
        self.l4 = 100  # Lower leg

    def legIK(self, point):
        """역기구학: 발끝 위치 → 관절 각도"""
        x, y, z = point[0], point[1], point[2]
        l1, l2, l3, l4 = self.l1, self.l2, self.l3, self.l4
        
        try:
            F = math.sqrt(x**2 + y**2 - l1**2)
        except ValueError:
            F = l1
        
        G = F - l2
        H = math.sqrt(G**2 + z**2)
        
        theta1 = -math.atan2(y, x) - math.atan2(F, -l1)
        
        D = (H**2 - l3**2 - l4**2) / (2 * l3 * l4)
        D = max(-1, min(1, D))  # 범위 제한
        
        try:
            theta3 = math.acos(D)
        except ValueError:
            theta3 = 0
        
        theta2 = math.atan2(z, G) - math.atan2(l4 * math.sin(theta3), l3 + l4 * math.cos(theta3))
        
        return (theta1, theta2, theta3)


class PyBulletInPlaceWalking:
    """
    PyBullet에서 제자리 걷기 시뮬레이션
    """
    def __init__(self):
        # PyBullet 초기화
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # 물리 엔진 설정
        p.setPhysicsEngineParameter(
            fixedTimeStep=1./240.,
            numSolverIterations=50,
            numSubSteps=4
        )

        # 지면 로드
        self.plane = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane, -1, lateralFriction=1.0)
        
        # 로봇 URDF 로드
        urdf_path = self._find_urdf()
        if urdf_path:
            # 로봇을 더 높은 위치에서 시작
            self.robot = p.loadURDF(urdf_path, [0, 0, 0.25], 
                                   p.getQuaternionFromEuler([0, 0, 0]))
            self.robot_loaded = True
            print(f"URDF 로드 성공: {urdf_path}")
        else:
            print("Warning: URDF 파일을 찾을 수 없습니다.")
            self.robot = None
            self.robot_loaded = False
            return

        # 보행 패턴 및 기구학 (항상 SimpleTrottingGait 사용 - 안정적)
        self.trotting = SimpleTrottingGait()
        self.kinematics = SimpleKinematic()

        # 관절 정보 수집
        self.joint_info = self._get_joint_info()
        
        # PD 제어 파라미터
        self.kp = 0.045
        self.kd = 0.4
        self.max_force = 25.0

        # 시작 시간 (3초 대기 후 보행 시작)
        self.start_time = time.time()
        self.gait_start_delay = 3.0  # 초기 자세 안정화 대기 시간

        # GUI 슬라이더 추가
        self.height_slider = p.addUserDebugParameter("Height", -50, 50, 0)
        self.step_height_slider = p.addUserDebugParameter("Step Height", 0, 100, 40)
        self.start_walking = p.addUserDebugParameter("Start Walking", 0, 1, 0)
        
        # 카메라 설정
        p.resetDebugVisualizerCamera(
            cameraDistance=0.8,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.15]
        )
        
        # 초기 자세 설정
        self._set_home_position()
        
        # 연결 상태
        self.connected = True

    def _find_urdf(self):
        """URDF 파일 경로 탐색"""
        possible_paths = [
            os.path.join(project_root, "urdf", "spotmicroai_gen.urdf.xml"),
            os.path.join(project_root, "urdf", "spotmicroai.urdf"),
            "../../urdf/spotmicroai_gen.urdf.xml",
            "../../urdf/spotmicroai.urdf",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None

    def _get_joint_info(self):
        """관절 정보 수집"""
        joint_info = {}
        num_joints = p.getNumJoints(self.robot)
        
        for i in range(num_joints):
            info = p.getJointInfo(self.robot, i)
            joint_name = info[1].decode('utf-8')
            joint_info[joint_name] = i
            
        print(f"관절 개수: {num_joints}")
        return joint_info

    def _set_home_position(self):
        """초기 홈 포지션 설정 - 안정적인 서있는 자세"""
        if not self.robot_loaded:
            return
        
        # spotmicroai.py에서 사용하는 초기 발 위치 (Lp)
        # [X, Y, Z, 1] 형태 - Y가 높이(아래 방향이 음수)
        W = 120  # 몸체 폭/2
        initial_foot_positions = [
            [120, -100, W/2, 1],    # Front Left
            [120, -100, -W/2, 1],   # Front Right  
            [-50, -100, W/2, 1],    # Rear Left
            [-50, -100, -W/2, 1]    # Rear Right
        ]
        
        leg_names = ['front_left', 'front_right', 'rear_left', 'rear_right']
        
        # 각 다리에 IK로 계산한 각도 적용
        for leg_idx, (leg_name, foot_pos) in enumerate(zip(leg_names, initial_foot_positions)):
            try:
                # IK 계산
                angles = self.kinematics.legIK(foot_pos)
                
                for i, part in enumerate(['shoulder', 'leg', 'foot']):
                    joint_name = f"{leg_name}_{part}"
                    if joint_name in self.joint_info:
                        joint_id = self.joint_info[joint_name]
                        angle = angles[i]
                        
                        # 방향 보정 (오른쪽 다리는 shoulder 반전 필요)
                        if 'right' in leg_name and part == 'shoulder':
                            angle = -angle
                        
                        p.resetJointState(self.robot, joint_id, angle)
                        p.setJointMotorControl2(
                            self.robot, joint_id,
                            p.POSITION_CONTROL,
                            targetPosition=angle,
                            force=self.max_force,
                            positionGain=self.kp,
                            velocityGain=self.kd
                        )
            except Exception as e:
                print(f"IK 오류 ({leg_name}): {e}")
        
        # 초기 안정화를 위해 더 많은 스텝 시뮬레이션
        for _ in range(240):
            p.stepSimulation()
            time.sleep(1./240.)

    def get_foot_positions(self, t):
        """시간 t에서 발끝 위치 반환 (제자리 걷기)"""
        kb_offset = {
            'IDstepLength': 0.0,
            'IDstepWidth': 0.0,
            'IDstepAlpha': 0.0
        }
        return self.trotting.positions(t, kb_offset)

    def foot_to_angles(self, foot_pos):
        """발끝 위치를 관절 각도로 변환 (IK)"""
        x, y, z = foot_pos[0], foot_pos[1], foot_pos[2]
        angles = self.kinematics.legIK([x, y, z, 1])
        return angles

    def set_leg_angles(self, leg_name, angles):
        """한 다리의 관절 각도 설정"""
        if not self.robot_loaded:
            return
        
        parts = ['shoulder', 'leg', 'foot']
        
        for i, part in enumerate(parts):
            joint_name = f"{leg_name}_{part}"
            if joint_name in self.joint_info:
                joint_id = self.joint_info[joint_name]
                angle = angles[i]
                
                # 방향 보정
                if 'right' in leg_name and part == 'shoulder':
                    angle = -angle
                
                p.setJointMotorControl2(
                    self.robot, joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=angle,
                    force=self.max_force,
                    positionGain=self.kp,
                    velocityGain=self.kd
                )

    def step(self):
        """한 시뮬레이션 스텝 실행"""
        if not self.connected:
            return 0
            
        elapsed = time.time() - self.start_time
        
        try:
            # GUI에서 파라미터 읽기
            height = p.readUserDebugParameter(self.height_slider)
            step_height = p.readUserDebugParameter(self.step_height_slider)
            walking = p.readUserDebugParameter(self.start_walking)
            self.trotting.Sh = step_height
        except:
            self.connected = False
            return elapsed
        
        leg_names = ['front_left', 'front_right', 'rear_left', 'rear_right']
        
        # 대기 시간 후에만 보행 시작
        if elapsed > self.gait_start_delay and walking > 0.5:
            gait_time = elapsed - self.gait_start_delay
            foot_positions = self.get_foot_positions(gait_time)
            
            for leg_name, foot_pos in zip(leg_names, foot_positions):
                try:
                    foot_pos_adjusted = foot_pos.copy()
                    foot_pos_adjusted[1] -= height
                    
                    angles = self.foot_to_angles(foot_pos_adjusted)
                    self.set_leg_angles(leg_name, angles)
                except Exception as e:
                    pass
        
        try:
            p.stepSimulation()
        except:
            self.connected = False
            
        return elapsed

    def run(self):
        """시뮬레이션 실행"""
        print("=" * 60)
        print("Week 06: PyBullet 제자리 걷기 시뮬레이션")
        print("=" * 60)
        print()
        print("조작 방법:")
        print("  - 'Start Walking' 슬라이더를 1로 올리면 보행 시작")
        print("  - 'Height' 슬라이더로 로봇 높이 조절")
        print("  - 'Step Height' 슬라이더로 발 높이 조절")
        print("  - 창을 닫거나 Ctrl+C로 종료")
        print()
        print(f"초기 자세 안정화 중... ({self.gait_start_delay}초 대기)")
        print()

        try:
            while self.connected:
                elapsed = self.step()
                
                if int(elapsed * 10) % 20 == 0 and elapsed > 0:
                    print(f"\rTime: {elapsed:.1f}s", end="", flush=True)
                
                time.sleep(1./240.)
                
        except KeyboardInterrupt:
            print("\n\n사용자에 의해 종료됨")
        finally:
            if self.connected:
                try:
                    p.disconnect()
                except:
                    pass
            print("시뮬레이션 종료")


def main():
    """메인 함수"""
    sim = PyBulletInPlaceWalking()
    if sim.robot_loaded:
        sim.run()
    else:
        print("로봇 로드 실패. 시뮬레이션을 시작할 수 없습니다.")


if __name__ == "__main__":
    main()

