"""
6-4.py: 기본 과제 - PyBullet 기반 TrottingGait 파라미터 실험

Week 06 - 5.1절 기본 과제 (PyBullet 시뮬레이션)
1. TrottingGait 파라미터 변경 실험 (t0, t1, t2, t3, Sh)
2. 각 다리의 Swing/Stance phase를 실시간으로 시각화

실행 방법:
    cd study/robert
    python 6-4.py
"""

import sys
import os
import numpy as np
import math
import time

# 상위 디렉토리의 모듈을 import하기 위한 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import pybullet as p
import pybullet_data


class SimpleTrottingGait:
    """
    TrottingGait 파라미터 실험을 위한 클래스
    """
    def __init__(self):
        # 기본 시간 파라미터 (밀리초)
        self.t0 = 300   # State 0: 지면 대기
        self.t1 = 1200  # State 1: 지면 이동 (Stance)
        self.t2 = 300   # State 2: 지면 대기
        self.t3 = 200   # State 3: 발 들어올림 (Swing)
        
        # 거리 파라미터
        self.Sl = 0.0   # Step Length (보폭)
        self.Sw = 0     # Step Width
        self.Sh = 40    # Step Height (발 들어올리는 높이)
        
        # 다리 오프셋
        self.Spf = 87
        self.Spr = 77
        self.Fo = 120
        self.Ro = 50

    def get_total_period(self):
        return self.t0 + self.t1 + self.t2 + self.t3

    def get_phase(self, t_ms):
        """현재 phase 반환"""
        Tt = self.get_total_period()
        t = t_ms % Tt
        
        if t < self.t0:
            return 'Wait1'
        elif t < self.t0 + self.t1:
            return 'Stance'
        elif t < self.t0 + self.t1 + self.t2:
            return 'Wait2'
        else:
            return 'Swing'

    def calcLeg(self, t_ms, x, y, z):
        """단일 다리의 발끝 위치 계산"""
        startLp = np.array([x - self.Sl/2.0, y, z - self.Sw, 1])
        endLp = np.array([x + self.Sl/2, y, z + self.Sw, 1])
        
        Tt = self.get_total_period()
        t = t_ms % Tt
        
        if t < self.t0:
            return startLp, 'Wait1'
        elif t < self.t0 + self.t1:
            td = t - self.t0
            tp = td / self.t1 if self.t1 > 0 else 0
            diffLp = endLp - startLp
            curLp = startLp + diffLp * tp
            return curLp, 'Stance'
        elif t < self.t0 + self.t1 + self.t2:
            return endLp, 'Wait2'
        else:
            td = t - (self.t0 + self.t1 + self.t2)
            tp = td / self.t3 if self.t3 > 0 else 0
            diffLp = startLp - endLp
            curLp = endLp + diffLp * tp
            curLp[1] += self.Sh * math.sin(math.pi * tp)
            return curLp, 'Swing'

    def positions(self, t_sec):
        """4개 다리의 발끝 위치 및 phase 정보 계산"""
        Tt = self.get_total_period()
        Tt2 = Tt / 2
        t_ms = t_sec * 1000
        
        td = t_ms % Tt
        t2 = (t_ms - Tt2) % Tt
        
        Fx, Rx = self.Fo, -self.Ro
        Fy, Ry = -100, -100
        
        results = [
            self.calcLeg(td, Fx, Fy, self.Spf),
            self.calcLeg(t2, Fx, Fy, -self.Spf),
            self.calcLeg(t2, Rx, Ry, self.Spr),
            self.calcLeg(td, Rx, Ry, -self.Spr)
        ]
        
        positions = np.array([r[0] for r in results])
        phases = [r[1] for r in results]
        
        return positions, phases


class SimpleKinematic:
    """역기구학"""
    def __init__(self):
        self.l1 = 50
        self.l2 = 20
        self.l3 = 100
        self.l4 = 100

    def legIK(self, point):
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
        D = max(-1, min(1, D))
        
        try:
            theta3 = math.acos(D)
        except ValueError:
            theta3 = 0
        
        theta2 = math.atan2(z, G) - math.atan2(l4 * math.sin(theta3), l3 + l4 * math.cos(theta3))
        
        return (theta1, theta2, theta3)


class PyBulletParameterExperiment:
    """
    PyBullet 기반 TrottingGait 파라미터 실험
    """
    def __init__(self):
        # PyBullet 초기화
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        p.setPhysicsEngineParameter(
            fixedTimeStep=1./240.,
            numSolverIterations=50,
            numSubSteps=4
        )

        # 지면 로드
        self.plane = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane, -1, lateralFriction=1.0)
        
        # 로봇 URDF 로드 (공중에 고정)
        urdf_path = self._find_urdf()
        if urdf_path:
            self.robot = p.loadURDF(
                urdf_path, 
                [0, 0, 0.3],  # 더 높은 위치
                useFixedBase=True  # 공중에 고정 (넘어지지 않음)
            )
            self.robot_loaded = True
            print(f"URDF 로드 성공: {urdf_path}")
            print("로봇이 공중에 고정되어 있습니다 (useFixedBase=True)")
        else:
            print("URDF 파일을 찾을 수 없습니다.")
            self.robot = None
            self.robot_loaded = False
            return

        # 보행 패턴 및 기구학
        self.trotting = SimpleTrottingGait()
        self.kinematics = SimpleKinematic()

        # 관절 정보 수집
        self.joint_info = self._get_joint_info()
        
        # PD 제어 파라미터
        self.kp = 0.045
        self.kd = 0.4
        self.max_force = 25.0

        # GUI 슬라이더 추가 - 파라미터 조절용
        self._create_sliders()
        
        # 카메라 설정
        p.resetDebugVisualizerCamera(
            cameraDistance=0.8,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.15]
        )
        
        # Phase 표시용 디버그 텍스트
        self.phase_text_ids = [None, None, None, None]
        
        # 초기 자세 설정
        self._set_home_position()
        
        self.connected = True
        self.start_time = time.time()

    def _find_urdf(self):
        possible_paths = [
            os.path.join(project_root, "urdf", "spotmicroai_gen.urdf.xml"),
            os.path.join(project_root, "urdf", "spotmicroai.urdf"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    def _get_joint_info(self):
        joint_info = {}
        num_joints = p.getNumJoints(self.robot)
        for i in range(num_joints):
            info = p.getJointInfo(self.robot, i)
            joint_name = info[1].decode('utf-8')
            joint_info[joint_name] = i
        print(f"관절 개수: {num_joints}")
        return joint_info

    def _create_sliders(self):
        """파라미터 조절 슬라이더 생성"""
        # 시간 파라미터
        self.slider_t0 = p.addUserDebugParameter("t0 (Wait1 ms)", 0, 500, 300)
        self.slider_t1 = p.addUserDebugParameter("t1 (Stance ms)", 500, 2000, 1200)
        self.slider_t2 = p.addUserDebugParameter("t2 (Wait2 ms)", 0, 500, 300)
        self.slider_t3 = p.addUserDebugParameter("t3 (Swing ms)", 50, 500, 200)
        
        # 높이 파라미터
        self.slider_sh = p.addUserDebugParameter("Sh (Step Height)", 10, 100, 40)
        self.slider_height = p.addUserDebugParameter("Body Height", -50, 50, 0)
        
        # 시작/정지
        self.slider_start = p.addUserDebugParameter("Start Walking", 0, 1, 0)

    def _set_home_position(self):
        """초기 홈 포지션 설정"""
        if not self.robot_loaded:
            return
        
        W = 120
        initial_foot_positions = [
            [120, -100, W/2, 1],
            [120, -100, -W/2, 1],
            [-50, -100, W/2, 1],
            [-50, -100, -W/2, 1]
        ]
        
        leg_names = ['front_left', 'front_right', 'rear_left', 'rear_right']
        
        for leg_name, foot_pos in zip(leg_names, initial_foot_positions):
            try:
                angles = self.kinematics.legIK(foot_pos)
                for i, part in enumerate(['shoulder', 'leg', 'foot']):
                    joint_name = f"{leg_name}_{part}"
                    if joint_name in self.joint_info:
                        joint_id = self.joint_info[joint_name]
                        angle = angles[i]
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
        
        for _ in range(120):
            p.stepSimulation()

    def _update_phase_display(self, phases):
        """Phase 정보를 화면에 표시"""
        leg_names = ['FL', 'FR', 'RL', 'RR']
        phase_colors = {
            'Wait1': [0.5, 0.5, 0.5],
            'Stance': [1, 0, 0],
            'Wait2': [1, 0.5, 0],
            'Swing': [0, 0, 1]
        }
        
        for i, (name, phase) in enumerate(zip(leg_names, phases)):
            text = f"{name}: {phase}"
            color = phase_colors.get(phase, [1, 1, 1])
            
            if self.phase_text_ids[i] is not None:
                p.removeUserDebugItem(self.phase_text_ids[i])
            
            self.phase_text_ids[i] = p.addUserDebugText(
                text,
                [0.3, 0.2 - i * 0.1, 0.3],
                textColorRGB=color,
                textSize=1.5
            )

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
            # 슬라이더에서 파라미터 읽기
            self.trotting.t0 = p.readUserDebugParameter(self.slider_t0)
            self.trotting.t1 = p.readUserDebugParameter(self.slider_t1)
            self.trotting.t2 = p.readUserDebugParameter(self.slider_t2)
            self.trotting.t3 = p.readUserDebugParameter(self.slider_t3)
            self.trotting.Sh = p.readUserDebugParameter(self.slider_sh)
            height = p.readUserDebugParameter(self.slider_height)
            walking = p.readUserDebugParameter(self.slider_start)
        except:
            self.connected = False
            return elapsed
        
        leg_names = ['front_left', 'front_right', 'rear_left', 'rear_right']
        
        if walking > 0.5:
            foot_positions, phases = self.trotting.positions(elapsed)
            
            # Phase 표시
            self._update_phase_display(phases)
            
            for leg_name, foot_pos in zip(leg_names, foot_positions):
                try:
                    foot_pos_adjusted = foot_pos.copy()
                    foot_pos_adjusted[1] -= height
                    
                    angles = self.kinematics.legIK(foot_pos_adjusted)
                    self.set_leg_angles(leg_name, angles)
                except:
                    pass
        
        try:
            p.stepSimulation()
        except:
            self.connected = False
            
        return elapsed

    def run(self):
        """시뮬레이션 실행"""
        print("=" * 60)
        print("Week 06: PyBullet 기반 TrottingGait 파라미터 실험")
        print("=" * 60)
        print()
        print("파라미터 조절:")
        print("  - t0, t1, t2, t3: 각 Phase 시간 (ms)")
        print("  - Sh: 발 들어올리는 높이 (mm)")
        print("  - Body Height: 몸체 높이 조절")
        print("  - Start Walking: 1로 올리면 보행 시작")
        print()
        print("Phase 색상:")
        print("  - 회색: Wait, 빨강: Stance, 파랑: Swing")
        print()

        try:
            while self.connected:
                elapsed = self.step()
                
                if int(elapsed * 5) % 10 == 0 and elapsed > 0:
                    Tt = self.trotting.get_total_period()
                    print(f"\rTime: {elapsed:.1f}s | Period: {Tt:.0f}ms | Sh: {self.trotting.Sh:.0f}", end="", flush=True)
                
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
    sim = PyBulletParameterExperiment()
    if sim.robot_loaded:
        sim.run()
    else:
        print("로봇 로드 실패")


if __name__ == "__main__":
    main()
