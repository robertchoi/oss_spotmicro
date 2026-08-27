"""
7-5.py: Week 07 Assignment - Forward Walking + Turning with Parameter Tuning
파라미터 튜닝 실험 및 보고서 생성

이 코드는 전진 보행과 회전 보행의 파라미터를 실험하고,
결과를 자동으로 기록하여 보고서를 생성합니다.

기능:
1. 키보드로 로봇 제어 (W/A/S/D/Q/E)
2. 파라미터 실험 모드 (자동 테스트)
3. 실험 결과 CSV 저장
4. 마크다운 보고서 자동 생성

조작 방법:
- W/A/S/D/Q/E: 이동 및 회전
- T: 파라미터 튜닝 실험 시작
- R: 로봇 리셋
- P: 현재 결과 출력
- M: 마크다운 보고서 생성
- ESC: 종료
"""

import sys
import os
import csv
import time
import math
from datetime import datetime

# 상위 디렉토리를 경로에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..", "..")
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "Kinematics"))
sys.path.insert(0, os.path.join(project_root, "Simulation"))

import pybullet as p
import pybullet_data
import numpy as np

from Kinematics.kinematics import Kinematic
from Kinematics.kinematicMotion import TrottingGait


class ParameterTuningExperiment:
    """
    파라미터 튜닝 실험 클래스
    다양한 파라미터 조합을 테스트하고 결과를 기록합니다.
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
        self.dirs = [[-1, 1, 1], [1, 1, 1], [-1, 1, 1], [1, 1, 1]]
        
        # PD 제어 파라미터
        self.kp = 0.045
        self.kd = 0.4
        self.max_force = 25.0
        
        # 현재 파라미터
        self.step_length = 60.0
        self.step_height = 40.0
        self.step_alpha = 15.0
        self.body_height = 20.0
        
        # 현재 명령
        self.current_step_length = 0.0
        self.current_step_alpha = 0.0
        self.current_command = "STOP"
        
        # 시작 시간
        self.start_time = time.time()
        self.running = True
        
        # 실험 결과 저장
        self.experiment_results = []
        
        # GUI 슬라이더
        self._create_sliders()
        
        # 카메라 설정
        p.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.15]
        )
        
        self._print_help()
    
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
        self.step_length_slider = p.addUserDebugParameter("Step Length", 20, 120, self.step_length)
        self.step_height_slider = p.addUserDebugParameter("Step Height", 20, 80, self.step_height)
        self.step_alpha_slider = p.addUserDebugParameter("Turn Angle", 5, 25, self.step_alpha)
        self.height_slider = p.addUserDebugParameter("Body Height", -30, 50, self.body_height)
    
    def _print_help(self):
        """도움말 출력"""
        print("=" * 70)
        print("Week 07 Assignment: Parameter Tuning Experiment")
        print("=" * 70)
        print("Movement Controls:")
        print("  W: Forward    S: Backward    A: Turn Left    D: Turn Right")
        print("  Q: Curve Left E: Curve Right")
        print("-" * 70)
        print("Experiment Controls:")
        print("  T: Run automatic parameter tuning experiment")
        print("  P: Print current experiment results")
        print("  M: Generate markdown report")
        print("  R: Reset robot    ESC: Exit")
        print("=" * 70)
    
    def reset_robot(self):
        """로봇 위치 리셋"""
        p.resetBasePositionAndOrientation(
            self.robot,
            self.init_position,
            self.init_orientation
        )
        p.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0])
        self.start_time = time.time()
        print("[INFO] Robot reset")
    
    def check_keyboard(self):
        """키보드 입력 확인"""
        keys = p.getKeyboardEvents()
        
        # 슬라이더 값 읽기
        self.step_length = p.readUserDebugParameter(self.step_length_slider)
        self.step_height = p.readUserDebugParameter(self.step_height_slider)
        self.step_alpha = p.readUserDebugParameter(self.step_alpha_slider)
        self.body_height = p.readUserDebugParameter(self.height_slider)
        
        # 기본값
        self.current_step_length = 0.0
        self.current_step_alpha = 0.0
        self.current_command = "STOP"
        
        # 이동 명령
        if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
            self.current_step_length = self.step_length
            self.current_command = "FORWARD"
        if ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN:
            self.current_step_length = -self.step_length
            self.current_command = "BACKWARD"
        if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
            self.current_step_alpha = -self.step_alpha
            self.current_command = "TURN_LEFT"
        if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
            self.current_step_alpha = self.step_alpha
            self.current_command = "TURN_RIGHT"
        if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN:
            self.current_step_length = self.step_length
            self.current_step_alpha = -self.step_alpha * 0.7
            self.current_command = "CURVE_LEFT"
        if ord('e') in keys and keys[ord('e')] & p.KEY_IS_DOWN:
            self.current_step_length = self.step_length
            self.current_step_alpha = self.step_alpha * 0.7
            self.current_command = "CURVE_RIGHT"
        
        # 실험 명령
        if ord('t') in keys and keys[ord('t')] & p.KEY_WAS_TRIGGERED:
            self.run_experiments()
        if ord('p') in keys and keys[ord('p')] & p.KEY_WAS_TRIGGERED:
            self.print_results()
        if ord('m') in keys and keys[ord('m')] & p.KEY_WAS_TRIGGERED:
            self.generate_report()
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            self.reset_robot()
        if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
            self.running = False
    
    def check_stability(self):
        """안정성 확인"""
        _, body_orn = p.getBasePositionAndOrientation(self.robot)
        euler = p.getEulerFromQuaternion(body_orn)
        if abs(euler[0]) > math.pi/4 or abs(euler[1]) > math.pi/4:
            return False
        return True
    
    def get_foot_positions(self, t):
        """발끝 위치 계산"""
        self.trotting.Sh = self.step_height
        kb_offset = {
            'IDstepLength': self.current_step_length,
            'IDstepWidth': 0.0,
            'IDstepAlpha': self.current_step_alpha
        }
        return self.trotting.positions(t, kb_offset)
    
    def apply_ik(self, foot_positions):
        """IK 적용"""
        body_rot = (0, 0, 0)
        body_pos = (0, self.body_height, 0)
        angles = self.kinematics.calcIK(foot_positions, body_rot, body_pos)
        
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
    
    def run_single_experiment(self, step_length, step_height, step_alpha, duration=5.0, description=""):
        """단일 실험 수행"""
        print(f"\n[EXPERIMENT] {description}")
        print(f"  StepLength={step_length}mm, StepHeight={step_height}mm, StepAlpha={step_alpha}°")
        
        self.reset_robot()
        time.sleep(0.5)  # 안정화 대기
        
        # 초기 위치 기록
        init_pos, init_orn = p.getBasePositionAndOrientation(self.robot)
        init_yaw = math.degrees(p.getEulerFromQuaternion(init_orn)[2])
        
        # 파라미터 설정
        self.trotting.Sh = step_height
        self.current_step_length = step_length
        self.current_step_alpha = step_alpha
        
        start_time = time.time()
        stable = True
        
        while time.time() - start_time < duration:
            if not self.check_stability():
                stable = False
                print("  [!] Robot fell - experiment aborted")
                break
            
            t = time.time() - self.start_time
            foot_positions = self.get_foot_positions(t)
            self.apply_ik(foot_positions)
            
            robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=robot_pos
            )
            
            time.sleep(1./240.)
        
        # 최종 위치 기록
        final_pos, final_orn = p.getBasePositionAndOrientation(self.robot)
        final_yaw = math.degrees(p.getEulerFromQuaternion(final_orn)[2])
        
        # 결과 계산
        distance = math.sqrt(
            (final_pos[0] - init_pos[0])**2 + 
            (final_pos[1] - init_pos[1])**2
        )
        yaw_change = final_yaw - init_yaw
        
        stability = "Stable" if stable else "Unstable"
        
        result = {
            'description': description,
            'step_length': step_length,
            'step_height': step_height,
            'step_alpha': step_alpha,
            'duration': duration,
            'distance': distance,
            'yaw_change': yaw_change,
            'stability': stability
        }
        
        self.experiment_results.append(result)
        
        print(f"  Distance: {distance:.3f}m | Yaw: {yaw_change:.1f}° | {stability}")
        
        # 정지
        self.current_step_length = 0.0
        self.current_step_alpha = 0.0
        
        return result
    
    def run_experiments(self):
        """자동 파라미터 튜닝 실험"""
        print("\n" + "=" * 70)
        print("Starting Parameter Tuning Experiments...")
        print("=" * 70)
        
        self.experiment_results = []
        
        # 1. Step Length 실험 (전진 보행)
        print("\n--- Step Length Experiments (Forward Walking) ---")
        for sl in [30, 60, 90, 120]:
            self.run_single_experiment(
                step_length=sl, step_height=40, step_alpha=0,
                duration=5.0, description=f"Forward StepLen={sl}mm"
            )
        
        # 2. Step Height 실험
        print("\n--- Step Height Experiments ---")
        for sh in [20, 40, 60, 80]:
            self.run_single_experiment(
                step_length=60, step_height=sh, step_alpha=0,
                duration=5.0, description=f"Forward StepHeight={sh}mm"
            )
        
        # 3. Step Alpha 실험 (회전)
        print("\n--- Step Alpha Experiments (Turning) ---")
        for sa in [5, 10, 15, 20]:
            self.run_single_experiment(
                step_length=0, step_height=40, step_alpha=sa,
                duration=5.0, description=f"Turn Alpha={sa}°"
            )
        
        # 4. 커브 주행 실험
        print("\n--- Curve Walking Experiments ---")
        for sa in [5, 10, 15]:
            self.run_single_experiment(
                step_length=60, step_height=40, step_alpha=sa,
                duration=5.0, description=f"Curve StepLen=60 Alpha={sa}°"
            )
        
        print("\n" + "=" * 70)
        print(f"Experiments completed: {len(self.experiment_results)} tests")
        print("Press 'P' to print results, 'M' to generate report")
        print("=" * 70)
    
    def print_results(self):
        """결과 출력"""
        if not self.experiment_results:
            print("[INFO] No experiment results yet. Press 'T' to run experiments.")
            return
        
        print("\n" + "=" * 70)
        print("EXPERIMENT RESULTS")
        print("=" * 70)
        print(f"{'Description':<30} {'Dist(m)':<10} {'Yaw(°)':<10} {'Stability':<10}")
        print("-" * 70)
        
        for r in self.experiment_results:
            print(f"{r['description']:<30} {r['distance']:<10.3f} {r['yaw_change']:<10.1f} {r['stability']:<10}")
        
        print("=" * 70)
    
    def generate_report(self):
        """마크다운 보고서 생성"""
        if not self.experiment_results:
            print("[INFO] No experiment results. Press 'T' to run experiments first.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"week07_report_{timestamp}.md"
        filepath = os.path.join(script_dir, filename)
        
        # CSV 저장
        csv_filename = f"week07_data_{timestamp}.csv"
        csv_filepath = os.path.join(script_dir, csv_filename)
        
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.experiment_results[0].keys())
            writer.writeheader()
            writer.writerows(self.experiment_results)
        
        # 마크다운 보고서 생성
        report = f"""# 7주차 과제: 전진 보행 + 좌우 회전 파라미터 튜닝 보고서

**생성일시**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. 실험 환경

- **OS**: Windows
- **Python**: 3.x
- **Simulation**: PyBullet
- **Robot**: SpotMicroAI

## 2. 전진 보행 실험

### 2.1 Step Length 변화

| Sl (mm) | 이동 거리 (5초) | 안정성 | 비고 |
|---------|----------------|--------|------|
"""
        # Step Length 실험 결과
        for r in self.experiment_results:
            if 'Forward StepLen' in r['description']:
                sl = r['step_length']
                report += f"| {sl} | {r['distance']:.3f}m | {r['stability']} | - |\n"
        
        report += """
### 2.2 Step Height 변화

| Sh (mm) | 이동 거리 | 안정성 | 비고 |
|---------|----------|--------|------|
"""
        # Step Height 실험 결과
        for r in self.experiment_results:
            if 'StepHeight' in r['description']:
                sh = r['step_height']
                report += f"| {sh} | {r['distance']:.3f}m | {r['stability']} | - |\n"
        
        report += """
## 3. 회전 보행 실험

### 3.1 Step Alpha 변화 (제자리 회전)

| Sa (deg) | 회전 각도 (5초) | 안정성 | 비고 |
|----------|----------------|--------|------|
"""
        # Turn 실험 결과
        for r in self.experiment_results:
            if 'Turn Alpha' in r['description']:
                sa = r['step_alpha']
                report += f"| {sa} | {r['yaw_change']:.1f}° | {r['stability']} | - |\n"
        
        report += """
### 3.2 전진 + 회전 조합 (커브)

| Sl (mm) | Sa (deg) | 이동 거리 | 회전 각도 | 비고 |
|---------|----------|----------|----------|------|
"""
        # Curve 실험 결과
        for r in self.experiment_results:
            if 'Curve' in r['description']:
                report += f"| {r['step_length']} | {r['step_alpha']} | {r['distance']:.3f}m | {r['yaw_change']:.1f}° | - |\n"
        
        # 최적 파라미터 도출
        best_forward = max([r for r in self.experiment_results if 'Forward StepLen' in r['description']], 
                          key=lambda x: x['distance'] if x['stability'] == 'Stable' else 0, default=None)
        
        best_turn = max([r for r in self.experiment_results if 'Turn Alpha' in r['description']], 
                       key=lambda x: abs(x['yaw_change']) if x['stability'] == 'Stable' else 0, default=None)
        
        report += f"""
## 4. 최적 파라미터 도출

### 추천 설정

| 모드 | Step Length | Step Height | Step Alpha | 비고 |
|------|-------------|-------------|------------|------|
| 전진 | {best_forward['step_length'] if best_forward else 60}mm | 40mm | 0° | 안정적 전진 |
| 회전 | 0mm | 40mm | {best_turn['step_alpha'] if best_turn else 15}° | 제자리 회전 |
| 커브 | 60mm | 40mm | 10° | 안정적 커브 |

## 5. 결론

1. **전진 보행**: Step Length {best_forward['step_length'] if best_forward else 60}mm에서 최대 이동 거리 달성
2. **회전 보행**: Step Alpha {best_turn['step_alpha'] if best_turn else 15}°에서 효율적인 회전
3. **안정성**: 모든 실험에서 로봇의 안정성 유지 여부 확인

## 6. 원시 데이터

CSV 파일: `{csv_filename}`
"""
        
        # 파일 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n[SUCCESS] Report generated: {filename}")
        print(f"[SUCCESS] CSV data saved: {csv_filename}")
    
    def run(self):
        """메인 루프"""
        try:
            while self.running:
                self.check_keyboard()
                
                if not self.check_stability():
                    self.reset_robot()
                
                t = time.time() - self.start_time
                foot_positions = self.get_foot_positions(t)
                self.apply_ik(foot_positions)
                
                robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
                p.resetDebugVisualizerCamera(
                    cameraDistance=1.0,
                    cameraYaw=45,
                    cameraPitch=-30,
                    cameraTargetPosition=robot_pos
                )
                
                time.sleep(1./240.)
                
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        except p.error:
            print("[INFO] PyBullet connection closed")
        finally:
            try:
                p.disconnect()
            except:
                pass
            print("[INFO] Simulation ended")


def main():
    sim = ParameterTuningExperiment()
    sim.run()


if __name__ == "__main__":
    main()
