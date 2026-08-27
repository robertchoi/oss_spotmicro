# g1 unitree 사용
# g1_sim_node.py (처음 작성한 코드 -> 뒤로 이동하는 문제)

https://github.com/user-attachments/assets/1e14047b-0a91-4df5-84c1-e321293e8839


```
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
from math import sqrt, atan2, acos, pi, sin, cos


# =========================================================
# 1. Kinematics (기존 동일)
# =========================================================
class G1Kinematics:
    def __init__(self):
        self.l1 = 0.30
        self.l2 = 0.30

    def leg_IK(self, x, y, z, target_pitch=0.0, torso_pitch=0.0):
        # 특이점 방지 및 길이 제한
        dist = sqrt(x ** 2 + y ** 2 + z ** 2)
        dist = np.clip(dist, 0.1, self.l1 + self.l2 - 0.002)

        cos_knee = (self.l1 ** 2 + self.l2 ** 2 - dist ** 2) / (2 * self.l1 * self.l2)
        theta_knee = pi - acos(np.clip(cos_knee, -1.0, 1.0))

        alpha = acos(np.clip((self.l1 ** 2 + dist ** 2 - self.l2 ** 2) / (2 * self.l1 * dist), -1.0, 1.0))
        theta_hip_pitch_base = atan2(x, -z) - alpha

        # 상체 각도 유지 보정
        theta_hip_pitch = theta_hip_pitch_base + torso_pitch
        theta_hip_roll = atan2(y, -z)

        # 발바닥 수평 유지
        theta_ankle_pitch = (-theta_hip_pitch_base - theta_knee) + target_pitch - torso_pitch
        theta_ankle_roll = -theta_hip_roll

        return [theta_hip_pitch, theta_hip_roll, 0.0, theta_knee, theta_ankle_pitch, theta_ankle_roll]


# =========================================================
# 2. Bio-Gait Generator (발 위치 후퇴 & 지지력 강화)
# =========================================================
class G1BioGait:
    def __init__(self):
        self.cycle_time = 1.0
        self.stride_len = 0.15
        self.step_height = 0.04
        self.stand_height = -0.57

        self.sway_amp = 0.02
        self.foot_spacing = 0.04

        # [핵심 수정 1] 발 기본 위치를 뒤로 더 뺌 (뒤로 넘어짐 방지)
        # -0.02 -> -0.06 (6cm 뒤로)
        # 상체를 폈기 때문에 발을 뒤로 보내야 무게중심이 앞꿈치에 실림
        self.base_x = -0.06

        self.heel_strike_angle = 0.2
        self.toe_off_angle = -0.3

    def get_trajectory(self, t):
        phase = (t % self.cycle_time) / self.cycle_time

        body_y_offset = -self.sway_amp * sin(2 * pi * phase)

        # 기본 위치에 base_x 적용
        l_pos = [self.base_x, self.foot_spacing - body_y_offset, self.stand_height]
        r_pos = [self.base_x, -self.foot_spacing - body_y_offset, self.stand_height]

        l_pitch, r_pitch = 0.0, 0.0
        stance_leg = 0

        # === [왼발 Swing] ===
        if phase < 0.5:
            p = phase / 0.5
            stance_leg = 1  # 오른발 지지

            # 발 이동 (base_x 기준 앞뒤로)
            l_pos[0] += (-self.stride_len / 2 + self.stride_len * p)

            if 0.1 < p < 0.9:
                h_p = (p - 0.1) / 0.8
                l_pos[2] += self.step_height * sin(pi * h_p)
                if h_p > 0.8: l_pitch = self.heel_strike_angle * ((h_p - 0.8) / 0.2)

            r_pos[0] += (self.stride_len / 2 - self.stride_len * p)

            if p > 0.8:  # Toe Off
                r_pitch = self.toe_off_angle * ((p - 0.8) / 0.2)
                r_pos[2] -= 0.015 * ((p - 0.8) / 0.2)  # 더 강하게 밀기

        # === [오른발 Swing] ===
        else:
            p = (phase - 0.5) / 0.5
            stance_leg = 2  # 왼발 지지

            r_pos[0] += (-self.stride_len / 2 + self.stride_len * p)

            if 0.1 < p < 0.9:
                h_p = (p - 0.1) / 0.8
                r_pos[2] += self.step_height * sin(pi * h_p)
                if h_p > 0.8: r_pitch = self.heel_strike_angle * ((h_p - 0.8) / 0.2)

            l_pos[0] += (self.stride_len / 2 - self.stride_len * p)

            if p > 0.8:
                l_pitch = self.toe_off_angle * ((p - 0.8) / 0.2)
                l_pos[2] -= 0.015 * ((p - 0.8) / 0.2)

        return l_pos, r_pos, l_pitch, r_pitch, stance_leg


# =========================================================
# 3. Simulator (발목 힘 강화)
# =========================================================
class G1Sim:
    def __init__(self, xml_path):
        if not os.path.exists(xml_path): xml_path = "../description/scene.xml"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.kin = G1Kinematics()
        self.gait = G1BioGait()

        # Actuator 인덱스 찾기
        self.actuators = {}
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            self.actuators[name] = i

        # 기본 게인 설정 (안정화)
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            kp, kd = 500.0, 15.0

            # [핵심 수정 2] 발목(Ankle) 힘 대폭 강화
            # 뒤로 넘어가는 것을 발목 힘으로 버텨야 함
            if 'ankle_pitch' in name:
                kp = 800.0
                kd = 30.0
            elif 'knee' in name:
                kp = 600.0
                kd = 20.0
            elif 'waist' in name:
                kp = 1000.0
                kd = 50.0

            self.model.actuator_gainprm[i, 0] = kp
            self.model.actuator_biasprm[i, 1] = -kp
            self.model.actuator_biasprm[i, 2] = -kd

        if self.model.nkey > 0: mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # ID 그룹핑
        self.left_leg_idxs = [i for i in range(self.model.nu) if
                              'left' in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)]
        self.right_leg_idxs = [i for i in range(self.model.nu) if
                               'right' in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)]

    def set_stiffness(self, idxs, kp_scale):
        for i in idxs:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            base_kp = 500.0
            if 'ankle' in name:
                base_kp = 800.0  # 기본 발목 강성
            elif 'knee' in name:
                base_kp = 600.0

            new_kp = base_kp * kp_scale
            self.model.actuator_gainprm[i, 0] = new_kp
            self.model.actuator_biasprm[i, 1] = -new_kp
            self.model.actuator_biasprm[i, 2] = -new_kp * 0.05

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            print("=== G1 Upright & Stable Walk ===")

            is_walking = False
            walk_time = 0.0

            while viewer.is_running():
                step_start = time.time()
                if self.data.time > 3.0: is_walking = True
                if is_walking: walk_time += self.model.opt.timestep

                # 1. 궤적 생성
                if is_walking:
                    l_xyz, r_xyz, l_pitch, r_pitch, stance_leg = self.gait.get_trajectory(walk_time)
                else:
                    # 정지 상태에서도 base_x (-0.06) 유지하여 안 넘어지게 함
                    l_xyz = [self.gait.base_x, 0.04, self.gait.stand_height]
                    r_xyz = [self.gait.base_x, -0.04, self.gait.stand_height]
                    l_pitch, r_pitch = 0.0, 0.0
                    stance_leg = 0

                # 2. 가변 강성 (지지발 강화)
                if stance_leg == 1:  # 오른발 지지
                    self.set_stiffness(self.right_leg_idxs, 1.5)
                    self.set_stiffness(self.left_leg_idxs, 0.8)
                elif stance_leg == 2:  # 왼발 지지
                    self.set_stiffness(self.left_leg_idxs, 1.5)
                    self.set_stiffness(self.right_leg_idxs, 0.8)
                else:
                    self.set_stiffness(self.left_leg_idxs, 1.2)  # 정지 시 둘 다 강화
                    self.set_stiffness(self.right_leg_idxs, 1.2)

                lean_forward = 0.0  # 값이 클수록 앞으로 숙임
                # 3. IK (상체 0도 고정)
                l_angles = self.kin.leg_IK(*l_xyz, target_pitch=l_pitch, torso_pitch=lean_forward)
                r_angles = self.kin.leg_IK(*r_xyz, target_pitch=r_pitch, torso_pitch=lean_forward)

                # 4. 제어
                self.data.ctrl[0:6] = l_angles
                self.data.ctrl[6:12] = r_angles
                self.data.ctrl[14] = 0.0  # 상체 수직 고정

                if self.model.nkey > 0:
                    self.data.ctrl[15:29] = self.model.key_ctrl[0, 15:29]

                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                time_until = self.model.opt.timestep - (time.time() - step_start)
                if time_until > 0: time.sleep(time_until)


if __name__ == "__main__":
    sim = G1Sim("scene.xml")
    sim.run()
```

# 찾은 오픈소스 코드
https://www.youtube.com/@DanTorresRobotics
https://github.com/sunilgora/ZMPpreviewControl

# lib_ZMPctrl.py 분석 문서

본 문서는 `lib_ZMPctrl.py`가 구현하고 있는 알고리즘 구조와 이론적 기반을 체계적으로 정리한 문서이다. 해당 코드는 Kajita et al. (2003)의 논문 **"Biped Walking Pattern Generation by using Preview Control of Zero-Moment Point"**를 기반으로 한 이족 보행 제어 라이브러리이며, ZMP 기반 Preview Control 이론을 실제 로봇 시뮬레이션 환경(MuJoCo)에서 구현한 구조이다.

---

## 전체 구조 개요

`lib_ZMPctrl.py`는 기능적으로 다음 세 영역으로 구성된다.

1. 궤적 생성 (Preview Control 기반 ZMP 제어)
2. 전신 제어 (Inverse Kinematics 기반 관절 제어)
3. 시뮬레이션 및 제어 (MuJoCo + PD Controller)

각 구성 요소는 이론 → 수학 모델 → 제어기 → 시뮬레이션으로 연결되는 구조를 가진다.

---

## 1. 궤적 생성: Preview Control

핵심 함수는 `mpc2humn()`이며, Kajita 논문의 핵심 알고리즘인 **Preview Control of ZMP**를 구현한다. 해당 부분은 논문의 Section 3.2 ~ 3.3에 해당한다.

### 1.1 Cart-Table Model (LIPM 모델링)

로봇은 선형 역진자 모델(LIPM, Linear Inverted Pendulum Model) 기반 Cart-Table Model로 단순화된다.

상태공간 방정식:

x(k+1) = A x(k) + B u(k)
p(k) = C x(k)

상태 벡터 x = [x, ẋ, ẍ]^T
제어 입력 u = jerk
출력 p = ZMP

코드에서는 다음과 같이 이산화된 상태공간 모델을 정의한다.

* A 행렬: 등가속도 운동 모델 이산화
* B 행렬: jerk 입력 모델
* C 행렬: ZMP 방정식 (p = x − (zc/g)·ẍ)

여기서 zc는 CoM 높이, g는 중력가속도이다.

---

### 1.2 최적 제어 및 Riccati 방정식

ZMP 추종 오차와 제어 입력 크기를 최소화하는 비용함수를 정의한다.

비용함수 구조:

* ZMP 오차 최소화
* 상태 오차 최소화
* 제어 입력 최소화

확장 시스템(Augmented System)을 구성하여 에러 적분 항을 포함한 서보 제어 구조를 만들고, DARE(Discrete Algebraic Riccati Equation)를 풀어 최적 피드백 게인을 계산한다.

사용 알고리즘:

* Discrete Algebraic Riccati Equation (DARE)
* Optimal State Feedback Control

출력:

* Gi : 적분 오차 게인
* Gx : 상태 피드백 게인

---

### 1.3 Preview Control 구조

미래의 ZMP 참조값을 미리 반영하여 현재 제어 입력을 결정하는 구조이다.

제어 입력 구조:

u(k) = −Gi · Σe(k) − Gx · x(k) − Σ Gp(j) · pref(k+j)

구성 요소:

* 현재 상태 피드백
* 누적 ZMP 오차 적분
* 미래 ZMP 참조값(Preview Horizon) 반영

코드 구조:

* Preview Gain (Gd) 계산
* 미래 ZMP 참조값(zmp_ref)을 미리 불러와 합산
* 현재 jerk 제어 입력 u 계산

이 구조를 통해 로봇은 미래 발 위치를 예측하여 무게중심(CoM)을 선제적으로 이동시킨다.

---

## 2. 전신 제어: Inverse Kinematics

Preview Control은 CoM 궤적만 생성한다. 실제 로봇 보행을 위해 이를 관절 각도로 변환해야 한다.

### 2.1 수치적 역기구학 (numik)

Jacobian 기반 수치해석적 IK 구조를 사용한다.

구성 요소:

* CoM Jacobian
* 양발 위치 Jacobian
* 양발 회전 Jacobian

작업 우선순위 구조(Task Stack):
1순위: CoM 위치 제어
2순위: 발 위치 제어
3순위: 발 자세 제어

Jacobian을 Stack 구조로 결합한 뒤, 의사역행렬(Pseudo-inverse)을 통해 관절 변화량을 계산한다.

반복 수렴 구조:

* 오차 계산
* Jacobian 기반 Δq 계산
* 반복 업데이트
* 오차 수렴 시 종료

---

### 2.2 궤적 변환 (cart2joint)

Preview Control에서 생성된 CoM 궤적과 발 궤적을 시간 순서대로 numik에 입력하여 전체 관절 궤적(qtraj)을 생성한다.

출력:

* 시간 기반 Joint Trajectory
* MuJoCo 제어용 목표 관절각 시퀀스

---

## 3. 보행 패턴 생성 및 지형 처리

### 3.1 보행 패턴 생성 (updateGait)

Swing Foot Trajectory를 생성한다.

구현 방식:

* Cubic Spline(3차 스플라인) 기반 궤적 생성
* 발 들기(zSw) 및 착지 궤적 부드럽게 생성
* ZMP 참조 경로와 연동

---

### 3.2 지형 적응 모델 (trnparam, DepthvsForce)

지형 물리 특성 모델링:

* 지면 강성(Stiffness)
* 감쇠(Damping)
* 계단 높이(stairs_height)

지면 반력 및 접촉 안정성 시뮬레이션을 위한 파라미터 구조이다.

---

## 4. 시뮬레이션 및 제어 구조

### 4.1 MuJoCo 인터페이스

`myRobot` 클래스가 MuJoCo model/data와 제어 알고리즘 사이의 인터페이스 역할을 수행한다.

역할:

* 로봇 상태 추출(CoM, 관절각, 발 위치)
* 제어 입력 전달

---

### 4.2 Controller 구조

제어 구조:

* 목표 관절각(qdes)
* 실제 관절각(q)
* PD 제어 기반 추종

구성:

* PIDcontrol 함수
* 관절별 독립 제어 구조

확장 구조(주석 처리):

* COM feedback control
* ZMP feedback control
* 외란 안정화 제어기

---

## 5. 전체 실행 흐름

1. 로봇 및 환경 로드 (selectRobot)
2. 지면 파라미터 설정 (trnparam)
3. 보행 궤적 생성 (mpc2humn)

   * LIPM 모델 기반 Preview Control
   * ZMP Reference 기반 CoM Trajectory 생성
4. 관절 궤적 변환 (cart2joint)

   * Numerical IK 기반 Joint Trajectory 생성
5. 시뮬레이션 실행 (sim)

   * MuJoCo 환경
   * PD Controller 기반 추종 제어

---

## 결론

`lib_ZMPctrl.py`는 Kajita et al. (2003) 논문의 Preview Control 이론을 기반으로 한 ZMP 보행 제어 구조를 충실히 구현한 라이브러리이다.

특징:

* LIPM 기반 동역학 모델
* Optimal Control + Riccati 기반 제어기
* Preview Control 구조
* Numerical IK 기반 전신 제어
* MuJoCo 시뮬레이션 연동

이 코드는 이론 → 수식 → 제어기 → 시뮬레이션으로 이어지는 구조를 명확하게 구현한 연구용 보행 제어 프레임워크이며, 이족보행 제어 이론 학습, 검증, 실험용으로 적합한 구조를 가진다.

---

# g1_sim_node.py 분석 및 보행 방식 비교 문서

본 문서는 `g1_sim_node.py`의 구조와 제어 방식을 분석하고, 기존 `lib_ZMPctrl.py` 및 SpotMicroAI(4족 보행 로봇) 코드와의 구조적·이론적 차이를 체계적으로 정리한 문서이다.

---

## 1. g1_sim_node.py 구조 분석

`g1_sim_node.py`는 물리 기반 균형 계산(ZMP, Dynamics)을 사용하지 않고, **기하학적(Geometric) 역기구학 + 사인파(Sine Wave) 패턴 기반 보행 제어기** 구조로 설계되어 있다.

핵심 철학은 “균형을 계산하지 않고, 정해진 패턴을 강한 모터 출력으로 수행한다”는 방식이다.

---

### 1.1 G1Kinematics (기구학 모델)

발끝 위치 (x, y, z)가 주어졌을 때, 관절각(Hip, Knee, Ankle)을 계산하는 해석적(Analytical) 역기구학 구조이다.

특징:

* Jacobian 기반 수치해석 IK가 아닌 기하학적 공식 기반 계산
* 코사인 법칙(Cosine Rule), 삼각함수 기반 구조
* 계산 속도 매우 빠름
* 로봇 링크 구조 변경 시 수식 재유도 필요

이는 범용성보다 **속도와 단순성**을 우선한 설계 방식이다.

---

### 1.2 G1BioGait (보행 패턴 생성기)

보행은 로봇의 물리 상태와 무관하게 시간(t)에 따라 생성된다.

구조:

* Sine Wave 기반 발 궤적 생성
* 주기적 패턴 반복 구조

주요 파라미터:

* cycle_time : 보행 주기
* stride_len : 보폭
* base_x : 무게중심 보정 오프셋
* sway_amp : 좌우 흔들림 진폭

물리적 균형(ZMP, CoM Dynamics)은 계산하지 않으며, **시간 기반 패턴 제어(Time-based Pattern Control)** 구조이다.

---

### 1.3 G1Sim (시뮬레이션 및 강성 제어 구조)

균형 제어기가 없기 때문에, 관절 제어는 매우 높은 강성(Stiffness) 기반 Position Control 구조를 사용한다.

특징:

* 높은 kp 값 사용
* 발목 관절 강성 강화
* 지지발(Stance Foot) 강성 증가
* 스윙발(Swing Foot) 강성 감소

이는 물리적 안정성 대신 **강제 안정성(Forced Stability)** 구조이다.

---

## 2. lib_ZMPctrl.py vs g1_sim_node.py 비교

### 제어 철학

* lib_ZMPctrl.py : 동역학 기반 제어 (Dynamic Control)

  * 무게중심(CoM)과 ZMP를 계산하여 균형 유지

* g1_sim_node.py : 기구학/패턴 기반 제어 (Kinematic Pattern Control)

  * 정해진 발 궤적을 반복 수행

---

### 핵심 알고리즘

* lib_ZMPctrl.py : Preview Control (LQR + Riccati 기반 최적제어)
* g1_sim_node.py : Sine Wave Pattern Generator

---

### 안정성 원리

* lib_ZMPctrl.py : ZMP 기반 동적 평형 (Dynamic Balance)
* g1_sim_node.py : 고강성 Position Control 기반 정적 안정

---

### IK 구조

* lib_ZMPctrl.py : Numerical IK (Jacobian 기반 반복해)
* g1_sim_node.py : Analytical IK (기하학 공식 기반)

---

## 3. SpotMicroAI(4족) vs Unitree G1(2족) 구조 비교

### 3.1 구조 및 관절

SpotMicroAI:

* 4족 구조
* 다리당 3DOF
* 점 접촉(Point Contact)
* 발목 관절 없음

Unitree G1:

* 2족 휴머노이드 구조
* 다리당 6DOF
* 면 접촉(Plane Contact)
* 발목 Pitch/Roll 필수 구조

---

### 3.2 기구학 계산 구조

SpotMicroAI:

* Body IK + Leg IK 구조
* 3-Link 기하학적 IK

Unitree G1:

* 상체 수직 유지(Torso Control)
* 발바닥 수평 유지(Ankle Compensation)
* 발목 Pitch/Roll 역보정 구조

---

### 3.3 보행 패턴

SpotMicroAI:

* Trotting gait
* 대각선 지지 구조
* 정적 안정성 기반

Unitree G1:

* Bipedal walking
* 좌우 교차 지지
* CoM 이동 필수 구조

---

## 4. 핵심 개념 비교

SpotMicroAI:

* "다리를 예쁜 궤적으로 움직인다"
* 안정성 부담이 구조적으로 낮음

Unitree G1:

* "넘어지지 않도록 무게중심을 옮긴다"
* 안정성 제어가 핵심 과제

---

## 결론

본 프로젝트 구조는 로봇 보행 제어의 두 철학을 동시에 보여준다.

1. 물리 기반 제어 (Dynamics-based Control)

   * ZMP
   * LIPM
   * Preview Control
   * Optimal Control

2. 패턴 기반 제어 (Pattern-based Control)

   * Sine Wave Gait
   * Geometric IK
   * High Stiffness Control

이 비교는 **"안정적인 강아지형 로봇"(Quadruped)**과 **"본질적으로 불안정한 인간형 로봇"(Humanoid)**이 보행을 구현하기 위해 전혀 다른 접근 방식을 취해야 함을 구조적으로 보여준다.

4족 보행은 구조적 안정성 덕분에 단순한 기구학적 제어로도 구현 가능하지만, 2족 보행은 구조적으로 불안정하므로 동역학 기반 균형 제어(ZMP, CoM Dynamics, Preview Control)가 필수적이다.

따라서 `lib_ZMPctrl.py`는 연구용·이론 기반 제어 구조이며, `g1_sim_node.py`와 SpotMicroAI 코드는 구현 중심·패턴 기반 제어 구조로 분류된다.
