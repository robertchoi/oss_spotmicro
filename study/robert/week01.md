---
title: Spot Mini 로봇 강화학습 10주 커리큘럼
subtitle: 과정 개요 및 주차별 로드맵
badge: SpotMicro Jetson 로보틱스
desc: 시뮬레이션 환경 구축부터 강화학습·MPC·통합 프로젝트까지
author: robert
header: Spot Mini 강화학습 10주 커리큘럼
toc: true
toc_depth: 2
---

PyBullet과 MuJoCo 두 시뮬레이터를 기반으로, 4족 보행 로봇 Spot Mini를 제어·학습시키는 10주 과정의 전체 로드맵입니다. 환경 설정에서 출발해 기구학·보행 생성·자세 안정화·MPC·강화학습을 거쳐, 마지막에는 배운 기술을 통합한 프로젝트로 마무리합니다.

## 과정 개요

- **대상**: 로봇 시뮬레이션과 강화학습 입문자
- **사용 도구**: PyBullet, MuJoCo, Gymnasium, Stable-Baselines3
- **로봇 모델**: Spot Mini (각 다리 3-DOF, 총 12개 관절)
- **진행 방식**: 주차별 개념 학습 + 시뮬레이터 실습

> 각 주차는 개념 → 핵심 내용 → 실습 순서로 구성됩니다. 두 시뮬레이터(PyBullet·MuJoCo)를 병행해 장단점을 비교하는 것이 이 과정의 특징입니다.

---

## 1주차: 환경 설정 & Spot Mini 모델 이해

PyBullet과 MuJoCo 설치 및 비교 실습을 시작합니다. Spot Mini URDF 모델을 두 시뮬레이터에 로드하고 차이점을 파악합니다.

**주요 학습 내용**

- PyBullet: spotmicro 오픈소스 URDF 로드
- MuJoCo: URDF를 MJCF로 변환 및 로드
- 관절 정보 추출(12개 관절: 각 다리당 3 DOF)

**실습**: 두 시뮬레이터에서 Spot Mini 로드 후 관절 구조 분석, GUI로 수동 제어

**참고 저장소**

- `OpenQuadruped/spot_mini_mini`
- `moribots/spot_mini_mini` (PyBullet용)

---

## 2주차: PyBullet 심화 — 기본 제어

PyBullet의 `stepSimulation`, `setJointMotorControl2` API를 마스터합니다. Position / Velocity / Torque 제어 모드를 실험하고 PD 게인 튜닝을 배웁니다.

**핵심 API**

- `p.setJointMotorControl2()` — 3가지 제어 모드
- `p.getJointState()` — 관절 상태 읽기
- `p.getLinkState()` — 발끝 위치 계산

**실습**: Spot Mini를 서 있는 자세(home position)로 제어, 중력 보상 토크 계산

---

## 3주차: MuJoCo 심화 — 물리 엔진 활용

MuJoCo의 장점인 빠른 시뮬레이션과 정확한 접촉 모델링을 활용합니다. `mujoco-py` 또는 새로운 `mujoco` Python 바인딩을 사용합니다.

**핵심 내용**

- `mjModel`, `mjData` 구조 이해
- actuator 설정(position, velocity, motor)
- site를 이용한 발끝 위치 추적

**실습**: MuJoCo에서 Spot Mini 기본 제어, PyBullet과 성능 비교

---

## 4주차: 역운동학(IK) 구현

Spot Mini의 다리 기구학을 분석하고 IK를 구현합니다. 각 다리는 shoulder-elbow-wrist 구조입니다.

**핵심 내용** (각 다리: Abduction-Hip-Knee, 3-DOF)

- Geometric IK: 2D 평면에서 해석적 해
- Numerical IK: `scipy.optimize` 활용
- Jacobian 기반 IK (고급)

**실습**: 4개 발끝을 원하는 위치에 배치하는 IK 솔버, PyBullet과 MuJoCo에서 동일 궤적 재현

---

## 5주차: CPG 기반 보행 생성

Trot gait를 위한 CPG(Central Pattern Generator)를 구현합니다. 사인파 기반 궤적으로 swing/stance phase를 생성합니다.

**Trot gait 패턴**

- Leg pair 1: FL + RR (대각선)
- Leg pair 2: FR + RL (대각선)
- Phase offset: 0.5 (50% 주기 차이)

**실습**: PyBullet에서 Spot Mini Trot 보행 구현, 보행 파라미터(stride length, height, frequency) 튜닝

---

## 6주차: 자세 안정화 — IMU & PID

몸체 자세(orientation)를 제어하는 PID 컨트롤러를 구현합니다. PyBullet의 `getBasePositionAndOrientation`으로 IMU를 시뮬레이션합니다.

**Body pose controller**

- Roll/Pitch 보상: 발 높이 조절로 수평 유지
- Yaw 제어: 선회 시 몸체 방향
- CoM tracking: 몸체 높이 일정 유지

**실습**: 경사면/요철 지형에서 몸체 수평 유지, MuJoCo의 정밀한 접촉 센서 활용

---

## 7주차: MuJoCo MPC — MIT Cheetah 방식

MuJoCo에서 Model Predictive Control을 구현합니다. Convex MPC 라이브러리를 활용하거나 간단한 버전을 직접 구현합니다.

**MPC 프레임워크**

- qpOASES 또는 CVXPY로 QP 문제 해결
- Single Rigid Body Dynamics 모델
- Ground Reaction Force 최적화

**실습**: MuJoCo에서 MPC 기반 보행, 외란(push)에 대한 강건성 테스트

**참고**: `mit-biomimetics/Cheetah-Software` 논문 및 코드

---

## 8주차: 강화학습 — Gym 환경 구축

Spot Mini를 Gymnasium 환경으로 래핑합니다. PyBullet과 MuJoCo 각각의 환경을 만들어 비교합니다.

**Gym 환경 구현**

- Observation: joint pos/vel, body orientation, foot contacts
- Action: target joint positions or torques
- Reward: forward velocity + alive bonus − energy cost

**실습**: Stable-Baselines3 PPO로 전진 보행 학습, PyBullet(병렬화 쉬움) vs MuJoCo(속도 빠름) 비교

---

## 9주차: 고급 RL — 지형 적응

복잡한 환경(계단, 장애물, 불규칙 지형)을 생성하고 curriculum learning을 적용합니다.

**환경 다양화**

- Heightfield terrain in MuJoCo
- Random obstacles in PyBullet
- Domain randomization: mass, friction, actuator 변화

**실습**: MuJoCo에서 높이맵 지형 학습, teacher-student distillation으로 센서 정보 압축

**참고**: `leggedrobotics/legged_gym` (Isaac Gym 기반이지만 아이디어 참고)

---

## 10주차: 최종 프로젝트 — 통합 시스템

배운 모든 기술을 통합한 완성형 프로젝트를 수행합니다.

**프로젝트 옵션**

1. **자율 내비게이션**: ROS2와 연동, SLAM + 경로계획 + 보행제어 통합
2. **계층적 제어**: High-level RL policy + Low-level MPC 결합
3. **Sim-to-Real**: 시뮬레이션 정책을 실제 로봇으로 전이 준비(domain randomization)

**실습**: PyBullet에서 빠른 프로토타이핑 → MuJoCo에서 정밀 검증 → 결과 비교 분석

---

## 학습 로드맵 요약

| 주차 | 주제 | 핵심 기술 |
|------|------|-----------|
| 1주차 | 환경 설정 | PyBullet, MuJoCo, URDF |
| 2주차 | PyBullet 제어 | Joint Motor Control, PD 게인 |
| 3주차 | MuJoCo 물리엔진 | Contact Modeling, Actuator |
| 4주차 | 역운동학 | IK Solver, Kinematics |
| 5주차 | 보행 생성 | CPG, Trot Gait |
| 6주차 | 자세 안정화 | IMU, PID Controller |
| 7주차 | MPC | Convex Optimization, QP |
| 8주차 | 강화학습 기초 | Gym Environment, PPO |
| 9주차 | 고급 RL | Domain Randomization, Curriculum Learning |
| 10주차 | 통합 프로젝트 | ROS2, Sim-to-Real |
