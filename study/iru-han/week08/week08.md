# SpotMicroAI: Physics Engine Benchmark Report

본 프로젝트는 4족 보행 로봇 SpotMicroAI를 활용하여 PyBullet과 MuJoCo 물리 엔진의 연산 효율성 및 보행 안정성을 정량적으로 비교 분석한 결과다.

---

## 1. 모델 변환 및 물리 파라미터 최적화 (URDF to MJCF)

기존 URDF 파일을 MuJoCo 전용 MJCF(XML) 포맷으로 변환하고 수치적 안정성을 확보하기 위해 보정하였다.

- **계층 구조 재구성**: URDF의 관절(Joint) 및 링크(Link) 구조를 MuJoCo의 일반화 좌표계 시스템에 최적화하여 재정의하였다.
- **관성 모멘트 보정**: MuJoCo의 수치적 민감도를 고려하여 모든 다리 마디의 diaginertia 수치를 $0.0005$ 수준으로 조정하여 mjMINVAL 오류 및 지면 뚫림 현상을 방지하였다.
- **접촉 모델 설정**: 발끝(Toe)의 충돌 기하학을 sphere 타입으로 지정하고, solref 및 solimp 파라미터를 통해 지면과의 물리적 상호작용(반발력, 강성)을 최적화하였다.

---

## 2. 시뮬레이션 구동 및 제어기 통합

TrottingGait 제어 알고리즘을 MuJoCo 엔진에 통합하여 실시간 보행 시뮬레이션을 구현하였다.

- **제어 파이프라인**: kinematics.py에서 산출된 역운동학 각도를 spotmicroai.py의 data.ctrl 배열에 직접 매핑하여 12개의 액추에이터를 정밀 제어한다.
- **관절 방향 보정**: 엔진별 좌표계 차이를 해결하기 위해 self.dirs 배열을 적용하여 각 관절의 회전 부호를 동기화하였다.
- **사용자 인터페이스**: mujoco.viewer를 통한 실시간 모니터링과 다중 프로세스 기반의 키보드 인터럽트로 보행 명령을 실시간 전달한다.

---

## 3. 물리 엔진 기술적 아키텍처 비교

두 엔진의 구조적 차이에 따른 특성을 비교 분석하였다.

| 항목 | PyBullet (LCP 기반) | MuJoCo (일반화 좌표계) |
|---|---|---|
| 수치 해석 방식 | 불연속적인 LCP(Linear Complementarity Problem) 솔버 사용 | 행렬 연산에 최적화된 일반화 좌표계(Generalized Coordinates) 사용 |
| 접촉 모델 | 강체 간의 불연속적 접촉 처리 | 연속적인 접촉 역학(Smooth Dynamics) 지원 |
| 설정 편의성 | 물리 수치 오차에 관대하며 안정적 구동 가능 | 수치 설정에 매우 엄격하며 정밀한 관성 튜닝 요구 |

---

## 4. 벤치마크 결과 (1,000 Steps)

동일한 제어 환경에서 측정된 엔진 성능 데이터다.

### (1) 연산 속도 (Computational Speed)

- 평균 스텝 소요 시간 (Avg_Step_ms): **MuJoCo(0.5292ms)**가 PyBullet(0.6240ms)보다 약 15.2% 더 빠른 연산 속도를 보여준다.
- FPS (Frames Per Second): MuJoCo는 **1889.63 FPS**를 기록하여 PyBullet(1602.52 FPS) 대비 우수한 연산 효율성을 입증하였다.

### (2) 보행 안정성 (Stability)

- IMU 분산 분석 (Roll/Pitch Variance): PyBullet은 약 1.4 * 10^{-4} 수준의 미세 진동이 발생한 반면, **MuJoCo는 0.000000**에 수렴하는 극도의 안정성을 기록하였다.
- 통계적 결론: MuJoCo는 물리적 진동(Jitter) 억제 능력이 탁월하며, 통계적으로 더 정밀하고 안정적인 보행 환경을 제공함을 확인하였다.

---

**종합 평가**: 연산 속도와 물리적 정밀도 모두에서 MuJoCo 엔진이 우수한 성능을 보였으며, 특히 통계적 분산 분석을 통해 MuJoCo의 보행 안정성이 압도적임을 증명하였다.
