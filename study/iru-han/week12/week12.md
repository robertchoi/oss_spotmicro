오픈소스 활용: https://github.com/unitreerobotics/unitree_rl_gym?tab=readme-ov-file<br/>

# 코드
## 학습
unitree_rl_gym.ipynb 파일<br/>

## 내보내기
윈도우 wsl2 환경에서 실행하기 위한 export_lstm.py 코드<br/>
강화학습을 통해 얻은 결과물은 보통 학습에 사용된 복잡한 라이브러리들이 있어야만 작동한다. 이 코드는 그런 불필요한 군더더기를 제거하고, 로봇의 뇌(Actor) 부분만 쏙 골라내어 MuJoCo 같은 다른 환경에서도 독립적으로 빠르게 돌아갈 수 있는 TorchScript(.pt) 형식으로 다시 만드는 역할을 한다.<br/>
```
import torch
import torch.nn as nn
import os

# --- 설정 (사용자 환경에 맞춰 확인 필요) ---
EXPORT_DIR = "Feb23_02-29-11_" # 학습 로그가 저장된 폴더명
CHECKPOINT_NUM = 7600           # 변환할 체크포인트 번호 (model_7600.pt)
# ----------------------------

# 배포용 LSTM 정책 클래스 정의
class LSTMPolicy(nn.Module):
    def __init__(self, obs_dim=47, lstm_hidden=64, actor_hidden=32, act_dim=12):
        """
        G1 모델 설정(G1RoughCfgPPO)에 맞춘 네트워크 구조 생성
        - obs_dim: 47 (입력 관측치 개수)
        - lstm_hidden: 64 (LSTM 은닉층 크기)
        - actor_hidden: 32 (MLP 은닉층 크기)
        - act_dim: 12 (출력 행동/관절 개수)
        """
        super().__init__()
        # 시계열 데이터를 처리하기 위한 LSTM 층
        self.lstm = nn.LSTM(obs_dim, lstm_hidden, num_layers=1, batch_first=True)
        
        # LSTM 출력을 최종 관절 명령으로 변환하는 MLP 층
        self.actor = nn.Sequential(
            nn.Linear(lstm_hidden, actor_hidden),
            nn.ELU(), # 학습 설정과 동일한 activation 사용
            nn.Linear(actor_hidden, act_dim)
        )
        
        # [중요] LSTM은 이전 상태를 기억해야 하므로 Hidden/Cell State를 버퍼로 등록
        # register_buffer를 사용하면 모델 저장 시 이 상태 값들도 함께 관리됨
        self.register_buffer("hidden_state", torch.zeros(1, 1, lstm_hidden))
        self.register_buffer("cell_state", torch.zeros(1, 1, lstm_hidden))

    def forward(self, x):
        """
        실시간 추론을 위한 순전파 함수
        x: [batch_size, obs_dim] 형태의 입력
        """
        # LSTM 입력을 위해 3차원 [batch, seq_len, features]로 확장 (unsqueeze)
        # 이전 스텝의 상태(hidden_state, cell_state)를 입력으로 함께 전달
        out, (h, c) = self.lstm(x.unsqueeze(1), (self.hidden_state, self.cell_state))
        
        # [메모리 업데이트] 이번 스텝에서 계산된 상태를 버퍼에 다시 저장 (다음 스텝에서 사용)
        self.hidden_state[:] = h
        self.cell_state[:] = c
        
        # LSTM의 출력을 다시 2차원으로 줄여 Actor MLP에 전달하여 최종 행동 계산
        return self.actor(out.squeeze(1))

    @torch.jit.export
    def reset_memory(self):
        """시뮬레이션 리셋 시 LSTM의 기억을 초기화하는 함수"""
        self.hidden_state.fill_(0.)
        self.cell_state.fill_(0.)

def export():
    # 1. 파일 경로 설정
    model_path = f"logs/g1/{EXPORT_DIR}/model_{CHECKPOINT_NUM}.pt"
    save_path = f"logs/g1/{EXPORT_DIR}/policy_1.pt"
    
    # 2. 학습된 가중치(Checkpoint) 불러오기
    print(f"불러오는 중: {model_path}")
    loaded_dict = torch.load(model_path, map_location='cpu')
    
    # 3. 배포용 구조를 가진 빈 모델 생성
    policy = LSTMPolicy()
    
    # 4. [핵심] 가중치 이름 맵핑 (Mapping)
    # rsl_rl에서 사용하는 변수명(memory_a.rnn...)을 우리가 정의한 구조(lstm...)로 매칭
    sd = loaded_dict['model_state_dict']
    new_sd = {
        # LSTM 층 가중치 연결
        "lstm.weight_ih_l0": sd["memory_a.rnn.weight_ih_l0"],
        "lstm.weight_hh_l0": sd["memory_a.rnn.weight_hh_l0"],
        "lstm.bias_ih_l0": sd["memory_a.rnn.bias_ih_l0"],
        "lstm.bias_hh_l0": sd["memory_a.rnn.bias_hh_l0"],
        
        # Actor(MLP) 층 가중치 연결
        "actor.0.weight": sd["actor.0.weight"],
        "actor.0.bias": sd["actor.0.bias"],
        "actor.2.weight": sd["actor.2.weight"],
        "actor.2.bias": sd["actor.2.bias"],
    }
    
    # 5. 매핑된 가중치를 모델에 이식
    policy.load_state_dict(new_sd, strict=False)
    policy.eval() # 추론 모드로 설정 (드롭아웃 등 비활성화)
    
    # 6. TorchScript 형식으로 컴파일 및 저장
    # 이 과정을 거쳐야 Python이 없는 환경이나 다른 시뮬레이터(MuJoCo)에서 빠르게 실행 가능
    scripted_model = torch.jit.script(policy)
    scripted_model.save(save_path)
    print(f"성공! LSTM 모델 변환 완료: {save_path}")

if __name__ == "__main__":
    export()
```
왜 단순 신경망이 아닌 LSTM인가?<br/>
코드에 적힌 LSTMPolicy는 일반적인 인공지능보다 한 단계 더 진화한 형태다.<br/>

기억하는 뇌: 일반 신경망(MLP)은 '현재'만 보지만, LSTM은 과거의 보행 상태를 기억한다.<br/>

연속성 보장: 이족보행 로봇은 한 걸음을 떼는 순간의 균형이 다음 걸음에 영향을 준다. 이 코드는 hidden_state와 cell_state라는 버퍼를 만들어 로봇이 **"내가 아까 어떻게 발을 뗐지?"**를 기억하며 걸을 수 있게 한다.<br/>
<br/>
코드의 작동 순서<br/>
구조 설계: 47개의 관측치를 받아 64개의 기억을 거친 뒤, 12개의 관절 토크를 계산하는 구조를 만든다.<br/>

가중치 이식: 학습용 데이터 파일(model_7600.pt)에서 제어에 꼭 필요한 근육(가중치) 정보만 뽑아내어 우리가 만든 새 구조에 입힌다.<br/>

메모리 리셋 기능: 로봇이 넘어지거나 시뮬레이션을 다시 시작할 때, 과거의 기억을 깨끗이 지우는 reset_memory 기능을 추가했다.<br/>

최종 추출: 파이썬이 없는 환경에서도 돌아가도록 컴파일하여 policy_1.pt라는 이름으로 저장한다.<br/>

## 시뮬레이션
deploy_mujoco.py

```
import time
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
from pynput import keyboard

# --- [전역 변수] 실시간 키보드 입력 속도값 ---
# 사용자가 키를 누를 때 업데이트되어 모델의 입력값(cmd)으로 들어갑니다.
vx, vy, wz = 0.0, 0.0, 0.0

def on_press(key):
    """키보드를 눌렀을 때 실행되는 콜백 함수"""
    global vx, vy, wz
    try:
        if key.char == 'w': vx = 0.5    # 전진 속도 (m/s)
        elif key.char == 's': vx = -0.3 # 후진 속도
        elif key.char == 'a': vy = 0.2  # 왼쪽 이동 속도
        elif key.char == 'd': vy = -0.2 # 오른쪽 이동 속도
        elif key.char == 'q': wz = 0.4  # 왼쪽 회전 속도 (rad/s)
        elif key.char == 'e': wz = -0.4 # 오른쪽 회전 속도
    except AttributeError:
        pass # 특수키 입력 시 무시

def on_release(key):
    """키보드에서 손을 뗐을 때 실행되는 콜백 함수 (정지)"""
    global vx, vy, wz
    try:
        if key.char in ['w', 's']: vx = 0.0
        elif key.char in ['a', 'd']: vy = 0.0
        elif key.char in ['q', 'e']: wz = 0.0
    except AttributeError:
        pass

def get_gravity_orientation(quaternion):
    """
    로봇의 쿼터니언(자세) 데이터를 이용해 중력 방향 벡터를 계산합니다.
    로봇이 얼마나 기울었는지 인공지능이 판단하게 돕는 핵심 데이터입니다.
    """
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    # 쿼터니언 회전 행렬의 마지막 행을 추출하여 투영된 중력 방향 계산
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """
    PD 제어기: 인공지능이 준 목표 각도를 실제 토크(힘)로 변환합니다.
    수식: 토크 = 강성 * (목표각도 - 현재각도) + 댐핑 * (목표속도 - 현재속도)
    """
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    # --- 1. 설정 및 파라미터 로드 ---
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config 파일 이름")
    args = parser.parse_args()
    
    # YAML 설정 파일 읽기
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{args.config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # 경로 및 물리 파라미터 추출
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"] # 제어 주기 (물리 스텝 n번당 1번 추론)
        kps = np.array(config["kps"], dtype=np.float32)   # 관절 강성
        kds = np.array(config["kds"], dtype=np.float32)   # 관절 댐핑
        default_angles = np.array(config["default_angles"], dtype=np.float32) # 기본 서 있는 자세
        
        # 스케일링 인자 (센서값을 AI가 이해하기 쉬운 범위로 변환)
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        num_actions = config["num_actions"] # 12
        num_obs = config["num_obs"]         # 47

    # --- 2. 키보드 조종 시작 ---
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # --- 3. MuJoCo 모델 및 데이터 초기화 ---
    m = mujoco.MjModel.from_xml_path(xml_path) # 로봇 물리 모델 로드
    d = mujoco.MjData(m)                       # 시뮬레이션 데이터 객체 생성
    m.opt.timestep = simulation_dt             # 물리 연산 시간 간격 설정
    policy = torch.jit.load(policy_path)       # 변환된 LSTM 정책(Policy) 로드

    # 변수 초기화
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    print("Control G1 with [W, A, S, D, Q, E] keys!")

    # --- 4. 메인 시뮬레이션 루프 ---
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # 현재 키보드 입력값으로 명령(cmd) 업데이트
            cmd = np.array([vx, vy, wz], dtype=np.float32)

            # [저수준 제어] PD 제어를 통해 관절에 가해질 토크 계산 및 적용
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            
            # 물리 엔진 한 스텝 진행
            mujoco.mj_step(m, d)

            counter += 1
            # [고수준 제어] 특정 주기(decimation)마다 AI가 상황을 보고 다음 동작 결정
            if counter % control_decimation == 0:
                # 관측치(Observation) 생성 - 로봇의 감각 정보를 수집
                qj = (d.qpos[7:] - default_angles) * dof_pos_scale # 관절 위치 오차
                dqj = d.qvel[6:] * dof_vel_scale                    # 관절 속도
                gravity_orientation = get_gravity_orientation(d.qpos[3:7]) # 몸체 기울기
                omega = d.qvel[3:6] * ang_vel_scale                 # 몸체 각속도

                # 보행 타이밍(Phase) 계산 - 로봇이 리드미컬하게 걷도록 돕는 신호
                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                # AI 입력 데이터(obs) 조립 (총 47개)
                obs[:3] = omega                                     # [0~2] 각속도
                obs[3:6] = gravity_orientation                      # [3~5] 중력방향
                obs[6:9] = cmd * cmd_scale                          # [6~8] 사용자 명령속도
                obs[9 : 9 + num_actions] = qj                       # [9~20] 관절 위치
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj    # [21~32] 관절 속도
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action # [33~44] 이전 행동
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase]) # [45~46] 페이즈
                
                # 인공지능 추론 (Inference)
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # AI가 다음 관절 각도 변화량을 계산함
                action = policy(obs_tensor).detach().numpy().squeeze()
                
                # AI의 출력을 실제 목표 각도로 변환
                target_dof_pos = action * action_scale + default_angles

            # 시뮬레이터 화면 업데이트
            viewer.sync()

            # 실시간 속도를 맞추기 위한 대기 (Real-time sync)
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
```

## 1. 프로젝트 개요 및 환경 구성
본 프로젝트는 Unitree G1 Humanoid 로봇이 복잡한 물리 환경에서 안정적으로 보행할 수 있도록 강화학습(Reinforcement Learning) 환경을 구축하는 것을 목표로 한다. 학습은 NVIDIA의 Isaac Gym을 활용하여 수천 대의 로봇을 병렬로 학습시켜 효율을 높였으며, 최종 결과물은 물리적 신뢰도가 높은 MuJoCo 시뮬레이터에서 검증(Sim-to-Sim)하였다.

---

## 2. 제어 모델 아키텍처: LSTM과 MLP의 비교
본 시뮬레이션의 핵심 지능인 정책(Policy) 모델은 LSTM(Long Short-Term Memory) 구조를 채택하고 있다. 이는 로봇 제어에서 다음과 같은 차별점을 가진다.

- **기억 유무의 차이**: 일반적인 MLP(Multi-Layer Perceptron)는 현재 입력된 관측치($Obs_t$)만을 보고 행동($Action_t$)을 결정한다. 반면, LSTM은 과거의 상태 정보를 '은닉 상태(Hidden State)'와 '셀 상태(Cell State)'에 저장하여 현재 판단에 반영한다.  
- **Humanoid 보행의 특수성**: 사족보행 로봇과 달리 이족보행인 G1은 무게 중심의 이동과 동역학적 변화가 매우 급격하다. 이전 스텝에서의 움직임 흐름을 기억해야만 불규칙한 지형에서도 균형을 잃지 않고 다음 발을 내디딜 수 있기에 LSTM 구조가 필수적이다.

---

## 3. 기술적 트러블슈팅: 64차원의 오해와 진실

개발 과정에서 가장 중요했던 기술적 변곡점은 모델 변환(Export) 과정에서 발생한 데이터 차원의 불일치 해결이었다.

| 비교 항목 | 초기 오해 (Error 분석 당시) | 실제 기술적 사실 (Final Fix) |
|------------|----------------------------|-------------------------------|
| 데이터의 정체 | 관측 데이터(Observation)가 64차원일 것이라 판단 | 입력 관측치는 47차원, 내부 은닉 상태가 64차원 |
| 64의 의미 | 로봇이 보고 느끼는 센서 값의 개수 | LSTM 신경망의 내부 기억 용량(Hidden Size) |
| 오류 원인 | 모델 구조를 MLP로 가정하여 입력값 크기 불일치 발생 | LSTM 레이어를 생략하고 가중치를 이식하려 했기 때문 |
| 해결 방법 | obs_dim을 64로 강제 확장 | 47차원의 입력을 받는 LSTM 배포용 Wrapper 구현 |

초기에는 에러 메시지에 나타난 64라는 숫자를 보고 관측치(Observation)가 확장된 것으로 오판하였으나, 정밀 분석 결과 이는 LSTM의 은닉층 크기임을 확인하였다. 이를 해결하기 위해 추론 시마다 64차원의 은닉 상태를 갱신하고 유지하는 별도의 추론 스크립트를 작성하여 모델 이식에 성공하였다.

---

## 4. 제어 수식 및 실시간 인터페이스

로봇의 최종 관절 제어는 인공지능이 계산한 목표 각도($q_{target}$)를 바탕으로 **PD 제어(Proportional-Derivative Control)**를 통해 수행된다.

### [핵심 수식: PD 제어기]

각 관절에 가해지는 토크 T 는 다음과 같은 물리적 상호작용을 통해 계산된다.

<img width="321" height="42" alt="image" src="https://github.com/user-attachments/assets/97910d53-7f41-48cd-9857-f9b6e2b63b96" />

<img width="546" height="64" alt="image" src="https://github.com/user-attachments/assets/370458af-3360-4760-b16f-96df5d4c0b1a" />


실시간 제어를 위해 pynput 라이브러리를 활용한 전역 키보드 리스너를 구현하였다. 이를 통해 사용자가 MuJoCo 창이나 터미널을 선택한 상태에서 W, A, S, D, Q, E 키를 입력하면, 해당 명령 속도가 실시간으로 LSTM 모델의 입력값(obs[6:9])에 반영되어 로봇의 보행 방향과 속도가 즉각적으로 변화한다.

---
