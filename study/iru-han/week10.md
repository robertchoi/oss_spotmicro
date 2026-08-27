```
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


# ============================================================================
# 1. SpotMicroEnv: Gymnasium 표준 인터페이스 환경 정의 (9-1.py 기반)
# ============================================================================
class SpotMicroEnv(gym.Env):
    """
    MuJoCo 시뮬레이터와 강화학습 알고리즘 사이를 연결하는 '환경' 클래스입니다.
    로봇의 센서 데이터를 에이전트에게 전달하고, 에이전트의 명령을 물리 엔진에 적용합니다.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        # 시뮬레이션의 물리 정보가 담긴 XML 파일을 로드합니다.
        xml_path = "scene.xml"
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"{xml_path}를 찾을 수 없습니다.")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # [Action Space] 에이전트가 제어할 12개 관절의 출력 범위를 -1 ~ 1로 정의합니다.
        self.n_joints = 12
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32)

        # [Observation Space] 에이전트가 관찰할 상태(State)를 30차원으로 정의합니다.
        # 관절 위치(12) + 관절 속도(12) + IMU 가속도(3) + IMU 각속도(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32)

        self.render_mode = render_mode
        self.viewer = None
        self.max_steps = 1000  # 한 에피소드당 최대 시뮬레이션 스텝 수 (약 20초)
        self.current_step = 0
        self.reward_shaper = RewardShaper()  # 보상 계산 엔진 초기화

    def _get_obs(self):
        """로봇의 현재 상태를 수집하여 신경망의 입력값으로 변환합니다."""
        # qpos[7:19]: 루트 바디의 위치/방향을 제외한 12개 관절의 각도
        # qvel[6:18]: 루트 바디의 속도를 제외한 12개 관절의 회전 속도
        qpos = self.data.qpos[7:19].copy()
        qvel = self.data.qvel[6:18].copy()

        # IMU 데이터: 몸체의 선가속도와 자이로(각속도) 데이터를 가져옵니다.
        imu_acc = self.data.qacc[:3].copy()
        imu_gyro = self.data.qvel[3:6].copy()

        return np.concatenate([qpos, qvel, imu_acc, imu_gyro]).astype(np.float32)

    def reset(self, seed=None, options=None):
        """에피소드가 끝났을 때 환경을 초기 상태로 되돌립니다."""
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # [초기 자세] 무릎을 살짝 구부린 안정적인 자세(Neutral Pose)로 시작합니다.
        # Hip=0.0, Upper=0.4, Lower=-0.8 (라디안)
        initial_pose = np.array([0.0, 0.4, -0.8] * 4)

        # 학습의 다양성을 위해 초기 자세에 아주 미세한 무작위 노이즈를 섞습니다.
        noise = np.random.uniform(-0.02, 0.02, size=12)
        self.data.qpos[7:19] = initial_pose + noise

        self.current_step = 0
        mujoco.mj_forward(self.model, self.data)  # 변경된 물리 상태 확정
        return self._get_obs(), {}

    def step(self, action):
        """에이전트의 행동을 물리 엔진에 적용하고 결과를 반환합니다."""
        # [Action Scaling] 에이전트의 출력(-1~1)을 '구부린 자세' 기준 주변으로 매핑합니다.
        # action_offset 주위로 ±0.4 라디안 만큼 움직일 수 있게 제어 범위를 제한합니다.
        action_offset = np.array([0.0, 0.4, -0.8] * 4)
        scaled_action = action_offset + (action * 0.4)
        self.data.ctrl[:] = scaled_action

        # [Frame Skip] 물리 시뮬레이션을 5번 반복하여 한 번의 행동이 지속되게 합니다.
        # 이는 학습 안정성을 높이고 연산량을 줄이는 효과가 있습니다.
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        self.current_step += 1
        obs = self._get_obs()

        # 현재 상태를 바탕으로 행동에 대한 보상(Reward)을 계산합니다.
        reward, _ = self.reward_shaper.compute_reward(self, action)

        # [종료 조건 (Termination)]
        # 1. 몸체 높이가 0.12m 이하로 떨어짐 (넘어짐 판단)
        # 2. 로봇이 뒤로 5cm 이상 밀려남 (후진 차단)
        x_pos = self.data.qpos[0]
        backwards_cutoff = x_pos < -0.05
        terminated = (self.data.qpos[2] < 0.12) or backwards_cutoff

        # [중단 조건 (Truncation)] 최대 스텝 도달 시 종료
        truncated = self.current_step >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {}

    def render(self):
        """시뮬레이션 화면을 갱신합니다."""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()


# ============================================================================
# 2. RewardShaper: 보상 함수 설계 (9-3.py 스타일)
# ============================================================================
class RewardShaper:
    """
    에이전트가 어떤 행동을 해야 하는지 알려주는 가이드라인(Reward Shaping)입니다.
    """

    def __init__(self):
        self.prev_action = np.zeros(12)
        self.config = {
            "alive_bonus": 1.0,  # 살아있기만 해도 주는 기본 보너스
            "forward_vel_weight": 8.0,  # 전진 속도에 대한 가중치 (핵심 목표)
            "height_weight": 2.5,  # 일정한 높이 유지에 대한 보상
            "target_height": 0.18,  # 무릎을 굽혔을 때의 목표 지상고
            "gait_weight": 1.5  # Trot 보행 패턴 유도 가중치
        }

    def compute_reward(self, env, action):
        rewards = {}
        vx = env.data.qvel[0]  # 로봇의 x축 전진 속도

        # [전진 보상] 앞으로 가면 큰 보상, 뒤로 가면 매우 큰 페널티를 줍니다.
        if vx > 0.01:
            rewards["forward"] = self.config["forward_vel_weight"] * vx
        elif vx < -0.01:
            rewards["forward"] = 30.0 * vx  # 비대칭 페널티 (후진 억제)
        else:
            rewards["forward"] = -0.2  # 정지 상태에 대한 미세한 페널티

        # [높이 보상] 목표 높이를 유지할수록 지수 함수적으로 큰 보상을 줍니다.
        h_err = abs(env.data.qpos[2] - self.config["target_height"])
        rewards["height"] = self.config["height_weight"] * np.exp(-10 * h_err)

        # [Trot Gait 보상] 대각선 발들의 접촉을 유도하여 4족 보행의 특성을 배웁니다.
        # FL-RR 쌍과 FR-RL 쌍이 번갈아 가며 지면을 디딜 때 보상을 부여합니다.
        contacts = self._get_foot_contacts(env)
        diag1 = contacts[0] or contacts[3]  # 앞왼발 혹은 뒤오른발 접촉
        diag2 = contacts[1] or contacts[2]  # 앞오른발 혹은 뒤왼발 접촉

        # 한 쌍은 땅을 딛고, 다른 한 쌍은 공중에 있을 때(Trot 패턴) 보상
        if diag1 != diag2:
            rewards["gait"] = self.config["gait_weight"]
        else:
            rewards["gait"] = -0.5

        self.prev_action = action.copy()
        total = sum(rewards.values()) + self.config["alive_bonus"]
        return total, rewards

    def _get_foot_contacts(self, env):
        """물리 엔진의 충돌 데이터를 분석하여 각 발의 접촉 여부를 판단합니다."""
        contacts = [False] * 4
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        for i in range(env.data.ncon):
            contact = env.data.contact[i]
            geom1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            for idx, name in enumerate(foot_names):
                if name in str(geom1) or name in str(geom2):
                    contacts[idx] = True
        return contacts


# ============================================================================
# 3. 학습 파이프라인 (9-2.py 및 9-3.py 통합)
# ============================================================================
def main():
    # [환경 래핑] 학습 통계 기록을 위해 Monitor로 감싸줍니다.
    def make_env():
        env = SpotMicroEnv(render_mode="human")
        return Monitor(env)

    # [정규화] VecNormalize를 통해 관측값과 보상을 평균 0, 표준편차 1 수준으로 정규화합니다.
    # 이는 신경망이 다양한 크기의 데이터를 안정적으로 학습하게 돕습니다.
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # [PPO 모델 설정] Proximal Policy Optimization 알고리즘을 사용합니다.
    model = PPO(
        "MlpPolicy",  # 다층 퍼셉트론(MLP) 신경망 정책
        env,
        verbose=1,
        learning_rate=3e-4,  # 학습률 (알고리즘의 보폭)
        n_steps=4096,  # 한 번 업데이트를 위해 수집하는 데이터 양 (시퀀스 학습 강화)
        batch_size=128,  # 신경망을 학습시킬 때 쪼개는 단위
        n_epochs=10,  # 같은 데이터를 반복 학습하는 횟수
        gamma=0.995,  # 할인율 (미래 보상을 얼마나 중시할지)
        tensorboard_log="./logs/"  # 실시간 학습 곡선 확인용 로그 경로
    )

    print("--- 100만 스텝 학습 시작 (Trot Gait 유도 및 정규화 적용) ---")
    try:
        # 실제 학습을 실행합니다. 100만 스텝은 상당한 시간이 소요됩니다.
        model.learn(total_timesteps=1000000)
    except KeyboardInterrupt:
        print("학습 중단 - 현재까지의 가중치를 저장합니다.")

    # 최종 가중치와 정규화 파라미터를 저장합니다.
    model.save("ppo_spot_gait_final")
    env.save("vec_normalize.pkl")
    print("학습 완료 및 모델/정규화 파일 저장 성공.")


if __name__ == "__main__":
    main()
```
# 🐕 SpotMicro 강화학습 보행 제어 프로젝트
이 프로젝트는 MuJoCo 물리 엔진과 PPO(Proximal Policy Optimization) 알고리즘을 사용하여 4족 보행 로봇인 SpotMicro가 스스로 걷는 법을 배우게 만드는 강화학습 프로젝트이다.<br/>
<br/>
# 📝 프로젝트 개요
로봇이 아무런 사전 지식 없이(무작위 움직임) 시작하여, 보상(Reward) 시스템을 통해 **'앞으로 곧게 걷는 보행 패턴'**을 스스로 찾아가는 과정을 구현했다.<br/>
<br/>
# 🛠 주요 코드 설명 (핵심 기능)
## 1. 로봇의 '눈'과 '근육' 설정 (SpotMicroEnv)
관측(Observation): 로봇의 관절 각도, 속도, 그리고 중심을 잡기 위한 IMU(가속도/각속도) 센서 등 총 30개의 데이터를 실시간으로 파악한다.<br/>

행동(Action): 12개의 관절 모터에 회전 명령을 내린다.

초기 자세: 로봇이 가장 안정적으로 서 있을 수 있도록 무릎을 살짝 구부린 자세에서 학습을 시작한다.<br/>

## 2. 똑똑한 보상 시스템 (RewardShaper)
단순히 "가라"고 하는 것이 아니라, 효율적인 보행을 위해 다음 조건들을 체크한다.<br/>

전진 보상: 앞으로 나아가면 점수를 주고, 뒤로 가면 큰 벌점을 주어 전진을 유도한다.<br/>

높이 유지: 몸체가 일정한 높이를 유지하며 안정적으로 걷게 한다.<br/>

Trot 보행 유도: 대각선 방향의 발들이 교차하며 땅을 딛을 때 보너스 점수를 주어 실제 로봇과 유사한 걸음걸이를 가르친다.<br/>

## 3. 학습 엔진 (PPO & Pipeline)
PPO 알고리즘: 가장 안정적이고 성능이 뛰어난 강화학습 알고리즘 중 하나를 사용했다.<br/>

데이터 정규화: 수많은 센서 데이터를 인공지능이 이해하기 쉬운 크기로 변환하여 학습 효율을 높였다.<br/>

# ⚠️ 현재 모델의 한계 및 개선 과제<br/>


https://github.com/user-attachments/assets/1011727e-808a-4c60-832c-45cdcec7e37b <br/>



https://github.com/user-attachments/assets/3884faee-9ce3-448f-b450-7ae6c5ea795f <br/>



학습을 진행하면서 다음과 같은 두 가지 주요 문제점을 발견했으며, 이를 해결하는 것이 다음 목표이다.<br/>

## 1. 후진 보행 문제 (Backstepping)<br/>
현상: 로봇이 앞으로 가는 것보다 뒤로 걷는 것이 넘어지지 않고 안정적이라고 판단하여 자꾸 뒤로 가려는 경향이 있음.<br/>

원인: 초기 학습 단계에서 앞으로 가려다 넘어지는 리스크보다, 제자리나 뒤에서 버티는 것이 생존 보너스를 받기에 유리하기 때문.<br/>

해결 방안: 뒤로 가는 속도에 대한 페널티(Penalty) 가중치를 대폭 높이고, 일정 거리 이상 뒤로 가면 강제로 에피소드를 종료시키는 로직 강화 필요.<br/>

## 2. 뒷다리 고정 문제 (Frozen Hind Legs)<br/>
현상: 앞다리만 휘저으며 나아가고 뒷다리는 거의 움직이지 않고 끌려가는 형태가 나타남.<br/>

원인: 12개 관절을 모두 조화롭게 움직이는 조합을 찾기 전, 앞다리만 써서 운 좋게 전진 보상을 얻은 경우 그 행동에만 안주하게 됨(Local Optima).<br/>

해결 방안: 각 다리의 보행 주기(Cycle)를 강제하는 보상을 추가하거나, 뒷다리의 움직임이 없을 때 페널티를 주는 방식 도입 필요.<br/>

## 3. 뒷구르기 및 뒤집힘 문제 (Back-flipping & Overturning)현상
로봇이 다리를 휘저으며 앞으로 전진하는 대신, 상체를 과도하게 들어 올리다가 뒤로 넘어가 버리는 현상.<br/>

원인:보상 설계의 허점: 전진 속도에 대한 보상이 매우 높게 설정되어 있을 때, 로봇이 단순히 다리를 밀어내며 상체를 드는 동작이 순간적으로 '빠른 전진'으로 오인되어 보상을 받게 됨.<br/>

안정성 페널티 부족: 몸체의 기울기(Pitch)가 과도하게 커지는 것에 대한 제약이 없어서 로봇이 균형을 무너뜨리며 보상을 극대화하려는 '보상 해킹(Reward Hacking)'이 발생함.<br/>

해결 방안:자세 보상 추가: 몸체의 기울기($roll, pitch$)가 수평을 유지할수록 보상을 주고, 일정 각도 이상 기울어지면 에피소드를 즉시 종료함.<br/>

각속도 페널티: 몸체가 급격하게 회전하는 동작(IMU 데이터의 각속도 값)에 대해 페널티를 주어 안정적인 움직임을 유도함.<br/>
