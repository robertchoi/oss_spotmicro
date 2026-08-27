import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer  # 뷰어 모듈 추가
import numpy as np
from stable_baselines3 import PPO
import os


class SpotMicroEnv(gym.Env):
    """
    숙제 1: Gymnasium 환경 래핑 (뷰어 기능 추가됨)
    """

    # 초기 설정 (환경 세팅)
    def __init__(self, xml_path='scene.xml', render_mode=None):
        super(SpotMicroEnv, self).__init__()

        # 1. 모델 로드
        if os.path.exists(xml_path):
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        else:
            print(f"Warning: {xml_path} not found. Loading spot_micro_mujoco.xml")
            self.model = mujoco.MjModel.from_xml_path('spot_micro_mujoco.xml')

        self.data = mujoco.MjData(self.model)

        # 2. Action & Observation Space 정의
        # Action Space (행동 공간) - AI가 할 수 있는 행동을 정의합니다, 여기서는 12개의 모터에 -1.0(뒤로 회전) ~ 1.0(앞으로 회전) 사이의 신호를 보내는 것입니다
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        # 관찰 공간: 12(위치) + 12(속도) + 4(쿼터니언) + 3(자이로) = 31개로 확장하는 것이 좋으나
        # 기존 코드 호환성을 위해 28개 유지 (위치12 + 속도12 + 쿼터니언4)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        # [수정됨] 렌더링 모드 설정
        self.render_mode = render_mode
        self.viewer = None

        # 'human' 모드일 경우 뷰어 실행
        if self.render_mode == 'human':
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 랜덤 초기화
        self.data.qpos[:] = self.init_qpos + np.random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        self.data.qvel[:] = self.init_qvel + np.random.uniform(low=-0.01, high=0.01, size=self.model.nv)

        mujoco.mj_step(self.model, self.data)

        # [수정됨] 리셋 시에도 화면 갱신
        if self.render_mode == 'human' and self.viewer.is_running():
            self.viewer.sync()

        return self._get_obs(), {}

    def step(self, action):
        # 1. Action 적용
        ctrl_range = 0.5
        self.data.ctrl[:] = action * ctrl_range

        # 2. 물리 시뮬레이션 (Frame Skip)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        # [수정됨] 화면 갱신 (여기가 핵심!)
        if self.render_mode == 'human' and self.viewer.is_running():
            self.viewer.sync()

        # 3. 관찰, 보상, 종료 확인
        observation = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_done()
        truncated = False

        return observation, reward, terminated, truncated, {}

    def _get_obs(self):
        qpos = self.data.qpos.flat[7:]
        qvel = self.data.qvel.flat[6:]
        quat = self.data.qpos.flat[3:7]
        return np.concatenate([qpos, qvel, quat]).astype(np.float32)

    def _compute_reward(self):
        # (1) 전진 보상
        forward_reward = 5.0 * self.data.qvel[0]  # 보상을 좀 더 키움
        # (2) 생존 보상
        healthy_reward = 1.0
        # (3) 에너지 페널티
        ctrl_cost = 0.01 * np.square(self.data.ctrl).sum()
        # (4) 안정성 페널티
        stability_cost = 0.05 * np.square(self.data.qacc[2])

        return forward_reward + healthy_reward - ctrl_cost - stability_cost

    def _is_done(self):
        height = self.data.qpos[2]
        if height < 0.25:
            return True
        return False

    def close(self):
        # 뷰어 종료 처리
        if self.viewer is not None:
            self.viewer.close()


# ==========================================
# 실행 부분
# ==========================================
def main():
    # [수정됨] render_mode='human'을 넣어주면 화면이 뜹니다!
    # 주의: 화면을 띄우면 학습 속도가 훨씬 느려집니다.
    env = SpotMicroEnv(xml_path='scene.xml', render_mode='human')

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_spot_tensorboard/")

    print("-------------- 학습 시작 (화면이 표시됩니다) --------------")
    # 화면을 보려면 학습 단계를 줄이거나, 천천히 지켜보세요.
    model.learn(total_timesteps=50000)

    print("-------------- 학습 완료 --------------")
    model.save("ppo_spot_micro")
    env.close()


if __name__ == "__main__":
    main()
