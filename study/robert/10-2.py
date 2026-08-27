# 10-2.py: Custom G1 Gymnasium Environment
"""
Unitree G1 휴머노이드 로봇을 Gymnasium 환경으로 래핑한 예제
Stable-Baselines3 등 기존 RL 라이브러리와 호환됩니다.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path


def get_model_path():
    """모델 경로 찾기"""
    # 현재 스크립트 위치에서 상대 경로로 찾기
    script_dir = Path(__file__).parent
    
    # 방법 1: 프로젝트 루트의 mujoco_menagerie 서브모듈
    submodule_path = script_dir.parent.parent / "mujoco_menagerie" / "unitree_g1" / "scene.xml"
    if submodule_path.exists():
        return str(submodule_path)
    
    # 방법 2: robot_descriptions 패키지 사용
    try:
        from robot_descriptions import unitree_g1_mj_description
        return unitree_g1_mj_description.MJCF_PATH
    except ImportError:
        pass
    
    raise FileNotFoundError(
        "Unitree G1 모델을 찾을 수 없습니다.\n"
        "다음 중 하나를 설치하세요:\n"
        "1. git submodule add https://github.com/google-deepmind/mujoco_menagerie.git\n"
        "2. uv add robot_descriptions"
    )


class UnitreeG1Env(gym.Env):
    """Unitree G1 휴머노이드 Gymnasium 환경"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # MuJoCo 모델 로드
        model_path = get_model_path()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 관절 수
        self.n_joints = self.model.nu  # 액추에이터 수
        
        # Action Space: 관절 토크 (-1 ~ 1)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_joints,),
            dtype=np.float32
        )
        
        # Observation Space: [qpos, qvel, IMU]
        obs_dim = self.model.nq + self.model.nv + 6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.viewer = None
        self.max_steps = 1000
        self.current_step = 0
        
        # pelvis body id 캐싱
        self._pelvis_id = None
        
    @property
    def pelvis_id(self):
        """pelvis body ID (캐싱)"""
        if self._pelvis_id is None:
            self._pelvis_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
            )
        return self._pelvis_id
        
    def _get_obs(self):
        """관측값 반환"""
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        # IMU 데이터 (가속도, 자이로)
        imu = np.zeros(6)
        if self.model.nsensor >= 6:
            imu = self.data.sensordata[:6].copy()
        
        return np.concatenate([qpos, qvel, imu]).astype(np.float32)
    
    def _get_info(self):
        """추가 정보"""
        pelvis_height = self.data.xpos[self.pelvis_id, 2]
        
        return {
            "step": self.current_step,
            "pelvis_height": pelvis_height,
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        
        # 초기 keyframe 적용 (서있는 자세)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        self.current_step = 0
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        # 토크 스케일링
        scaled_action = action * 50  # 최대 토크 50 Nm
        self.data.ctrl[:] = scaled_action
        
        # 물리 시뮬레이션 (여러 서브스텝)
        n_substeps = 4
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)
        
        self.current_step += 1
        
        obs = self._get_obs()
        info = self._get_info()
        
        # 보상 계산
        reward = self._compute_reward(action)
        
        # 종료 조건
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, action):
        """보상 함수"""
        reward = 0.0
        
        # 1. 생존 보상
        reward += 1.0
        
        # 2. 전진 속도 보상
        forward_vel = self.data.qvel[0]  # x축 속도
        reward += 2.0 * forward_vel
        
        # 3. 높이 유지 보상
        height = self.data.xpos[self.pelvis_id, 2]
        target_height = 0.75  # G1 서있는 높이
        height_error = abs(height - target_height)
        reward += np.exp(-5 * height_error)
        
        # 4. 에너지 효율 패널티
        energy = np.sum(np.abs(action * self.data.qvel[:len(action)]))
        reward -= 0.001 * energy
        
        return reward
    
    def _is_terminated(self):
        """종료 조건"""
        height = self.data.xpos[self.pelvis_id, 2]
        
        # 넘어지면 종료
        if height < 0.3:
            return True
        
        return False
    
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(
                    self.model, self.data
                )
            self.viewer.sync()
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# 테스트
if __name__ == "__main__":
    print("=" * 60)
    print("Unitree G1 Gymnasium 환경 테스트")
    print("=" * 60)
    
    env = UnitreeG1Env(render_mode="human")
    obs, info = env.reset()
    
    print(f"\nObservation shape: {obs.shape}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Number of actuators: {env.n_joints}")
    
    print("\n시뮬레이션 시작 (500 스텝)...")
    print("창을 닫으면 종료됩니다.\n")
    
    total_reward = 0
    import time
    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        
        if terminated or truncated:
            print(f"에피소드 종료! Step: {step}, Total Reward: {total_reward:.2f}")
            print("  → 5초간 대기 중...")
            time.sleep(5)  # 에피소드 사이 2초 pause
            obs, info = env.reset()
            total_reward = 0
    
    print(f"\n최종 누적 보상: {total_reward:.2f}")
    env.close()
    print("테스트 완료!")
