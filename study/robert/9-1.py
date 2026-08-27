# 9-1.py: Custom Gymnasium Environment for SpotMicro
# Week 09: Gymnasium 환경 래핑 실습

import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("MuJoCo not installed. Install with: pip install mujoco")


class SpotMicroEnv(gym.Env):
    """SpotMicro 로봇을 위한 Gymnasium 환경
    
    이 환경은 MuJoCo 시뮬레이터를 사용하여 SpotMicro 로봇을 제어합니다.
    Stable-Baselines3 등 다양한 RL 라이브러리와 호환됩니다.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, model_path=None, render_mode=None):
        """
        Args:
            model_path: MJCF 모델 파일 경로 (기본: ../../urdf/spot_micro.xml)
            render_mode: 렌더링 모드 ("human" 또는 "rgb_array")
        """
        super().__init__()
        
        if not MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo is required. Install with: pip install mujoco")
        
        # 모델 경로 설정
        if model_path is None:
            model_path = "../../urdf/spot_micro.xml"
        
        # MuJoCo 모델 로드
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            print("기본 테스트 모델을 사용합니다.")
            self._create_test_model()
        
        # 관절 수 (SpotMicro: 12 joints)
        self.n_joints = min(12, self.model.nu)  # actuator 수
        
        # Action Space: 각 관절의 목표 위치 (-1 ~ 1로 정규화)
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.n_joints,), 
            dtype=np.float32
        )
        
        # Observation Space: [관절 위치, 관절 속도, IMU 데이터]
        # 관절 위치(12) + 관절 속도(12) + IMU 가속도(3) + IMU 자이로(3) = 30차원
        obs_dim = self.n_joints * 2 + 6
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.viewer = None
        
        # 에피소드 설정
        self.max_steps = 1000
        self.current_step = 0
        
        # 시뮬레이션 파라미터
        self.frame_skip = 4  # 행동 반복 횟수
        self.dt = self.model.opt.timestep * self.frame_skip
        
        print(f"SpotMicroEnv 초기화 완료")
        print(f"  - Action space: {self.action_space}")
        print(f"  - Observation space: {self.observation_space}")
        print(f"  - Timestep: {self.dt:.4f}s")
    
    def _create_test_model(self):
        """테스트용 간단한 모델 생성"""
        xml = """
        <mujoco>
            <option timestep="0.002"/>
            <worldbody>
                <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                <geom type="plane" size="10 10 0.1" rgba=".9 .9 .9 1"/>
                <body name="body" pos="0 0 0.3">
                    <joint type="free"/>
                    <geom type="box" size="0.2 0.1 0.05" rgba="0.8 0.3 0.2 1" mass="1"/>
                </body>
            </worldbody>
            <actuator>
            </actuator>
        </mujoco>
        """
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
    
    def _get_obs(self):
        """현재 관측값 생성
        
        Returns:
            numpy.ndarray: [관절위치, 관절속도, IMU데이터]
        """
        # 관절 위치 (qpos에서 관절 부분만 추출)
        if self.n_joints > 0 and len(self.data.qpos) > 7:
            joint_pos = self.data.qpos[7:7+self.n_joints].copy()
        else:
            joint_pos = np.zeros(self.n_joints)
        
        # 관절 속도
        if self.n_joints > 0 and len(self.data.qvel) > 6:
            joint_vel = self.data.qvel[6:6+self.n_joints].copy()
        else:
            joint_vel = np.zeros(self.n_joints)
        
        # IMU 데이터 (센서가 있으면 사용, 없으면 0)
        if self.model.nsensor >= 6:
            imu_acc = self.data.sensordata[:3].copy()
            imu_gyro = self.data.sensordata[3:6].copy()
        else:
            # 센서가 없으면 body의 가속도/각속도 추정
            imu_acc = np.zeros(3)
            imu_gyro = np.zeros(3)
            if len(self.data.qvel) >= 6:
                imu_acc = self.data.qacc[:3] if len(self.data.qacc) >= 3 else np.zeros(3)
                imu_gyro = self.data.qvel[3:6]
        
        # 관측값 결합
        obs = np.concatenate([
            joint_pos,
            joint_vel,
            imu_acc,
            imu_gyro
        ]).astype(np.float32)
        
        return obs
    
    def _get_info(self):
        """추가 정보 반환"""
        # Body 위치 (있으면)
        body_id = 1 if self.model.nbody > 1 else 0
        body_pos = self.data.xpos[body_id].copy()
        body_quat = self.data.xquat[body_id].copy()
        
        return {
            "step": self.current_step,
            "body_height": body_pos[2],
            "body_pos": body_pos,
            "body_quat": body_quat,
            "sim_time": self.data.time,
        }
    
    def reset(self, seed=None, options=None):
        """환경 초기화
        
        Args:
            seed: 랜덤 시드
            options: 추가 옵션
            
        Returns:
            observation: 초기 관측값
            info: 추가 정보
        """
        super().reset(seed=seed)
        
        # MuJoCo 상태 초기화
        mujoco.mj_resetData(self.model, self.data)
        
        # 약간의 랜덤 초기화 (선택사항)
        if options and options.get("random_init", False):
            noise = self.np_random.uniform(-0.01, 0.01, len(self.data.qpos))
            self.data.qpos += noise
        
        self.current_step = 0
        
        # 초기 물리 시뮬레이션 (안정화)
        mujoco.mj_forward(self.model, self.data)
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """한 스텝 진행
        
        Args:
            action: 정규화된 행동 [-1, 1]
            
        Returns:
            observation: 새로운 관측값
            reward: 보상
            terminated: 에피소드 종료 여부
            truncated: 시간 초과 여부
            info: 추가 정보
        """
        # Action 클리핑 및 스케일링
        action = np.clip(action, -1.0, 1.0)
        scaled_action = action * 0.5  # [-0.5, 0.5] 라디안
        
        # Actuator에 제어 입력 (있으면)
        if self.n_joints > 0:
            self.data.ctrl[:self.n_joints] = scaled_action[:self.n_joints]
        
        # 물리 시뮬레이션 (frame_skip만큼 반복)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        self.current_step += 1
        
        # 관측값 계산
        observation = self._get_obs()
        info = self._get_info()
        
        # 보상 계산
        reward = self._compute_reward(action, info)
        
        # 종료 조건
        terminated = self._is_terminated(info)
        truncated = self.current_step >= self.max_steps
        
        return observation, reward, terminated, truncated, info
    
    def _compute_reward(self, action, info):
        """보상 함수
        
        기본적인 보상:
        1. 생존 보상: 살아있으면 +1
        2. 높이 보상: 적정 높이 유지 시 보너스
        3. 에너지 페널티: 큰 행동에 페널티
        """
        reward = 0.0
        
        # 1. 생존 보상
        reward += 1.0
        
        # 2. 높이 유지 보상
        target_height = 0.15
        height_error = abs(info["body_height"] - target_height)
        reward += np.exp(-5 * height_error)
        
        # 3. 에너지 페널티
        action_cost = 0.01 * np.sum(action ** 2)
        reward -= action_cost
        
        # 4. 넘어지면 큰 페널티
        if info["body_height"] < 0.05:
            reward -= 10.0
        
        return reward
    
    def _is_terminated(self, info):
        """종료 조건 확인"""
        # 너무 낮으면 넘어진 것으로 판단
        if info["body_height"] < 0.03:
            return True
        
        # 너무 높으면 (튕겨나감) 종료
        if info["body_height"] > 1.0:
            return True
        
        return False
    
    def render(self):
        """렌더링"""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            # RGB 이미지 반환 (추후 구현)
            pass
    
    def close(self):
        """환경 정리"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# ============================================================================
# 메인: 환경 테스트
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SpotMicroEnv 테스트")
    print("=" * 60)
    
    # 환경 생성 (MuJoCo 설치 필요)
    try:
        env = SpotMicroEnv(render_mode="human")
    except ImportError as e:
        print(f"환경 생성 실패: {e}")
        print("\nMuJoCo 없이 더미 테스트를 실행합니다...")
        
        # 더미 테스트
        class DummyEnv:
            def __init__(self):
                self.action_space = spaces.Box(-1, 1, (12,), np.float32)
                self.observation_space = spaces.Box(-np.inf, np.inf, (30,), np.float32)
            def reset(self, seed=None):
                return np.zeros(30, dtype=np.float32), {}
            def step(self, action):
                return np.zeros(30), 1.0, False, False, {}
            def close(self):
                pass
        
        env = DummyEnv()
    
    # 환경 정보 출력
    print(f"\nObservation shape: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    
    # 에피소드 실행
    obs, info = env.reset()
    print(f"\nInitial observation: {obs[:5]}... (showing first 5)")
    print(f"Initial info: {info}")
    
    total_reward = 0
    for step in range(100):
        # 랜덤 행동
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if hasattr(env, 'render'):
            try:
                env.render()
            except:
                pass
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step+1}")
            print(f"  - Terminated: {terminated}")
            print(f"  - Truncated: {truncated}")
            print(f"  - Total reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
    
    print("\n초기 테스트 완료! 뷰어에서 시뮬레이션을 계속 확인하세요.")
    print("뷰어 창을 닫으면 프로그램이 종료됩니다.\n")
    
    # MuJoCo 뷰어가 열려있는 동안 시뮬레이션 유지
    import time
    if hasattr(env, 'viewer') and env.viewer is not None:
        while env.viewer.is_running():
            # 랜덤 행동으로 계속 시뮬레이션
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            if terminated or truncated:
                obs, info = env.reset()
            
            time.sleep(0.01)  # CPU 사용량 조절
    
    env.close()
    print("\n테스트 완료!")
