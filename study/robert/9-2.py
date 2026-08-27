# 9-2.py: PPO Training Pipeline with Stable-Baselines3
# Week 09: PPO 학습 파이프라인 구축

import os
import argparse
import numpy as np

# Stable-Baselines3 임포트 (설치 필요)
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        CheckpointCallback,
        CallbackList,
        BaseCallback,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.env_util import make_vec_env
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Stable-Baselines3가 설치되지 않았습니다.")
    print("설치: pip install stable-baselines3 tensorboard")

import gymnasium as gym
from gymnasium import spaces

# ============================================================================
# 간단한 테스트 환경 (MuJoCo 없이 테스트 가능)
# ============================================================================
class SimpleWalkEnv(gym.Env):
    """PPO 학습 테스트를 위한 간단한 환경
    
    목표: 에이전트가 목표 높이를 유지하며 전진하도록 학습
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 60}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.n_joints = 12
        
        # Action: 관절 토크
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self.n_joints,), 
            dtype=np.float32
        )
        
        # Observation: [위치, 속도, 관절각, 관절속도]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(3 + 3 + self.n_joints * 2,),  # 33차원
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.max_steps = 500
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 초기 상태
        self.pos = np.array([0.0, 0.0, 0.2])  # x, y, z
        self.vel = np.zeros(3)
        self.joint_pos = np.zeros(self.n_joints)
        self.joint_vel = np.zeros(self.n_joints)
        self.current_step = 0
        self.prev_action = np.zeros(self.n_joints)
        
        return self._get_obs(), self._get_info()
    
    def _get_obs(self):
        return np.concatenate([
            self.pos,
            self.vel,
            self.joint_pos,
            self.joint_vel
        ]).astype(np.float32)
    
    def _get_info(self):
        return {
            "x_pos": self.pos[0],
            "height": self.pos[2],
            "step": self.current_step,
        }
    
    def step(self, action):
        action = np.clip(action, -1, 1)
        
        # 간단한 물리 시뮬레이션
        # 관절 업데이트
        self.joint_vel += action * 0.1
        self.joint_vel *= 0.95  # 감쇠
        self.joint_pos += self.joint_vel * 0.02
        self.joint_pos = np.clip(self.joint_pos, -1, 1)
        
        # 전진 속도 (관절 움직임에 비례)
        forward_impulse = np.sum(np.abs(self.joint_vel)) * 0.01
        self.vel[0] += forward_impulse
        self.vel *= 0.98  # 공기 저항
        
        # 높이 변화 (관절에 따라)
        height_change = np.mean(action[:4]) * 0.01
        self.pos[2] += height_change
        self.pos[2] = np.clip(self.pos[2], 0.05, 0.5)
        
        # 위치 업데이트
        self.pos += self.vel * 0.02
        
        self.current_step += 1
        
        # 보상 계산
        reward = self._compute_reward(action)
        
        # 종료 조건
        terminated = self.pos[2] < 0.08 or self.pos[2] > 0.4
        truncated = self.current_step >= self.max_steps
        
        self.prev_action = action.copy()
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _compute_reward(self, action):
        reward = 0.0
        
        # 1. 전진 보상
        reward += self.vel[0] * 2.0
        
        # 2. 높이 유지 보상 (목표: 0.2m)
        height_error = abs(self.pos[2] - 0.2)
        reward += np.exp(-5 * height_error)
        
        # 3. 생존 보상
        reward += 0.5
        
        # 4. 에너지 페널티
        reward -= 0.005 * np.sum(action ** 2)
        
        # 5. 부드러움 보상
        action_diff = np.sum((action - self.prev_action) ** 2)
        reward -= 0.01 * action_diff
        
        return reward
    
    def render(self):
        if self.render_mode == "human":
            print(f"Step {self.current_step:4d} | "
                  f"Pos: [{self.pos[0]:6.2f}, {self.pos[2]:6.2f}] | "
                  f"Vel: {self.vel[0]:6.3f}")


# ============================================================================
# 학습 콜백 (SB3가 설치된 경우에만 사용)
# ============================================================================
if SB3_AVAILABLE:
    class TensorboardCallback(BaseCallback):
        """커스텀 로깅 콜백"""
        
        def __init__(self, verbose=0):
            super().__init__(verbose)
        
        def _on_step(self) -> bool:
            # 추가 메트릭 로깅
            if self.n_calls % 1000 == 0:
                if "infos" in self.locals:
                    infos = self.locals["infos"]
                    if len(infos) > 0 and "x_pos" in infos[0]:
                        avg_x = np.mean([info.get("x_pos", 0) for info in infos])
                        self.logger.record("custom/avg_x_pos", avg_x)
            return True
else:
    # SB3가 없으면 더미 클래스 정의
    TensorboardCallback = None


# ============================================================================
# 환경 생성 헬퍼
# ============================================================================
def make_env(env_class=SimpleWalkEnv, render_mode=None, rank=0, seed=0):
    """환경 생성 헬퍼 함수"""
    def _init():
        env = env_class(render_mode=render_mode)
        if SB3_AVAILABLE:
            env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# ============================================================================
# 학습 함수
# ============================================================================
def train_ppo(
    total_timesteps=100_000,
    n_envs=4,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    save_dir="./models",
    log_dir="./logs",
):
    """PPO 학습 파이프라인
    
    Args:
        total_timesteps: 총 학습 스텝 수
        n_envs: 병렬 환경 수
        learning_rate: 학습률
        n_steps: 업데이트당 스텝 수
        batch_size: 미니배치 크기
        n_epochs: 에폭 수
        save_dir: 모델 저장 경로
        log_dir: 로그 저장 경로
    """
    if not SB3_AVAILABLE:
        print("Stable-Baselines3가 필요합니다!")
        return None
    
    # 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print("=" * 60)
    print("PPO Training Configuration")
    print("=" * 60)
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Parallel environments: {n_envs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Steps per update: {n_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {n_epochs}")
    print("=" * 60)
    
    # 1. 환경 설정
    env = DummyVecEnv([make_env(rank=i) for i in range(n_envs)])
    
    # 관측값/보상 정규화 (학습 안정성 향상)
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )
    
    # 2. PPO 모델 생성
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_dir,
        verbose=1,
        device="auto",
    )
    
    # 3. 콜백 설정
    # 평가 환경
    eval_env = DummyVecEnv([make_env()])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        training=False,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=max(10000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // n_envs, 1),
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="ppo_walk",
    )
    
    custom_callback = TensorboardCallback()
    
    callbacks = CallbackList([
        eval_callback,
        checkpoint_callback,
        custom_callback,
    ])
    
    # 4. 학습 시작
    print("\n학습을 시작합니다...")
    print("TensorBoard 모니터링: tensorboard --logdir=" + log_dir)
    print("-" * 60)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n학습이 중단되었습니다.")
    
    # 5. 최종 모델 저장
    final_model_path = os.path.join(save_dir, "ppo_final")
    model.save(final_model_path)
    env.save(os.path.join(save_dir, "vec_normalize.pkl"))
    
    print("=" * 60)
    print(f"학습 완료! 모델 저장: {final_model_path}")
    print("=" * 60)
    
    return model


# ============================================================================
# 평가 함수
# ============================================================================
def evaluate_model(model_path, n_episodes=5, render=True):
    """학습된 모델 평가
    
    Args:
        model_path: 모델 파일 경로
        n_episodes: 평가 에피소드 수
        render: 렌더링 여부
    """
    if not SB3_AVAILABLE:
        print("Stable-Baselines3가 필요합니다!")
        return
    
    print("=" * 60)
    print(f"모델 평가: {model_path}")
    print("=" * 60)
    
    # 환경 로드
    render_mode = "human" if render else None
    env = DummyVecEnv([make_env(render_mode=render_mode)])
    
    # 정규화 통계 로드 (있으면)
    vec_norm_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
    if os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False
        print(f"정규화 통계 로드: {vec_norm_path}")
    
    # 모델 로드
    model = PPO.load(model_path, env=env)
    
    # 평가
    episode_rewards = []
    episode_lengths = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            
            if render:
                env.envs[0].render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {ep+1}/{n_episodes}: "
              f"Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    env.close()
    
    # 결과 출력
    print("-" * 60)
    print(f"평균 보상: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"평균 길이: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print("=" * 60)


# ============================================================================
# 메인
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training Pipeline")
    parser.add_argument("--train", action="store_true", help="Train new model")
    parser.add_argument("--eval", type=str, default=None, 
                        help="Evaluate model (path to model file)")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering during evaluation")
    
    args = parser.parse_args()
    
    if args.train:
        train_ppo(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
        )
    elif args.eval:
        evaluate_model(
            args.eval,
            render=not args.no_render,
        )
    else:
        print("사용법:")
        print("  학습: python 9-2.py --train [--timesteps N] [--n-envs N]")
        print("  평가: python 9-2.py --eval <model_path> [--no-render]")
        print("\n예시:")
        print("  python 9-2.py --train --timesteps 50000")
        print("  python 9-2.py --eval ./models/ppo_final")
