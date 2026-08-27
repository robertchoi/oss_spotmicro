import gymnasium as gym
from stable_baselines3 import PPO
import mujoco
import mujoco.viewer
import time

# 방금 만든 학습 코드(train_spot.py)에서 환경 클래스를 가져옵니다.
# 만약 파일명이 다르면 'from 파일명 import SpotMicroEnv'로 수정하세요.
from train_spot import SpotMicroEnv


def test():
    # 1. 테스트용 환경 생성 (render_mode='human' 필수!)
    # 화면을 띄워서 눈으로 봐야 하니까 'human' 모드로 켭니다.
    env = SpotMicroEnv(xml_path='scene.xml', render_mode='human')

    # 2. 저장된 모델 불러오기
    # 확장자 .zip은 생략 가능합니다.
    try:
        model = PPO.load("ppo_spot_micro")
        print("학습된 모델을 성공적으로 불러왔습니다!")
    except FileNotFoundError:
        print("오류: 'ppo_spot_micro.zip' 파일을 찾을 수 없습니다.")
        print("먼저 train_spot.py를 실행해서 모델을 학습시켜주세요.")
        return

    # 3. 테스트 루프
    obs, _ = env.reset()

    print("테스트 시작! (Ctrl+C로 종료)")
    while True:
        # AI에게 현재 상황(obs)을 보여주고 행동(action)을 결정하게 함
        # deterministic=True: 학습 때와 달리 '모험'을 하지 않고 가장 확실한 행동만 함
        action, _ = model.predict(obs, deterministic=True)

        # 결정한 행동으로 시뮬레이션 한 스텝 진행
        obs, reward, done, truncated, info = env.step(action)

        # 만약 넘어졌다면(done) 다시 일으켜 세움
        if done:
            print("넘어짐! 다시 시작합니다.")
            obs, _ = env.reset()
            time.sleep(1.0)  # 넘어졌을 때 잠깐 대기

        # 너무 빠르면 눈으로 보기 힘드니 속도 조절 (선택 사항)
        # time.sleep(0.01)


if __name__ == "__main__":
    test()