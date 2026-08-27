### 1. wsl

PowerShell 키고

```jsx
wsl -d Ubuntu
sudo apt update
sudo apt install git
sudo apt install python3-pip
sudo apt install python3.12-venv
sudo apt install build-essential python3-dev # C/C++컴파일러 및 빌드도구, 파이썬 개발 헤더 파일
cd ~/
mkdir projects
cd projects
git clone https://github.com/Road-Balance/SpotMicroJetson.git
cd SpotMicroJetson
python3 -m venv venv
source venv/bin/activate
cd Simulation
pip install -r requirements.txt
```

여기까지 진행 후 버전 에러 나서 requirements.txt 수정

vi requirements.txt

```bash
gym==0.26.0
matplotlib==3.8.2
numpy==1.26.4
pybullet==3.2.7
keyboard==0.13.5
```

esc + :wq 로 저장

```jsx
pip install --upgrade setuptools wheel # 빌드 도구 업데이트
pip install -r requirements.txt # 패키지 설치
python pybullet_automatic_gait.py
```

![](https://blog.kakaocdn.net/dna/1JZZv/dJMcachhbXs/AAAAAAAAAAAAAAAAAAAAAH3veYwi_3IV2vCqo1UouEeb5zGEPB2J7Yl9sMKmeWos/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1767193199&allow_ip=&allow_referer=&signature=9GeH1fpHUDeI4%2B582GCr4e2PwUo%3D)

키보드 입력이 안 먹어서 키보드 이벤트를 잘 처리하도록 도와주는 라이브러리

```jsx
sudo apt install x11-xserver-utils
```

설치후 다시 해봤는데 렉이 걸려서 wsl에서 시뮬레이션 구동하는건 해결 못한 상태

source venv/bin/activate

명령어 해주어야 함

### 2. Power Shell

파이썬은 여기에서 다운로드를 진행했다

https://www.python.org/downloads/release/pymanager-250/

Zip 파일 다운받아서 압축 해제하기

(경로: C:\Users\lkno0\OneDrive\문서\SpotMicroJetson-master\Simulation)

https://github.com/Road-Balance/SpotMicroJetson

관리자 권한으로 Power Shell 키고

```jsx
ExecutionPolicy
Set-ExecutionPolicy Unrestricted # y 누르기
cd C:\Users\lkno0\OneDrive\문서\SpotMicroJetson-master\Simulation
python -m venv spot_env # 가상 환경 생성
.\spot_env\Scripts\Activate.ps1 # 가상 환경 활성화
```

requirements.txt 버전 변경

```
gym==0.26.0
matplotlib==3.8.2
numpy==1.26.4
pybullet==3.2.7
keyboard==0.13.5
```

시뮬레이션 구동

```jsx
python.exe -m pip install --upgrade pip
pip install -U -r requirements.txt # 라이브러리 설치
python .\pybullet_automatic_gait.py
```

해결이 안됐던 부분: 시뮬레이션을 껐다가 파워셸로 파이뷸렛을 다시 키면 에러가 나면서 안뜹니다

```jsx
(spot_env) PS ...\SpotMicroJetson-master\Simulation> python .\pybullet_automatic_gait.py
pybullet build time: Dec  2 2025 22:41:41
starting thread 0
started testThreads thread 0 with threadHandle 00000000000003F8
argc=2
argv[0] = --unused
argv[1] = --start_demo_name=Physics Server
ExampleBrowserThreadFunc started
Version = 4.6.0 Compatibility Profile Context 25.10.2.251010
Vendor = ATI Technologies Inc.
Renderer = AMD Radeon RX 9070 XT
b3Printf: Selected demo: Physics Server
starting thread 0
started MotionThreads thread 0 with threadHandle 00000000000005AC
MotionThreadFunc thread started
b3Printf: b3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,152]:

b3Printf: URDF file 'plane_transparent.urdf' not found

Traceback (most recent call last):
  File "...\SpotMicroJetson-master\Simulation\pybullet_automatic_gait.py", line 53, in <module>
    robot=spotmicroai.Robot(True,True,reset)
  File "...\SpotMicroJetson-master\Simulation\spotmicroai.py", line 75, in __init__
    self.quadruped = self.loadModels()
                     ~~~~~~~~~~~~~~~^^
  File "...\SpotMicroJetson-master\Simulation\spotmicroai.py", line 154, in loadModels
    planeUid = p.loadURDF("plane_transparent.urdf", [0, 0, 0], orn)
pybullet.error: Cannot load URDF file.
numActiveThreads = 0
stopping threads
Thread with taskId 0 with handle 00000000000005AC exiting
Thread TERMINATED
finished
numActiveThreads = 0
btShutDownExampleBrowser stopping threads
Thread with taskId 0 with handle 00000000000003F8 exiting
Thread TERMINATED
```

