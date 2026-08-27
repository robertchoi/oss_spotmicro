1. Kinematics(운동학) vs Dynamics(동역학)
<img width="1568" height="1212" alt="image" src="https://github.com/user-attachments/assets/a256e5a6-7fd8-47d5-91ac-88d66eb27b9b" />

로봇이 '어떻게' 움직이는지에 대한 두 가지 관점을 설명한다.

* Kinematics (운동학): 힘(전기, 토크)은 신경 쓰지 않고 **'위치, 각도, 속도'**만 따지는 것.
* Dynamics (동역학): 운동학에 **'힘(Force)'**을 더한 것. 로봇의 무게, 관성, 모터의 힘을 계산한다.
* 자유도 (DOF): 그림 속 6DOF는 관절이 6개라는 뜻이다. SpotMicro는 다리당 3개, 총 12자유도 로봇이다.

======================================================

2. Forward(순운동학) vs Inverse(역운동학)
<img width="1533" height="935" alt="image" src="https://github.com/user-attachments/assets/029c9bc0-7575-4eb4-b7a4-100058ac9400" />

* Forward Kinematics (FK, 순운동학): 각도(theta)를 넣어서 좌표(x, y, z)를 얻는 것, "모터를 30도 돌리면 내 로봇 발끝은 어디에 가 있지?"
-> 각도를 알 때 위치 찾기

* Inverse Kinematics (IK, 역운동학): 좌표(x, y, z)를 넣어서 각도(theta)를 얻는 것. "발끝을 좌표 (10, 5, 0)에 두고 싶은데 모터를 몇 도 돌려야 해?"
-> 위치를 알 때 각도 찾기

로봇은 세상을 좌표(x, y, z)로 이해하고, 몸은 각도(theta)로 움직인다.

======================================================

3. 2 자유도일 때 발의 위치 (x, y)에 가기 위해 각 관절이 몇 도(theta_1, theta_2)가 되어야 하는지 계산하는 과정(IK)
<img width="662" height="960" alt="image" src="https://github.com/user-attachments/assets/77456b3c-89ae-49be-8dee-beef5418c48a" />

* 문제의 정의: 로봇 다리 구조는 링크 두 개(l_1, l_2)가 연결된 형태. 로봇의 다리는 길이(l_1, l_2)는 이미 알고있다.
* 목표: 발끝을 좌표 (x, y)에 두고 싶다. 그러려면 첫 번째 관절 각도 theta_1과 두 번째 관절 각도 theta_2를 얼마로 꺾어야 할까?

* 순운동학(Forward Kinematics): 박스 친 수식 부분. 각도(theta_1, theta_2)를 알 때 발끝 좌표(x, y)가 어디일지 구함.
  <img width="264" height="89" alt="image" src="https://github.com/user-attachments/assets/5a68983e-e97f-4f86-8ec8-8f0fc1d861fe" />

* 역운동학(Inverse Kinematics): 사진의 중간부터 아래까지.
   삼각함수의 합성: x와 y 식을 제곱해서 더하고 정리하면 sin(theta_1 + alpha) 꼴의 형태가 나온다.
* 역운동학 결과값 (theta_1): 결국 아래 수식처럼 theta_1을 x, y, l_1, l_2만으로 표현하는 데 성공했다.
  <img width="458" height="97" alt="image" src="https://github.com/user-attachments/assets/c4a96c4d-b61d-4ff6-a3ca-5708c4e65d38" />

* 역운동학 공식에 원하는 좌표 (x, y)와 다리 길이(l_1, l_2)만 넣으면 바로 첫 번째 관절 각도가 나오게된다.

* theta_2 구하기
  <img width="700" height="960" alt="image" src="https://github.com/user-attachments/assets/fb111974-58ac-4fe3-ba3b-07f6d7612c74" />

  첫 번째 사진에서 theta_1을 구했으니, 두 번째 사진에서는 비슷한 방식으로 theta_2 값을 구하고 있다.

======================================================

4. 로봇 다리 설계도
<img width="708" height="960" alt="image" src="https://github.com/user-attachments/assets/b7879eb1-3093-4051-b170-fb3debb39fca" />

Jetson: 로봇의 뇌(컴퓨터)가 들어있는 몸통

theta_1, theta_2, theta_3 (세타): 움직여야 할 모터 3개.
* theta_1: 어깨 관절. 다리를 몸통 쪽으로 붙이거나 바깥으로 벌리는 역할 (좌우 이동).
* theta_2: 허벅지 관절. 다리를 앞뒤로 흔드는 역할.
* theta_3: 무릎 관절. 다리를 굽혔다 폈다 하는 역할.

로봇은 이 3개의 모터를 적절히 돌려서 발끝을 원하는 위치에 갖다 놓아야한다. 이 3개의 모터가 유기적으로 움직여야 다리가 꼬이지 않고 원하는 곳으로 뻗을 수 있다.

======================================================

5. 로봇의 다리에 위(3번)에서 구했던 공식 적용해서 theta_1 계산
<img width="658" height="960" alt="image" src="https://github.com/user-attachments/assets/c44bdb1a-db55-4e08-baf8-f4e296a3cff4" />

* 로봇 다리는 몸통 한가운데 붙어 있는 게 아니라, 옆에(L_1만큼 떨어져서) 붙어 있다. 그래서 그냥 목표 지점을 바라보고 다리를 뻗으면 엉뚱한 곳을 짚게 된다. 따라서 "어깨너비"만큼 보정을 해줘야 함.
* 다리를 위에서 봤을 때, 몸체에서 발까지의 거리와 각도를 이용해 첫 번째 모터를 얼마나 돌려야 할지를 계산한 것.
결론적으로 아래 수식과 같은 아주 단순한 정답을 찾아냄

<img width="98" height="40" alt="image" src="https://github.com/user-attachments/assets/55f13d8b-70f9-4f26-abd1-5f862a66ad84" />

* (x, y): 발을 딛고 싶은 목표 위치.
* alpha (알파): 몸통 중심에서 목표 지점까지의 전체 각도.
* beta (베타): 내 어깨너비(L_1) 때문에 생기는 오차 각도.

======================================================

6. 3D 통합 계산
<img width="660" height="960" alt="image" src="https://github.com/user-attachments/assets/94b38462-d691-4d86-ae7d-f92c919684a8" />

앞에서 구한 평면 계산을 입체(3D)로 확장하는 과정. 
최종적으로 목표 좌표 (x, y, z)를 주면, 3개의 모터 각도 (theta_1, theta_2, theta_3)를 구할 수 있다.
발끝의 위치만 알면 각각의 점에 대해 각도를 알 수 있다.
