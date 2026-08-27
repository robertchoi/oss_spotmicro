w를 누르면 앞으로, s를 누르면 뒤로 이동한다. 스페이스바를 누르면 로봇이 멈춘다.

1. 땅바닥에서 실험
kp:로봇이 원하는 각도와 현재 각도의 차이에 비례하여 모터에 힘을 주는 정도를 결정한다.
→ 0으로 설정하면 로봇이 멈춰있고, 최대로 하면 로봇이 빠르게 가는 것 같다.
<img width="842" height="580" alt="image" src="https://github.com/user-attachments/assets/86b50614-5eb8-41e9-b29a-7bed705af743" />

kd: 모터가 목표 각도를 향해 움직이는 속도 변화를 감지하여 제동력(댐핑)을 주는 정도를 결정한다. 
→ 0으로 설정하면 로봇이 제자리에서 회전하면서 앞으로 나아가지 못하고, 최대로 설정하면 안정적으로 나아가긴 하는데 느린것 같기도 함.

MaxForce: 모터가 목표 각도를 맞추기 위해 낼 수 있는 최대 힘을 제한한다. 
이 값이 너무 낮으면 로봇의 무게를 이기지 못하거나 빠르게 움직이지 못하고, 너무 높으면 에너지를 많이 소모하거나 로봇이 거칠게 움직일 수 있다. 
→ 0으로 설정하니까 로봇이 주저앉았다
<img width="854" height="581" alt="image" src="https://github.com/user-attachments/assets/2b0a5dd0-180a-4350-8a51-e8013512892c" />

step length: 걸음 폭. 로봇이 한 걸음에 앞으로 몇 mm 이동할지 설정한다.
→ 최대로 하니까 빨리 가는 것 같긴 한데 - 로 한다고 뒤 방향으로 이동하는 것 같진 않다.

step width: 옆 걸음 폭. 로봇이 옆으로 몇 mm 이동할지 설정한다.

step height: 발 들기 높이. 로봇이 발을 공중에 들 때 최고 높이가 몇 mm가 되도록 할지 설정한다.

step alpha: 발끝 회전 각도. 발을 땅에서 떼서 이동할 때, 발끝을 Z축 기준으로 얼마나 회전시킬지 설정한다.

t0: 대기시간. 발을 땅에 딛고 잠시 머무르는 시간. 값이 작으면 작을수록 로봇이 발을 땅에서 빨리 뗀다.

t1: 지지시간. 발을 땅에 딛고 있으면서 몸통을 목표 방향으로 밀어주는 시간.

t2: 대기시간. 발이 땅에 닿은 뒤 잠시 머무르는 시간. 값이 작으면 작을수록 로봇이 발을 땅에서 빨리 뗀다.

t3: 공중 이동 시간. 발을 땅에서 떼어 최고 높이까지 들었다가 다시 착지 지점으로 이동시키는 시간.
→ 로봇의 발걸음 속도가 달라지는 것 같다.

--

2. 공중에서 실험
height: 로봇 몸통의 높이를 조절한다.
<img width="796" height="1083" alt="image" src="https://github.com/user-attachments/assets/924883e2-4e27-418c-bbb7-682c97569372" />

super front: 앞다리 옆 간격을 조정한다.
→ 앞다리 간격을 좁게하고, 뒷다리 옆 간격을 넓게하면 로봇이 뒤로간다.
<img width="711" height="967" alt="image" src="https://github.com/user-attachments/assets/c5df6c07-243f-48f7-b12b-78da017e8518" />

super rear: 뒷다리 옆 간격을 조정한다
<img width="704" height="977" alt="image" src="https://github.com/user-attachments/assets/45a6ec1b-51bb-463f-8505-925b2c7e722f" />

front offset: 앞다리 앞뒤 간격을 조정한다
<img width="691" height="1001" alt="image" src="https://github.com/user-attachments/assets/2704bf85-fee6-47c0-bfd8-5d9ad816d032" />

rear offset: 뒷다리 앞뒤 간격을 조정한다
<img width="710" height="966" alt="image" src="https://github.com/user-attachments/assets/5959f4af-8dd9-43b5-87b4-6d099e1dda4e" />

--

로봇의 자세마다 (앞다리 간격, 뒷다리 간격 등) 걸음 방향과 속도가 달라진다는게 인상적이다.

