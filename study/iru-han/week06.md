# Trot gait 패턴 이해
트롯 보행은 4족 보행 로봇의 가장 대표적인 보행 방식으로, 대각선 방향의 두 다리가 한 쌍이 되어 동시에 움직이는 것이 특징이다.
* 다리 쌍 구성: (앞 왼쪽 - 뒤 오른쪽) 다리와 (앞 오른쪽 - 뒤 왼쪽) 다리가 함께 움직임.
* 리듬: 한 쌍이 지면을 밀 때(Stance), 다른 쌍은 공중에서 앞으로 이동(Swing)하며 이 과정을 교차함. ->  다리 쌍 1이 땅을 딛고 있을 때 다리 쌍 2는 공중으로 이동하고, 그다음에는 반대로 움직인다.
* 코드 내 구현: kinematicMotion.py의 positions 함수를 보면 Tt/2(전체 주기의 절반)만큼의 시차를 두고 다리 위치를 계산하는 것을 볼 수 있다.

kinematicMotion.py의 핵심 로직
```
self.t0=p.readUserDebugParameter(self.IDt0)
self.t1=p.readUserDebugParameter(self.IDt1)
self.t2=p.readUserDebugParameter(self.IDt2)
self.t3=p.readUserDebugParameter(self.IDt3)

# 4. Positions: 보행 패턴 생성 (위상차)
Tt = (self.t0 + self.t1 + self.t2 + self.t3) # 전체 한 주기 시간
Tt2 = Tt / 2 # 주기의 절반 (시차)

# % Tt (Modulo 연산): 시간이 계속 흘러도 보행은 반복되어야 하므로, 총 주기로 나눈 나머지를 사용해 무한 반복되는 사이클 만들기
# Tt2: 위상차 수식. 1번 그룹 다리가 시작할 때, 2번 그룹 다리는 정확히 주기의 절반(180도) 뒤에서 시작하게 해서 Trot(트롯) 보행을 만듦. 이걸 Tt / 4로 바꾸고 4개 다리를 각각 밀어주면 고양이 보행.
td = (t * 1000) % Tt       # 1번 그룹 시간 (Group A)
t2 = (t * 1000 - Tt2) % Tt # 2번 그룹 시간 (Group B: 절반만큼 늦게 시작)
```

* `t0`: 지면에서 대기 (보통 0으로 설정)
* `t1 (Stance)`: 지면을 밀어내며 몸을 앞으로 전진시킴
* `t2`: 지면에서 다시 대기 (보통 0으로 설정)
* `t3 (Swing)`: 다리를 들어 올려 다음 위치로 이동

  
# Swing/Stance phase 구현
kinematicMotion.py 파일의 calcLeg 함수에서 트롯 보행의 핵심인 **Swing(공중 이동)**과 Stance(지면 지지) 단계를 구현한다
```
def calcLeg(self, t, x, y, z):
        """
        시간(t)에 따른 다리 하나하나의 3차원 좌표를 계산함
        """
        # 발의 시작점과 끝점 계산 (보폭 Sl의 절반씩 앞뒤로 배치)
        startLp = np.array([x - self.Sl/2.0, y, z - self.Sw/2, 1])
        endLp = np.array([x + self.Sl/2, y, z + self.Sw/2, 1])

        # 1. Stance Phase 전 대기 (t0)
        if(t < self.t0):
            return startLp
            
        # 2. Stance Phase (지면 지지기): 발을 뒤로 밀어 로봇을 앞으로 보냄
        elif(t < self.t0 + self.t1):
            td = t - self.t0 # td: 경과 시간

            # 1. Stance Phase: 지면 밀기 (추진력 생성)
            tp = td / self.t1 # Stance 진행률 (0~1), 현재 Stance 단계(t1)가 얼마나 진행되었는지 나타내는 비율. 시작할 때 0, 끝날 때 1.
            diffLp = endLp - startLp # 발이 이동해야 할 총 직선 거리(end - start)를 나타내는 벡터
            curLp = startLp + diffLp * tp # 직선으로 발을 뒤로 끎(drag). 선형 보간 수식. 시작점에서 목표점까지 tp 비율만큼 이동한 현재 위치를 계산

            # 2. 회전 변환 수식 (방향 제어)
            # 발의 회전(Sa, 알파각) 적용
            psi = (math.pi/180 * self.Sa) * tp # self.Sa(Step Alpha, 회전각)를 라디안으로 바꾼 뒤 진행률(tp)을 곱한 값. 즉, 발이 뒤로 가면서 동시에 몸체도 서서히 회전하게 만듦
            Ry = np.array([[np.cos(psi), 0, np.sin(psi), 0],
                           [0, 1, 0, 0],
                           [-np.sin(psi), 0, np.cos(psi), 0],
                           [0, 0, 0, 1]]) # Y축(수직축)을 기준으로 좌표를 회전시키는 3D 회전 행렬
            return Ry.dot(curLp) # dot: 행렬 곱셈을 통해 계산된 직선 좌표(curLp)를 회전된 좌표로 변환. 로봇이 커브를 틀며 걷게 해주는 핵심 수식

        # 3. Swing Phase 전 대기 (t2)
        elif(t < self.t0 + self.t1 + self.t2):
            return endLp

        # 4. Swing Phase (유각기): 발을 들어올려 다시 원위치로 복귀
        elif(t < self.t0 + self.t1 + self.t2 + self.t3):
            td = t - (self.t0 + self.t1 + self.t2)
            tp = td / self.t3 # Swing 진행률 (0~1)
            diffLp = startLp - endLp
            curLp = endLp + diffLp * tp

            # 3. Swing Phase: 발 들기 (장애물 극복 및 복귀)
            # Y축(높이)에 Sine 곡선을 적용하여 발을 들어올림 (Sh: 발 높이)
            curLp[1] += self.Sh * math.sin(math.pi * tp)
            # math.pi * tp: tp가 0에서 1까지 변할 때, 사인 함수의 입력값은 0에서 180까지 변함.
            # sin(...): 발을 들기 시작해서 중간에 가장 높이 찍고 다시 내려오는 부드러운 아치형 궤적
            # self.Sh: 이 사인 값에 Sh(Step Height)를 곱해서 실제 발이 얼마나 높이 뜰지 결정
            return curLp
```

* 정리
<img width="667" height="267" alt="image" src="https://github.com/user-attachments/assets/d23efe27-9978-4f8c-ba4f-022e96a2cb3c" />

  
# 제자리 걷기 시뮬레이션
kinematicMotion.py
```
def calcLeg(self, t, x, y, z):
        # x, y, z는 로봇 설계상의 기본 발 위치.
        # 제자리 걷기이므로 시작과 끝을 그냥 기본 위치(x, y, z)로 통일.
        fixedLp = np.array([x, y, z, 1])
        
        # 1. Stance Phase (땅에 딛고 있는 모든 구간)
        if (t < self.t0 + self.t1 + self.t2):
            # drag나 stay 상관없이 무조건 기본 위치 고정
            return fixedLp
            
        # 2. Swing Phase (발만 수직으로 들기)
        elif (t < self.t0 + self.t1 + self.t2 + self.t3):
            td = t - (self.t0 + self.t1 + self.t2)
            tp = td / self.t3
            
            curLp = fixedLp.copy()
            # X, Z는 그대로 두고 Y(높이)만 Sine 파형으로 변화
            curLp[1] += self.Sh * math.sin(math.pi * tp)
            return curLp
        
        else:
            return fixedLp
```
-> 기구학적으로 X, Z축 이동량을 제거하였음에도 보행 주기 내 무게 중심(CoM)의 미세한 변화와 지면 접촉 시의 충격으로 인해 미세한 변위가 발생함
  
  


  
spotmicroai.py
```
    def changeDynamics(self,quadruped):
        nJoints = p.getNumJoints(quadruped)
        for i in range(nJoints):
            p.changeDynamics(
                quadruped, 
                i, 
                lateralFriction=0.5,           # 측면 마찰력 (미끄러짐 방지)
                spinningFriction=0.1,          # 회전 마찰력 (제자리 회전 방지)
                rollingFriction=0.01,          # 구름 마찰력
                restitution=0.0,               # 탄성 계수 (발이 땅에 닿을 때 튀지 않게 함)
                linearDamping=0.04,            # 선형 댐핑 (진동 억제)
                angularDamping=0.04,           # 각속도 댐핑 (회전 흔들림 억제)
                localInertiaDiagonal=[0.001, 0.001, 0.001] # 관성 모멘트 (너무 작으면 물리적 오류 발생)
            )
```
마찰력을 수정해야 제자리에서 걷는다 (아니면 앞으로 감)

<img width="397" height="350" alt="image" src="https://github.com/user-attachments/assets/8306a91a-695e-473a-86ac-3743ab43500c" />
  
  
  
  


# 논문 - Bridge the Gap: Enhancing Quadruped Locomotion with Vertical Ground Perturbations
## 1. 연구의 핵심 목표

로봇 개(사족 보행 로봇)는 산길이나 거친 땅은 잘 다니지만, 위아래로 출렁거리는 구름다리 같은 곳에서는 중심을 잡기 매우 어려움.  
이 논문은 로봇이 **'바닥이 위아래로 심하게 흔들리는 환경'**에서도 넘어지지 않고 안정적으로 걸을 수 있도록 만드는 방법을 연구함.

---

## 2. 실험 방법: 가상 세계에서 스파르타 훈련

- **진짜 같은 가상 공간**  
  연구진은 13.24m 길이의 실제 흔들리는 다리를 컴퓨터 시뮬레이션 속에 똑같이 만듦.

- **강화학습(AI 훈련)**  
  로봇에게 보상을 주며 스스로 걷는 법을 배우게 하는 '강화학습(PPO)' 방식을 사용함.  
  로봇은 수천 번 넘어지면서 바닥이 흔들릴 때 어떻게 발을 디뎌야 균형을 잡을 수 있는지 스스로 터득함.

- **다양한 걸음걸이**  
  Trot(트롯), Pace(페이스), Bound(바운드), Free(자유 보행), Default 등 5가지의 다양한 보행 패턴을 훈련시켜 어떤 상황에서도 적응하게 함.

---

## 3. 결과: "흔들려 본 로봇이 훨씬 잘 걷는다"

- **압도적인 안정성**  
  처음부터 딱딱한 바닥에서만 연습한 로봇은 다리가 흔들리자마자 넘어졌지만,  
  흔들리는 다리에서 훈련받은 로봇은 실제 환경에서도 아주 능숙하게 다리를 건넘.

- **적응력**  
  이 로봇은 훈련받지 않은 새로운 형태의 흔들림이 나타나도 당황하지 않고 보폭과 높이를 조절하며 균형을 유지함.

---

## [분석 의견] 시뮬레이션과 실제 물리 법칙의 상관관계

### 1. 기구학적 설계와 AI 제어의 일치성

본 프로젝트에서 구현한 Trot gait(트롯 보행) 알고리즘은 논문에서 사용된 로봇의 기본 보행 원리와 동일함.  
논문은 AI가 스스로 파라미터를 찾게 했고, 우리는 수학적 수식(calcLeg, positions)을 통해 그 원리를 직접 구현함.  
결국, 로봇이 안정적으로 걷기 위해서는 발을 딛는 시간(Stance)과 드는 시간(Swing)의 완벽한 비율이 필수적임을 확인함.

---

### 2. 시뮬레이션 환경 설정의 중요성 (Sim-to-Real)

논문에서 강조한 '도메인 랜덤화(물리 수치를 무작위로 바꾸는 것)'는 우리가 changeDynamics를 통해 **마찰력(Friction)**과 **댐핑(Damping)**을 조절한 것과 같은 맥락임.  
제자리 걷기 시뮬레이션에서 보폭을 0으로 설정했음에도 로봇이 조금씩 이동했던 현상은,  
논문에서 언급한 '지면 충격량'과 '무게 중심의 변화'가 실제 물리 세계에서 얼마나 큰 변수인지를 보여주는 실례임.

---

### 3. 제자리 보행 안정성 확보의 의미

제자리 걷기 실험을 통해 마찰력과 관성 모멘트를 튜닝한 과정은 논문에서 다룬 **'수직 섭동(Vertical Perturbation) 대응'**의 기초 단계라고 볼 수 있음.  
지면의 마찰력을 조절하여 미끄러짐을 방지하고, 발의 높이(Sh)를 최적화하여 착지 충격을 줄인 것은  
로봇이 외부의 흔들림(섭동)에 저항할 수 있는 물리적 기초를 다지는 필수적인 과정임.

---

### 4. 향후 발전 방향

단순히 평평한 지면에서의 보행을 넘어,  
논문의 사례처럼 시뮬레이션 환경에 미세한 흔들림이나 경사면을 추가하여 테스트한다면,  
우리가 구현한 기구학 코드가 얼마나 강건(Robust)한지 더 깊이 있게 검증할 수 있을 것으로 판단됨.

