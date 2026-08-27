# pybullet_automatic_gait.py 를 다음과 같이 수정했을 때 로봇이 회전을 하며 이동했다
```
"""
Automatic Gait Simulation (Final Homework Version)
"""
import sys
sys.path.append("..")
import numpy as np
import time
import math
import pybullet as p
import spotmicroai
from kinematicMotion import TrottingGait

# 전역 타이머 리셋 함수
rtime = time.time()
def reset_timer():
    global rtime
    rtime = time.time()

def main():
    # [설정 1] 로봇 생성 (useFixedBase=False 필수)
    robot = spotmicroai.Robot(useFixedBase=False, useStairs=False, resetFunc=reset_timer)

    # [설정 2] 디버그 슬라이더 설정
    IDheight = p.addUserDebugParameter("Body Height", -40, 90, 20)

    # [설정 3] 초기 발 위치
    spurWidth = robot.W/2 + 20
    iXf = 120
    Lp = np.array([[iXf, -100, spurWidth, 1], [iXf, -100, -spurWidth, 1],
                   [-50, -100, spurWidth, 1], [-50, -100, -spurWidth, 1]])

    trotting = TrottingGait()

    print("Simulation Start... (Homework Mode)")

    while True:
        # 로봇 상태 읽기
        bodyPos = robot.getPos()
        bodyOrn, _, _ = robot.getIMU()
        xr, yr, _ = p.getEulerFromQuaternion(bodyOrn)

        # 리셋 로직 (너무 멀리 가면 원위치)
        if math.sqrt(bodyPos[0]**2 + bodyPos[1]**2) > 50:
            robot.resetBody()
            reset_timer()

        d = time.time() - rtime
        height = p.readUserDebugParameter(IDheight)
        ir = xr / (math.pi/180)

        # ====================================================
        # 아래 변수들의 숫자를 바꿔가며 캡처.
        # ====================================================

        # 1. 보행 제어 (전진 및 회전)
        target_step_length = 30   # 보폭 (0이면 제자리, 30이면 전진)
        target_rotation = 15      # 회전 (0이면 직진, 15면 좌회전)

        # 2. 자세 제어 (Body Pitch / Roll)
        # 0.0은 수평. 0.3이나 -0.3 등을 넣기.
        control_pitch = 0      # 앞뒤 기울기
        control_roll = 0       # 좌우 기울기

        # 3. CoM (무게중심) 추적 제어
        # 몸통(Body)을 다리 중심에서 강제로 밀어버리는 오프셋
        # 0이면 정상, 50 이상이면 앞으로 쏠려서 넘어질 수 있음
        com_offset_x = 0         # 앞뒤 무게중심 이동
        com_offset_z = 0         # 좌우 무게중심 이동

        # ====================================================

        # 보행 명령 생성
        cmd = {
            'IDstepLength': target_step_length,
            'IDstepWidth': 0,
            'IDstepAlpha': target_rotation
        }

        # 로봇 발 위치 제어 (Inverse Kinematics)
        robot.feetPosition(trotting.positions(d, cmd))

        # Body Rotation 제어 적용
        # 위에서 설정한 pitch, roll 값을 로봇에게 전달
        robot.bodyRotation((control_roll, control_pitch, 0))

        # Body Position (CoM) 제어 적용
        # CoM 위치 = 기본 위치 + 오프셋
        final_x = 50 + (yr * 10) + com_offset_x
        final_z = -ir + com_offset_z

        robot.bodyPosition((final_x, 40 + height, final_z))

        # 물리 시뮬레이션 1스텝 진행
        robot.step()

if __name__ == "__main__":
    main()
```

control_pitch 가 0.5 <br>
<img width="264" height="262" alt="image" src="https://github.com/user-attachments/assets/bd60b2f4-6545-4428-ae73-06370ef301e7" /> <br>
<img width="151" height="157" alt="image" src="https://github.com/user-attachments/assets/b3ebc6cb-a850-4d49-bed6-536abba60078" /> <br>
<br><br>
control_pitch 가 -0.5 <br>
<img width="162" height="180" alt="image" src="https://github.com/user-attachments/assets/cdddbef1-7303-4886-abe2-c65b02cf757c" /><br>
<img width="285" height="220" alt="image" src="https://github.com/user-attachments/assets/444aa85c-29ca-4255-9cf4-c4e32904432e" /><br>
<img width="181" height="144" alt="image" src="https://github.com/user-attachments/assets/5bdf4057-3a37-4cab-9ab8-4727a025805a" /><br>
<br><br>


control_roll 이 0.5  
<img width="147" height="186" alt="image" src="https://github.com/user-attachments/assets/38f14577-d411-4bc8-aed8-9223544f3915" />  <br>
<img width="149" height="143" alt="image" src="https://github.com/user-attachments/assets/1ecad31e-8de8-43d8-b632-83d2539702ec" />  <br>
<br><br>
control_roll 이 -0.5  
<img width="201" height="224" alt="image" src="https://github.com/user-attachments/assets/cc817c7b-b457-420e-a3cf-f586b57c951b" />  <br>
<img width="106" height="108" alt="image" src="https://github.com/user-attachments/assets/acb6e565-3cce-425a-bc48-78fdcf425ad4" />  <br>
<br><br>

com_offset_x 가 50  
<img width="265" height="191" alt="image" src="https://github.com/user-attachments/assets/4f994c2d-d9d6-425b-8a7c-dad9dbc76746" />  <br>
<img width="229" height="183" alt="image" src="https://github.com/user-attachments/assets/49770b48-d56d-4eac-b7cd-59a9d8cf4548" />  <br>
<img width="134" height="148" alt="image" src="https://github.com/user-attachments/assets/502155e9-f990-4ac1-bcf5-0849b88a2627" />  <br>
<br><br>

com_offset_z 가 20 (30 이상으로 하면 옆으로 넘어져버림)  
<img width="175" height="201" alt="image" src="https://github.com/user-attachments/assets/401709db-ed36-4ee2-8dcf-91a1a0c81159" />  <br>
<img width="150" height="174" alt="image" src="https://github.com/user-attachments/assets/c045f0fa-1f5c-44cc-be2f-f47b2668031f" />  <br>
<br><br>

# spotmicroai.py

로봇 발 위치 제어  <br>
```
    def feetPosition(self,Lp):
        self.Lp=Lp
```

자세 제어  <br>
```
    def bodyRotation(self,rot):
        self.rot=rot
```

무게중심 제어<br>
```
    def bodyPosition(self,pos):
        self.pos=pos
```

step 함수를 보면 결국 calcIK 로 Lp, rot, pos 를 전달한다 (실질적으로 조정하는것으로 보임)<br>
calcIK 는 kinematics.py 파일에 있다
```
self.angles = self.kin.calcIK(self.Lp, self.rot, self.pos)
```

# kinematics.py
```
    def calcIK(self,Lp,angles,center):
        (omega,phi,psi)=angles
        (xm,ym,zm)=center
        
        (Tlf,Trf,Tlb,Trb)= self.bodyIK(omega,phi,psi,xm,ym,zm)

        Ix=np.array([[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        return np.array([self.legIK(np.linalg.inv(Tlf).dot(Lp[0])),
        self.legIK(Ix.dot(np.linalg.inv(Trf).dot(Lp[1]))),
        self.legIK(np.linalg.inv(Tlb).dot(Lp[2])),
        self.legIK(Ix.dot(np.linalg.inv(Trb).dot(Lp[3])))])
```
bodyIK: "몸통이 움직이면 어깨는 어디로 가는지?" <br>-> 이 함수는 **몸통의 자세(기울기, 위치)**를 입력받아서, 네 다리의 어깨(Shoulder) 위치가 공간상에서 어디로 이동하는지 계산<br><br>
legIK: "발 끝이 저기 있으려면 무릎은 얼마나 굽히는지?" <br>-> 이 함수는 어깨를 기준으로 **발 끝의 위치(x, y, z)**를 주면, 모터 3개(골반, 허벅지, 종아리)의 각도를 계산<br><br>
calcIK: "모든 걸 종합해서 모터 돌려라!" <br>-> 이 함수가 bodyIK와 legIK를 합쳐서 최종 명령을 내리는 사령관<br><br>

```
def bodyIK(self, omega, phi, psi, xm, ym, zm):
    # omega, phi, psi: Roll, Pitch, Yaw (회전 각도)
    # xm, ym, zm: X, Y, Z (몸통 위치)
```
작동 원리:<br>
회전 행렬(Rx, Ry, Rz): 입력받은 각도만큼 수학적으로 좌표를 회전.<br>
이동 행렬(T): 입력받은 위치(xm, ym, zm)만큼 좌표를 이동시킴.<br>
어깨 위치 계산: 원래 몸통 중심에서 떨어져 있던 네 개의 어깨(Tlf, Trf...)가 몸통이 회전하고 이동함에 따라 **새로운 좌표(공간상의 위치)**로 바뀜.<br>
요약: "내가 몸을 앞으로 30도 기울이고 5cm 낮추면, 내 왼쪽 앞 어깨는 땅에서 몇 cm 높이에 있게 될까?"를 계산한다.<br>

```
def legIK(self, point):
    # point: 어깨 기준으로 발이 어디 있는지 (x, y, z)
    # l1, l2, l3, l4: 다리 뼈 길이들
```
작동 원리 (삼각함수 & 코사인 법칙):<br>
사람 다리랑 똑같다. 발을 엉덩이 쪽으로 당기려면 무릎을 더 많이 굽혀야 한다.<br>
l1, l2...는 로봇 다리 부품의 길이다. 변하지 않는다.<br>
피타고라스 정리와 코사인 제2법칙(acos)을 써서, 발이 특정 좌표에 닿으려면 뼈와 뼈 사이의 각도(theta1, theta2, theta3)가 몇 도가 되어야 하는지 역산한다.<br>
요약: "어깨로부터 10cm 아래, 5cm 앞에 발을 두려면 무릎은 45도, 허벅지는 30도 꺾어야 함."<br>

```
def calcIK(self, Lp, angles, center):
    # Lp: 목표로 하는 발바닥 위치 (4개)
    # angles, center: 몸통 자세와 위치
```
핵심 로직:<br>
먼저 bodyIK를 불러서 현재 몸통 자세에 따른 어깨 위치를 구함. (Tlf, Trf...)<br>
그다음 상대 좌표 변환.<br>
np.linalg.inv(Tlf).dot(Lp[0]) 이 부분이 마법임.<br>
"발바닥 위치(Lp)는 그대로 고정하고 싶은데, 몸통(어깨)이 움직여버렸네? 그럼 어깨 입장에서 발은 어디에 있는 셈이지?" 를 계산.<br>
그 상대 위치를 legIK에 넣어서 각도를 구함.<br>

## 결론
로봇이 "몸통을 낮춰라(Height Down)" 명령을 받는다.<br>
bodyIK: 어깨 위치가 아래로 내려갔다고 계산한다.<br>
calcIK: 발은 땅에 붙어 있어야 하는데 어깨가 내려왔으니, 어깨 입장에서는 발이 위로 올라온 것과 똑같다.<br>
legIK: "어깨 입장에서 발이 위로 올라왔네? 그럼 무릎을 더 굽혀야겠군." -> 모터 각도 변경.<br>
결과: 로봇이 무릎을 굽히면서 몸이 낮아진다.<br>

main 함수에서 control_pitch, height, com_offset_x 값을 바꿈.<br>
그 값들이 calcIK 함수의 angles(회전)와 center(위치) 파라미터로 들어감.<br>
calcIK는 변한 몸통 위치에 맞춰서, 발을 땅에 고정하기 위해 다리를 어떻게 뻗거나 접어야 할지 계산.<br>
그래서 숫자만 바꿨는데도 로봇이 알아서 균형을 잡고 자세를 바꾼 것.<br>
