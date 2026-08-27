'''
다양한 보행 패턴
Includes Walking Mechanism for quadruped robot
'''

import time
import numpy as np
import math
import pybullet as p

class KinematicLegMotion:

    def __init__(self,LLp):
        self.rtime=time.time()
        self.running=False
        self.LLp=LLp

    def moveTo(self,newLLp,rtime,func=None):
        if self.running:
            # TODO: Queue the Requests
            print("Movement already running, please try again later.")
            return False
        self.startTime=time.time()
        self.startLLp=self.LLp
        self.func=func
        self.targetLLp=newLLp
        self.endTime=time.time()+rtime/1000
        self.running=True
        return True
    
    def update(self):
        diff=time.time()-self.startTime
        ldiff=self.targetLLp-self.startLLp
        tdiff=self.endTime-self.startTime
        ldiff/(tdiff*diff)
        p=1/tdiff*diff

        if time.time()>self.endTime and self.running:
            self.running=False
            p=1
        self.LLp=self.startLLp+ldiff*p
        if self.func:
            self.LLp=self.func(p,self.LLp)

    def step(self):
        if self.running:
            self.update()
        return self.LLp

class KinematicMotion:

    def __init__(self,Lp):
        self.Lp=Lp
        self.legs=[KinematicLegMotion(Lp[x]) for x in range(4)]

    def moveLegsTo(self,newLp,rtime):
        [self.legs[x].moveTo(newLp[x],rtime) for x in range(4)]

    def moveLegTo(self,leg,newLLp,rtime,func=None):
        return self.legs[leg].moveTo(newLLp,rtime,func)

    def step(self):
        return [x.step() for x in self.legs]


"""
This class will define the trotting-gait function
A complete cycle is tone in Tt
Each leg has the following "states"
0 - wait on ground for t0
1 - move on ground for steplength Sl for t1
2 - wait on ground for t2
3 - lift leg by Sh and Sl for t3 back to 0
"""
class TrottingGait:
    
    def __init__(self):
        self.maxSl=2
        self.bodyPos=(0,100,0)
        self.bodyRot=(0,0,0)
        self.t0=0 # senseless i guess
        self.t1=510
        self.t2=00
        self.t3=185
        self.Sl=0.0
        self.Sw=0
        self.Sh=60
        self.Sa=0
        self.Spf=87
        self.Spr=77

        # gait mode 추가
        self.IDgaitMode = p.addUserDebugParameter("Gait Mode (0:Trot, 1:Pace, 2:Bound, 3:Walk)", 0, 4, 0)
        self.IDspurFront= p.addUserDebugParameter("spur front", 20, 150, self.Spf)
        self.IDspurRear= p.addUserDebugParameter("spur rear", 20, 150, self.Spr)
        self.IDstepLength = p.addUserDebugParameter("step length", -150, 150, self.Sl)
        self.IDstepWidth = p.addUserDebugParameter("step width", -150, 150, self.Sw)
        self.IDstepHeight = p.addUserDebugParameter("step height", 0, 150, self.Sh)
        self.IDstepAlpha = p.addUserDebugParameter("step alpha", -90, 90, self.Sa)
        self.IDt0 = p.addUserDebugParameter("t0", 0, 1000, self.t0)
        self.IDt1 = p.addUserDebugParameter("t1", 0, 1000, self.t1)
        self.IDt2 = p.addUserDebugParameter("t2", 0, 1000, self.t2)
        self.IDt3 = p.addUserDebugParameter("t3", 0, 1000, self.t3)
        self.IDfrontOffset = p.addUserDebugParameter("front Offset", 0,200, 120)
        self.IDrearOffset = p.addUserDebugParameter("rear Offset", 0,200, 50)
        
        self.Rc=[-50,0,0,1] # rotation center


    """
    calculates the Lp - LegPosition for the configured gait for time t and original Lp of x,y,z
    """
    def calcLeg(self,t,x,y,z):
        startLp=np.array([x-self.Sl/2.0,y,z-self.Sw/2,1])
        endLp=np.array([x+self.Sl/2,y,z+self.Sw/2,1])
        
        if(t<self.t0): # stay on ground
            print("stay")
            return startLp
        elif(t<self.t0+self.t1): # drag foot over ground
            print("drag")
            td=t-self.t0
            tp=1/(self.t1/td)
            diffLp=endLp-startLp
            curLp=startLp+diffLp*tp

            psi=(math.pi/180*self.Sa)*tp
            Ry = np.array([[np.cos(psi),0,np.sin(psi),0],
                    [0,1,0,0],
                    [-np.sin(psi),0,np.cos(psi),0],[0,0,0,1]])
            curLp=Ry.dot(curLp)
            return curLp
        elif(t<self.t0+self.t1+self.t2): # stay on ground again
            print("stay")
            return endLp
        elif(t<self.t0+self.t1+self.t2+self.t3): # Lift foot
            print("lift")
            td=t-(self.t0+self.t1+self.t2)
            tp=1/(self.t3/td)
            diffLp=startLp-endLp
            curLp=endLp+diffLp*tp
            curLp[1]+=self.Sh*math.sin(math.pi*tp)
            return curLp
            
    def stepLength(self,len):
        self.Sl=len

    def positions(self, t, kb_offset={}):
        # 1. 파라미터 읽기
        spf = p.readUserDebugParameter(self.IDspurFront)
        spr = p.readUserDebugParameter(self.IDspurRear)
        self.Sh = p.readUserDebugParameter(self.IDstepHeight)
        
        # 보행 모드 읽기 (반올림하여 정수로 사용)
        gait_mode = int(p.readUserDebugParameter(self.IDgaitMode))

        # Pybullet 파라미터 적용
        if list(kb_offset.values()) == [0.0, 0.0, 0.0]:
            self.Sl = p.readUserDebugParameter(self.IDstepLength)
            self.Sw = p.readUserDebugParameter(self.IDstepWidth)
            self.Sa = p.readUserDebugParameter(self.IDstepAlpha)
        else:
            self.Sl = kb_offset['IDstepLength']
            self.Sw = kb_offset['IDstepWidth']
            self.Sa = kb_offset['IDstepAlpha']

        # 시간 파라미터
        self.t0 = p.readUserDebugParameter(self.IDt0)
        self.t1 = p.readUserDebugParameter(self.IDt1)
        self.t2 = p.readUserDebugParameter(self.IDt2)
        self.t3 = p.readUserDebugParameter(self.IDt3)

        # 전체 주기 계산
        Tt = (self.t0 + self.t1 + self.t2 + self.t3)
        if Tt == 0: Tt = 1000 # 0 나누기 방지 안전장치

        # ---------------------------------------------------------
        # [핵심] 보행 패턴별 위상차(Phase Offsets) 정의
        # offsets 배열 순서: [FL(앞왼), FR(앞오), RL(뒤왼), RR(뒤오)]
        # 값의 의미: 주기의 몇 배만큼 늦게 시작할 것인가 (0.0 ~ 1.0)
        # ---------------------------------------------------------
        
        # Mode 0: Trot (트롯) - 대각선 동기화
        # FL(0)과 RR(0)이 같이, FR(0.5)과 RL(0.5)이 같이 움직임
        if gait_mode == 0:
            offsets = [0.0, 0.5, 0.5, 0.0] 

        # Mode 1: Pace (페이스) - 좌우 동기화
        # 왼쪽(FL, RL)이 같이, 오른쪽(FR, RR)이 같이 움직임 (불안정할 수 있음)
        elif gait_mode == 1:
            offsets = [0.0, 0.5, 0.0, 0.5]

        # Mode 2: Bound (바운드) - 앞뒤 동기화 (토끼뜀)
        # 앞다리(FL, FR)가 같이, 뒷다리(RL, RR)가 같이 움직임
        elif gait_mode == 2:
            offsets = [0.0, 0.0, 0.5, 0.5]

        # Mode 3: Free / Walk (자유/평보) - 4박자 분리 (고양이 걸음)
        # 순서: FL -> RR -> FR -> RL (가장 안정적)
        elif gait_mode == 3:
            offsets = [0.0, 0.5, 0.75, 0.25]

        # Mode 4: Default (기본) - 보통 Trot이나 Stand로 둠
        else:
            offsets = [0.0, 0.5, 0.5, 0.0] # Trot과 동일하게 설정

        # ---------------------------------------------------------

        # 2. 각 다리별 시간 계산 (위상차 적용)
        # (현재시간 - (전체주기 * 오프셋)) % 전체주기
        t_legs = []
        for offset in offsets:
            t_val = (t * 1000 - Tt * offset) % Tt
            t_legs.append(t_val)

        # 오프셋 설정
        Fx = p.readUserDebugParameter(self.IDfrontOffset)
        Rx = -p.readUserDebugParameter(self.IDrearOffset)
        Fy = -100
        Ry = -100

        # 3. calcLeg 호출 및 결과 반환
        # 순서: [FL, FR, RL, RR] (offsets 배열 순서와 일치해야 함)
        r = np.array([
            self.calcLeg(t_legs[0], Fx, Fy, spf),   # FL
            self.calcLeg(t_legs[1], Fx, Fy, -spf),  # FR
            self.calcLeg(t_legs[2], Rx, Ry, spr),   # RL
            self.calcLeg(t_legs[3], Rx, Ry, -spr)   # RR
        ])
        
        return r
