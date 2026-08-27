import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from math import *

# --- 1. Kinematic Class ---
class Kinematic:
    def __init__(self):
        # 로봇 기구학적 길이 (mm)
        self.l1 = 50
        self.l2 = 20
        self.l3 = 100
        self.l4 = 100
        self.L = 140
        self.W = 75
        
        self.LEG_FRONT = 0
        self.LEG_BACK = 2
        self.LEG_LEFT = 0
        self.LEG_RIGHT = 1

    # 몸통(Body)이 움직이거나 회전할 때, 네 다리의 시작점인 어깨(Shoulder)가 각각 어디로 가야 하는지
    def bodyIK(self, omega, phi, psi, xm, ym, zm):
        # 1. 회전 행렬(Rotation Matrix) 생성
        # Rx, Ry, Rz는 각각 X, Y, Z축을 기준으로 몸통을 얼마나 기울일지 결정함
        # omega(Roll), phi(Pitch), psi(Yaw) 각도를 행렬로 변환하는 과정
        Rx = np.array([[1,0,0,0],
                    [0,np.cos(omega),-np.sin(omega),0],
                    [0,np.sin(omega),np.cos(omega),0],[0,0,0,1]])
                    
        Ry = np.array([[np.cos(phi),0,np.sin(phi),0],
                    [0,1,0,0],
                    [-np.sin(phi),0,np.cos(phi),0],[0,0,0,1]])
                    
        Rz = np.array([[np.cos(psi),-np.sin(psi),0,0],
                    [np.sin(psi),np.cos(psi),0,0],[0,0,1,0],[0,0,0,1]])
        
        # Rx, Ry, Rz를 모두 곱해서 하나의 통합 회전 행렬(Rxyz)을 만듦
        Rxyz = Rx.dot(Ry.dot(Rz))

        # 2. 이동 행렬(Translation Matrix) 생성 및 결합
        # xm, ym, zm만큼 몸통을 통째로 이동시키고, 위에서 만든 회전(Rxyz)을 더함
        # Tm은 몸통 전체의 움직임을 나타내는 최종 '변환 행렬'이 됨
        T = np.array([[0,0,0,xm],[0,0,0,ym],[0,0,0,zm],[0,0,0,0]])
        Tm = T + Rxyz

        # 3. 로봇의 규격 설정 (L: 몸통 길이, W: 몸통 폭)
        sHp = np.sin(pi/2)
        cHp = np.cos(pi/2)
        (L, W) = (self.L, self.W)

        # 4. 각 다리 어깨 좌표 계산 및 반환
        # Tm(몸통 움직임)에 각 다리의 기본 위치(앞뒤 L/2, 좌우 W/2)를 곱함(dot)
        # 결과값은 4개 다리 어깨의 [위치 + 방향] 정보를 담은 리스트
        return([
            Tm.dot(np.array([[cHp,0,sHp, L/2],[0,1,0,0],[-sHp,0,cHp, W/2],[0,0,0,1]])), # 왼쪽 앞(LF)
            Tm.dot(np.array([[cHp,0,sHp, L/2],[0,1,0,0],[-sHp,0,cHp,-W/2],[0,0,0,1]])), # 오른쪽 앞(RF)
            Tm.dot(np.array([[cHp,0,sHp,-L/2],[0,1,0,0],[-sHp,0,cHp, W/2],[0,0,0,1]])), # 왼쪽 뒤(LB)
            Tm.dot(np.array([[cHp,0,sHp,-L/2],[0,1,0,0],[-sHp,0,cHp,-W/2],[0,0,0,1]]))  # 오른쪽 뒤(RB)
        ])

    # 목표 발 위치(x, y, z)를 주면, 그 위치에 발을 갖다 놓기 위해 **관절을 몇 도씩 꺾어야 하는지(theta1, 2, 3)**를 거꾸로 계산해내는 과정
    def legIK(self, point):
        # 1. 입력받은 발의 목표 좌표 (x, y, z)와 로봇 다리 마디 길이들
        (x, y, z) = (point[0], point[1], point[2])
        (l1, l2, l3, l4) = (self.l1, self.l2, self.l3, self.l4)

        # 2. 어깨 관절(Hip) 계산을 위한 보조 변수 F 계산
        # l1(어깨 폭)을 고려해서 다리가 닿을 수 있는 평면상의 거리를 구함
        try:        
            F = sqrt(x**2 + y**2 - l1**2)
        except ValueError:
            # 수학적으로 계산 불가능한 위치(너무 가까움)일 경우 에러 방지
            F = l1

        # 3. 허벅지와 종아리가 움직일 수 있는 유효 거리(G, H) 계산
        G = F - l2  # 사이드 마디(l2)를 뺀 실제 다리 길이 방향 거리
        H = sqrt(G**2 + z**2) # 피타고라스 정리를 이용한 어깨~발끝 직선 거리

        # 4. theta1: 어깨 관절(Hip) 각도 계산
        # 발의 x, y 위치를 보고 다리를 옆으로 얼마나 벌릴지 결정함
        theta1 = -atan2(y, x) - atan2(F, -l1)
        
        # 5. theta3: 무릎 관절 각도 계산 (코사인 법칙 활용)
        # 다리 전체 길이(H)와 각 마디(l3, l4) 길이를 이용해 무릎을 얼마나 굽힐지 계산
        D = (H**2 - l3**2 - l4**2) / (2 * l3 * l4)
        try:        
            # np.clip은 계산 값이 -1~1 사이를 벗어나 에러가 나는 것을 방지함
            theta3 = acos(np.clip(D, -1.0, 1.0)) 
        except ValueError:
            theta3 = 0

        # 6. theta2: 고관절(Thigh) 각도 계산
        # 무릎이 굽혀진 상태(theta3)를 고려해서 다리 전체를 얼마나 들지 결정
        theta2 = atan2(z, G) - atan2(l4 * sin(theta3), l3 + l4 * cos(theta3))
        
        # 최종적으로 계산된 세 개의 관절 각도를 반환
        return (theta1, theta2, theta3)

    # **3개의 관절 각도($\theta_1, \theta_2, \theta_3$)**를 가지고, 실제 로봇 다리의 마디마디(Joints) 좌표를 계산하는 함수
    def calcLegPoints(self, angles):
        # 1. 로봇 다리의 각 부분 길이 (l1: 어깨쪽, l2: 사이드, l3: 허벅지, l4: 종아리)
        (l1, l2, l3, l4) = (self.l1, self.l2, self.l3, self.l4)
        
        # 2. 계산된 관절 각도들
        # theta1: 힙(Hip) 회전, theta2: 허벅지 각도, theta3: 무릎 각도
        (theta1, theta2, theta3) = angles
        
        # theta23은 허벅지와 종아리 각도의 합 (최종 발끝 방향을 알기 위함)
        theta23 = theta2 + theta3

        # --- 좌표 계산 (삼각함수를 이용한 순운동학, Forward Kinematics) ---

        # T0: 다리의 시작점 (어깨 부착점, 0,0,0)
        T0 = np.array([0, 0, 0, 1])

        # T1: 첫 번째 관절 (Hip 조인트)
        # l1 길이만큼 theta1 각도로 회전한 지점
        T1 = T0 + np.array([-l1*cos(theta1), l1*sin(theta1), 0, 0])

        # T2: 두 번째 관절 (어깨에서 옆으로 튀어나온 마디 끝)
        # l2(오프셋)만큼 이동한 지점
        T2 = T1 + np.array([-l2*sin(theta1), -l2*cos(theta1), 0, 0])

        # T3: 세 번째 관절 (무릎 조인트)
        # 허벅지 길이(l3)와 각도(theta2)를 반영하여 위/아래/앞/뒤 위치 계산
        T3 = T2 + np.array([-l3*sin(theta1)*cos(theta2), 
                            -l3*cos(theta1)*cos(theta2), 
                            l3*sin(theta2), 0])

        # T4: 최종 목적지 (발끝, Foot-tip)
        # 종아리 길이(l4)와 무릎 꺾임 각도(theta23)를 반영하여 최종 좌표 확정
        T4 = T3 + np.array([-l4*sin(theta1)*cos(theta23), 
                            -l4*cos(theta1)*cos(theta23), 
                            l4*sin(theta23), 0])
                
        # 계산된 5개의 점(T0~T4)을 배열로 묶어서 반환
        # 이 점들을 선으로 이으면 우리가 아는 로봇 다리 모양이 됨
        return np.array([T0, T1, T2, T3, T4])

    # 모든 수학적 계산(BodyIK, LegIK, calcLegPoints)을 하나로 합쳐서 화면에 시각화
    # BodyIK로 몸통 기울기를 잡고
    # LegIK로 다리 꺾임을 계산해서
    # Scatter/Plot으로 화면에 점과 선을 찍어줌.
    def drawRobot(self, ax, Lp, angles, center):
        # 1. 몸통의 회전(angles)과 위치(center) 정보 추출
        (omega, phi, psi) = angles
        (xm, ym, zm) = center

        # 2. 몸통 역운동학(bodyIK)을 통해 4개 어깨의 변환 행렬 계산
        (Tlf, Trf, Tlb, Trb) = self.bodyIK(omega, phi, psi, xm, ym, zm)

        # 4개 다리의 정보를 리스트에 담아 반복문 준비
        transforms = [Tlf, Trf, Tlb, Trb]
        
        # 3. 좌/우 반전을 위한 Mirror 행렬 (Ix)
        # X축 값을 반전시켜서 오른쪽 다리가 왼쪽과 반대 방향으로 뻗게 함
        Ix = np.array([[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        
        for i, T in enumerate(transforms):
            # 목표 발 위치 (아까 원형 궤적으로 계산한 그 좌표)
            target_pos = Lp[i]
            
            # 4. 다리 각도 계산 로직
            if i % 2 == 1: # 홀수 번호(1, 3)는 오른쪽 다리 (FR, RR)
                # [오른쪽 다리 로직]
                # 1) 목표 위치를 로봇 어깨 좌표계로 가져옴 (inv_T)
                # 2) 오른쪽은 왼쪽의 반대이므로 Ix를 곱해 미러링 적용
                inv_T = Ix.dot(np.linalg.inv(T).dot(target_pos))
                # 3) 역운동학으로 관절 각도 계산
                leg_angles = self.legIK(inv_T)
                # 4) 순운동학으로 다리 마디 점(T0~T4) 계산
                points = self.calcLegPoints(leg_angles)
                # 5) 다시 월드 좌표(전체 공간)로 돌려놓을 때도 Ix 적용
                final_points = [T.dot(Ix.dot(p)) for p in points]
            
            else: # 짝수 번호(0, 2)는 왼쪽 다리 (FL, RL)
                # [왼쪽 다리 로직] 미러링 없이 계산
                inv_T = np.linalg.inv(T).dot(target_pos)
                leg_angles = self.legIK(inv_T)
                points = self.calcLegPoints(leg_angles)
                final_points = [T.dot(p) for p in points]

            # 5. 시각화 (Matplotlib 3D 그래프에 그리기)
            # 중요: Matplotlib과 로봇의 축 방향이 다르기 때문에 Y와 Z를 바꿔서 출력함
            xs = [p[0] for p in final_points]
            ys = [p[2] for p in final_points] # 로봇의 Z(좌우) -> 그래프의 Y
            zs = [p[1] for p in final_points] # 로봇의 Y(상하) -> 그래프의 Z
            
            # 다리 뼈대(검은 선), 어깨(파란 점), 발끝(빨간 점) 그리기
            ax.plot(xs, ys, zs, 'k-', lw=2) 
            ax.scatter(xs[0], ys[0], zs[0], color='b', s=50) 
            ax.scatter(xs[-1], ys[-1], zs[-1], color='r', s=50)

# --- 2. Animation Setup ---
def update_graph(num, ax, kin):
    ax.clear()
    
    # 3D 뷰 설정
    limit = 200
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_xlabel("X (Forward)")
    ax.set_ylabel("Z (Side)")
    ax.set_zlabel("Y (Up/Down)")
    
    # --- 원형 궤적 생성 (Circular Trajectory) ---
    t = num * 0.15  # 속도 조절
    radius = 25.0   # 반지름 (너무 크면 IK 에러 발생 가능)
    base_y = -120.0 # 로봇 높이
    
    feet_pos = []
    
    # 각 다리의 초기 X, Z 위치
    # X: 앞뒤, Z: 좌우
    # X축 (첫 번째 값): 로봇의 앞뒤 방향
    # Z축 (두 번째 값): 로봇의 좌우 방향
    # 원의 중심이된다
    default_xz = [
        [100, 75],    # 왼쪽 앞다리 (앞으로 10cm, 왼쪽으로 7.5cm)
        [100, -75],   # 오른쪽 앞다리 (앞으로 10cm, 오른쪽으로 7.5cm)
        [-100, 75],   # 왼쪽 뒷다리 (뒤로 10cm, 왼쪽으로 7.5cm)
        [-100, -75]   # 오른쪽 뒷다리 (뒤로 10cm, 오른쪽으로 7.5cm)
    ]
    
    for i in range(4):
        # ==========================================================
        # 1. 보행 방식
        # 첫번째 방법
        # 대각선 다리끼리 위상을 맞춤 (0번(왼쪽 앞다리)-3번(오른쪽 뒷다리) / 1번(오른쪽 앞다리)-2번(왼쪽 뒷다리))
        # phase = 0: 원형 궤적의 시작점(0도)에서 출발.
        # phase = np.pi: 180도 뒤처진 지점에서 출발.
        # 0번과 3번 다리가 원의 가장 높은 곳에 있을 때, 1번과 2번 다리는 정확히 반대편인 원의 가장 낮은 곳에 있게됨
        phase = 0 if (i == 0 or i == 3) else np.pi

        # 두번째 방법
        # 네 다리가 따로 움직임 - 실제 로봇이라면 이렇게 천천히 한 다리씩 움직일 때 세 다리가 지면에 붙어있어야 넘어지지 않음
        # 4개 다리가 90도씩 차이나게 설정 (0, 90, 180, 270도)
        # i는 0(FL), 1(FR), 2(RL), 3(RR) 순서임
        # phase = i * (np.pi / 2)

        # 세번째 방법
        # 네 다리가 한꺼번에 앞으로 뻗었다가 한꺼번에 뒤로 당겨진다 - Jump
        # 모든 다리의 위상차를 0으로 설정
        # phase = 0
        # ==========================================================
        
        # ==========================================================
        # 2. 궤적 공식
        # 옆면(X-Y 평면)에서 봤을 때 원을 그리도록 설정
        # radius (반지름): 원의 크기. 이 값이 커지면 로봇이 다리를 더 크게 휘두르고, 작아지면 좁게 움직임
        # t (시간): 시뮬레이션이 시작된 후 흐른 시간. 시간이 흐를수록 angle이 커지면서 발끝이 원을 따라 회전
        # phase (위상): 시작 지점. 이 값이 0이면 원의 오른쪽 끝에서 출발하고, pi이면 왼쪽 끝에서 출발

        # 첫번째 방법 - 원형
        dx = radius * np.cos(t + phase) # 원의 좌우(앞뒤) 위치를 결정
        dy = radius * np.sin(t + phase) # 원의 상하(높이) 위치를 결정

        # 두번째 방법 - 타원
        # 옆으로 긴 타원 (앞뒤 30mm, 위아래 10mm)
        # radius_x = 30.0 
        # radius_y = 10.0
        # dx = radius_x * np.cos(t + phase)
        # dy = radius_y * np.sin(t + phase)

        # 세번째 방법 - 노젓는 궤적 (앞뒤로만 움직임)
        # dx = radius * np.cos(t + phase)
        # dy = 0  # 높이 변화 없음!

        # 네번째 방법 - 삼각함수 위치바꾸기 (로봇이 뒤로 걷게됨)
        # 반대로 걷기 (역회전)
        # dx = radius * np.sin(t + phase)
        # dy = radius * np.cos(t + phase)
        
        # 컴퓨터는 t(시간)가 흐를 때마다 각도를 조금씩 올림. (0도 -> 10도 -> 20도...)
        # **cos(0도)**는 1임. 그래서 처음엔 dx가 가장 먼 곳에 가 있다.
        # **sin(0도)**는 0. 그래서 dy는 중간 높이에 있다.
        # 각도가 90도가 되면 cos는 0이 되고, sin은 1이 됨.
        # 즉, 발이 **"가운데로 오면서 동시에 위로 번쩍 들린 상태"**가 됨.
        # 이 과정이 무한 반복되면서 발끝이 동그란 선을 그리게 된다.
        # ==========================================================
        
        # ==========================================================
        # 3. **움직임 변화량(dx, dy)**을 로봇의 실제 위치 좌표로 합치는 과정

        # 앞뒤 위치
        # default_xz[i][0]: 로봇 다리의 원래 앞뒤 기준점, dx: 원을 그리기 위해 앞뒤로 왔다 갔다 하는 변화량
        # 다리가 기준점(100)을 중심으로 앞뒤로 뱅글뱅글 돌음
        px = default_xz[i][0] + dx

        # 앞다리든 뒷다리든 상관없이 모든 다리가 로봇 몸통 정중앙(0)으로 모여서 원을 그림
        # px = dx

        # 높이 위치
        # base_y: 로봇 다리가 지면에서 들려 있는 기본 높이, dy: 원을 그리기 위해 위아래로 들썩이는 변화량
        # 다리가 기본 높이(-120)를 중심으로 위아래로 움직임
        py = base_y + dy

        # 다리가 앞뒤로만 움직임
        # py = base_y

        # 좌우 위치
        # default_xz[i][1]: 로봇 다리의 좌우 폭
        # 다리가 옆으로는 안 벌어지고 딱 고정된 채로 앞뒤/위아래로만 원을 그림
        pz = default_xz[i][1]

        # 다리가 앞뒤가 아니라 옆으로 뱅글뱅글 돌음
        # pz = default_xz[i][1] + dx
        # ==========================================================
        
        # 발의 좌표(X, Y, Z)를 하나의 덩어리(벡터)로 묶어서 리스트에 담기
        feet_pos.append(np.array([px, py, pz, 1]))

    # IK 계산 및 그리기 호출
    # (0,0,0) (0,0,0)은 몸체의 회전과 위치가 원점에 고정됨을 의미
    kin.drawRobot(ax, feet_pos, (0,0,0), (0,0,0))

    # 로봇이 옆으로 갸우뚱한 상태로 원형 궤적 그림
    # kin.drawRobot(ax, feet_pos, (0.2, 0, 0), (0, 0, 0))

    # 로봇 몸통이 위로 50mm 붕 뜬 상태에서 발을 뻗으려고 시도
    # kin.drawRobot(ax, feet_pos, (0, 0, 0), (0, 50, 0))
    
    ax.set_title(f"IK Circular Trajectory (t={t:.2f})")

def main():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    kin = Kinematic()
    
    # 애니메이션 실행 (Frames=100, Interval=50ms)
    ani = FuncAnimation(fig, update_graph, frames=200, fargs=(ax, kin), interval=50)
    
    print("Matplotlib 애니메이션 시작...")
    plt.show()

if __name__ == "__main__":
    main()
