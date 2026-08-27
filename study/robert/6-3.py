"""
6-3.py: Matplotlib 제자리 걷기 애니메이션

Week 06 - 4.3절 코드를 통합한 완전한 실행 파일
TrottingGait 클래스를 사용하여 Matplotlib으로 발끝 궤적을 애니메이션합니다.

실행 방법:
    cd study/robert
    python 6-3.py
"""

import sys
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle

# 한글 폰트 설정 (Windows)
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass



class SimpleTrottingGait:
    """
    TrottingGait의 간소화된 독립 구현
    """
    def __init__(self):
        # 시간 파라미터 (밀리초)
        self.t0 = 300   # State 0: 지면 대기
        self.t1 = 1200  # State 1: 지면 이동 (Stance)
        self.t2 = 300   # State 2: 지면 대기
        self.t3 = 200   # State 3: 발 들어올림 (Swing)
        
        # 거리 파라미터
        self.Sl = 0.0   # Step Length (보폭)
        self.Sw = 0     # Step Width
        self.Sh = 40    # Step Height (발 들어올리는 높이)
        self.Sa = 0     # Step Alpha (회전)
        
        # 다리 오프셋
        self.Spf = 87   # Front leg spur width
        self.Spr = 77   # Rear leg spur width
        self.Fo = 120   # Front leg X offset
        self.Ro = 50    # Rear leg X offset

    def calcLeg(self, t, x, y, z):
        """단일 다리의 발끝 위치 계산"""
        startLp = np.array([x - self.Sl/2.0, y, z - self.Sw, 1])
        endLp = np.array([x + self.Sl/2, y, z + self.Sw, 1])
        
        if t < self.t0:
            return startLp
        elif t < self.t0 + self.t1:
            td = t - self.t0
            tp = td / self.t1 if self.t1 > 0 else 0
            diffLp = endLp - startLp
            curLp = startLp + diffLp * tp
            return curLp
        elif t < self.t0 + self.t1 + self.t2:
            return endLp
        elif t < self.t0 + self.t1 + self.t2 + self.t3:
            td = t - (self.t0 + self.t1 + self.t2)
            tp = td / self.t3 if self.t3 > 0 else 0
            diffLp = startLp - endLp
            curLp = endLp + diffLp * tp
            curLp[1] += self.Sh * math.sin(math.pi * tp)
            return curLp
        return startLp

    def positions(self, t, kb_offset={}):
        """4개 다리의 발끝 위치 계산"""
        spf = self.Spf
        spr = self.Spr
        
        if kb_offset:
            self.Sl = kb_offset.get('IDstepLength', 0.0)
            self.Sw = kb_offset.get('IDstepWidth', 0.0)
            self.Sa = kb_offset.get('IDstepAlpha', 0.0)
        
        Tt = self.t0 + self.t1 + self.t2 + self.t3
        Tt2 = Tt / 2
        
        td = (t * 1000) % Tt
        t2 = (t * 1000 - Tt2) % Tt
        
        Fx = self.Fo
        Rx = -self.Ro
        Fy = -100
        Ry = -100
        
        return np.array([
            self.calcLeg(td, Fx, Fy, spf),
            self.calcLeg(t2, Fx, Fy, -spf),
            self.calcLeg(t2, Rx, Ry, spr),
            self.calcLeg(td, Rx, Ry, -spr)
        ])


def animate_in_place_walking():
    """
    제자리 걷기 발끝 궤적 애니메이션
    """
    # SimpleTrottingGait 사용 (ZeroDivisionError 방지)
    trotting = SimpleTrottingGait()
    
    trotting.Sh = 40  # 발 들어올리는 높이

    kb_offset = {
        'IDstepLength': 0.0,
        'IDstepWidth': 0.0,
        'IDstepAlpha': 0.0
    }

    # Figure 생성
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('Week 06: 제자리 걷기 Trot Gait 애니메이션', fontsize=14, fontweight='bold')

    # 측면도 (X-Y, 높이)
    ax1 = fig.add_subplot(121)
    ax1.set_xlim(-200, 200)
    ax1.set_ylim(-180, -40)
    ax1.set_xlabel('X Position (mm)', fontsize=11)
    ax1.set_ylabel('Y Position (Height, mm)', fontsize=11)
    ax1.set_title('Side View - Foot Height')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_aspect('equal')
    
    # 지면 표시
    ax1.axhline(y=-100, color='brown', linestyle='-', linewidth=2, alpha=0.5, label='Ground')
    ax1.fill_between([-200, 200], [-180, -180], [-100, -100], color='burlywood', alpha=0.3)

    # 상단도 (X-Z)
    ax2 = fig.add_subplot(122)
    ax2.set_xlim(-200, 200)
    ax2.set_ylim(-150, 150)
    ax2.set_xlabel('X Position (mm)', fontsize=11)
    ax2.set_ylabel('Z Position (mm)', fontsize=11)
    ax2.set_title('Top View - Foot Positions')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_aspect('equal')
    
    # 몸체 표시 (상단도)
    body_rect = Rectangle((-70, -60), 140, 120, fill=True, 
                          facecolor='lightgray', edgecolor='black', linewidth=2, alpha=0.5)
    ax2.add_patch(body_rect)
    ax2.text(0, 0, 'BODY', ha='center', va='center', fontsize=10, fontweight='bold')

    # 다리 색상 및 이름
    colors = ['blue', 'red', 'green', 'orange']
    leg_names = ['FL', 'FR', 'RL', 'RR']
    leg_full_names = ['Front Left', 'Front Right', 'Rear Left', 'Rear Right']

    # 발끝 점 생성
    dots1 = [ax1.scatter([], [], c=c, s=150, label=n, zorder=5, edgecolors='black', linewidths=1)
             for c, n in zip(colors, leg_names)]
    dots2 = [ax2.scatter([], [], c=c, s=150, label=n, zorder=5, edgecolors='black', linewidths=1)
             for c, n in zip(colors, leg_names)]

    # 궤적 히스토리 저장
    trail_length = 30
    trails_x = [[] for _ in range(4)]
    trails_y = [[] for _ in range(4)]
    trail_lines1 = [ax1.plot([], [], c=c, alpha=0.3, linewidth=2)[0] for c in colors]

    ax1.legend(loc='upper right', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)

    # 시간 표시 텍스트
    time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=11,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 상태 표시
    status_texts = []
    for i, (color, name) in enumerate(zip(colors, leg_names)):
        txt = ax2.text(0.02, 0.98 - i*0.08, '', transform=ax2.transAxes, fontsize=9,
                       verticalalignment='top', color=color, fontweight='bold')
        status_texts.append(txt)

    def get_phase_name(t, phase_offset):
        """현재 시간에서 phase 이름 반환"""
        Tt = trotting.t0 + trotting.t1 + trotting.t2 + trotting.t3
        td = ((t * 1000) - phase_offset) % Tt
        
        if td < trotting.t0:
            return "Wait", "gray"
        elif td < trotting.t0 + trotting.t1:
            return "Stance", "red"
        elif td < trotting.t0 + trotting.t1 + trotting.t2:
            return "Wait", "gray"
        else:
            return "Swing", "blue"

    def update(frame):
        t = frame / 30.0  # 30 FPS

        positions = trotting.positions(t, kb_offset)
        
        # Phase offsets for each leg (Trot gait)
        Tt2 = (trotting.t0 + trotting.t1 + trotting.t2 + trotting.t3) / 2
        phase_offsets = [0, Tt2, Tt2, 0]  # FL, FR, RL, RR

        for i, (pos, dot1, dot2) in enumerate(zip(positions, dots1, dots2)):
            x, y, z = pos[0], pos[1], pos[2]
            
            # 측면도 (X, Y)
            dot1.set_offsets([[x, y]])
            # 상단도 (X, Z)
            dot2.set_offsets([[x, z]])
            
            # 궤적 히스토리 업데이트
            trails_x[i].append(x)
            trails_y[i].append(y)
            if len(trails_x[i]) > trail_length:
                trails_x[i].pop(0)
                trails_y[i].pop(0)
            trail_lines1[i].set_data(trails_x[i], trails_y[i])
            
            # 상태 표시 업데이트
            phase_name, phase_color = get_phase_name(t, phase_offsets[i])
            status_texts[i].set_text(f"{leg_names[i]}: {phase_name} (Y={y:.0f})")

        # 시간 표시 업데이트
        time_text.set_text(f'Time: {t:.2f}s\nStep Height: {trotting.Sh}mm')

        return dots1 + dots2 + trail_lines1 + [time_text] + status_texts

    # 애니메이션 생성
    anim = FuncAnimation(fig, update, frames=300, interval=33, blit=True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

    return anim


def visualize_gait_cycle():
    """
    한 Gait Cycle의 발끝 궤적을 정적으로 시각화
    """
    # SimpleTrottingGait 사용 (ZeroDivisionError 방지)
    trotting = SimpleTrottingGait()
    
    trotting.Sh = 40

    kb_offset = {
        'IDstepLength': 0.0,
        'IDstepWidth': 0.0,
        'IDstepAlpha': 0.0
    }

    # 한 주기 동안의 궤적 계산
    Tt = (trotting.t0 + trotting.t1 + trotting.t2 + trotting.t3) / 1000  # 초 단위
    times = np.linspace(0, Tt, 100)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Trot Gait - One Cycle Foot Trajectories (제자리 걷기)', fontsize=14, fontweight='bold')
    
    leg_names = ['Front Left (FL)', 'Front Right (FR)', 'Rear Left (RL)', 'Rear Right (RR)']
    colors = ['blue', 'red', 'green', 'orange']
    
    for idx, (ax, name, color) in enumerate(zip(axes.flatten(), leg_names, colors)):
        x_vals = []
        y_vals = []
        
        for t in times:
            positions = trotting.positions(t, kb_offset)
            x_vals.append(positions[idx][0])
            y_vals.append(positions[idx][1])
        
        # 궤적 그리기
        ax.plot(x_vals, y_vals, color=color, linewidth=2, label='Trajectory')
        ax.scatter(x_vals[0], y_vals[0], color='green', s=100, zorder=5, label='Start', marker='o')
        ax.scatter(x_vals[-1], y_vals[-1], color='red', s=100, zorder=5, label='End', marker='x')
        
        # Swing phase 강조 (Y가 높은 부분)
        swing_mask = np.array(y_vals) > -100 + 5
        if np.any(swing_mask):
            swing_indices = np.where(swing_mask)[0]
            ax.plot(np.array(x_vals)[swing_mask], np.array(y_vals)[swing_mask], 
                    color='cyan', linewidth=4, alpha=0.5, label='Swing Phase')
        
        # 지면 표시
        ax.axhline(y=-100, color='brown', linestyle='--', linewidth=1, alpha=0.7, label='Ground')
        
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('Y Position (Height, mm)')
        ax.set_title(name)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()


def main():
    """메인 함수"""
    print("=" * 60)
    print("Week 06: Matplotlib 제자리 걷기 애니메이션")
    print("=" * 60)
    print()
    print("1. 정적 궤적 시각화 (한 주기)")
    print("2. 실시간 애니메이션")
    print()
    
    # 1. 정적 궤적 시각화
    print("[1/2] 한 주기 궤적 시각화...")
    visualize_gait_cycle()
    
    # 2. 실시간 애니메이션
    print("[2/2] 실시간 애니메이션 시작...")
    print("(창을 닫으면 종료됩니다)")
    animate_in_place_walking()
    
    print()
    print("시각화 완료!")


if __name__ == "__main__":
    main()
