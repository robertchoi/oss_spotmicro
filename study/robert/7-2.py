"""
7-2.py: CoM (Center of Mass) Tracking Visualization
Week 07 - CoM 추적 및 Support Polygon 시각화

이 코드는 Trot Gait 동안 CoM(무게 중심)과 Support Polygon(지지 다각형)의
변화를 Matplotlib 애니메이션으로 시각화합니다.

학습 목표:
- CoM (Center of Mass) 개념 이해
- Support Polygon과 안정성 관계 파악
- Trot Gait 시 phase에 따른 지지 다각형 변화 관찰
"""
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
from matplotlib.path import Path


class CoMCalculator:
    """
    Center of Mass 계산 클래스
    
    로봇의 무게 중심과 지지 다각형을 계산하고,
    안정성을 판단합니다.
    """
    
    def __init__(self):
        # 각 부분의 질량 (예시 값, kg)
        self.body_mass = 1.5       # 몸체
        self.shoulder_mass = 0.1   # 어깨
        self.upper_leg_mass = 0.1  # 상단 다리
        self.lower_leg_mass = 0.05 # 하단 다리
        
        # 총 질량
        self.leg_mass = self.shoulder_mass + self.upper_leg_mass + self.lower_leg_mass
        self.total_mass = self.body_mass + 4 * self.leg_mass
        
    def calculate_com(self, body_pos, foot_positions):
        """
        전체 로봇의 CoM 계산
        
        Parameters:
        -----------
        body_pos : tuple
            몸체 중심 위치 (x, y, z)
        foot_positions : numpy array
            4개 다리의 발끝 위치 [[x,y,z], ...]
        
        Returns:
        --------
        com : numpy array
            Center of Mass 위치 [x, y, z]
        """
        # 간단화된 계산: 몸체 + 다리 질량 기반 가중 평균
        com_x = body_pos[0] * self.body_mass
        com_y = body_pos[1] * self.body_mass
        com_z = body_pos[2] * self.body_mass
        
        for foot_pos in foot_positions:
            # 다리의 CoM은 발끝과 몸체 사이 중간 정도로 근사
            leg_com_x = (body_pos[0] + foot_pos[0]) / 2
            leg_com_z = (body_pos[2] + foot_pos[2]) / 2
            
            com_x += leg_com_x * self.leg_mass
            com_z += leg_com_z * self.leg_mass
        
        com_x /= self.total_mass
        com_z /= self.total_mass
        com_y = body_pos[1]  # Y는 높이이므로 그대로
        
        return np.array([com_x, com_y, com_z])
    
    def get_support_polygon(self, foot_positions, stance_mask):
        """
        지지 다각형 계산
        
        Parameters:
        -----------
        foot_positions : numpy array
            4개 다리의 발끝 위치
        stance_mask : list
            각 다리의 Stance 여부 [FL, FR, RL, RR]
            True = Stance (지면 접촉)
            False = Swing (공중)
        
        Returns:
        --------
        polygon : list
            지지 다각형의 꼭짓점 좌표 [(x, z), ...]
        """
        polygon = []
        for i, (pos, is_stance) in enumerate(zip(foot_positions, stance_mask)):
            if is_stance:
                polygon.append([pos[0], pos[2]])  # X, Z 좌표 (상단 뷰)
        
        # 다각형 꼭짓점 정렬 (시계 방향)
        if len(polygon) >= 3:
            polygon = self._sort_polygon_vertices(polygon)
        
        return polygon
    
    def _sort_polygon_vertices(self, vertices):
        """
        다각형 꼭짓점을 시계 방향으로 정렬
        """
        if len(vertices) < 3:
            return vertices
            
        # 중심점 계산
        center_x = sum(v[0] for v in vertices) / len(vertices)
        center_z = sum(v[1] for v in vertices) / len(vertices)
        
        # 각도 기준으로 정렬
        def angle_from_center(vertex):
            return np.arctan2(vertex[1] - center_z, vertex[0] - center_x)
        
        return sorted(vertices, key=angle_from_center)
    
    def is_stable(self, com, support_polygon):
        """
        CoM이 지지 다각형 내에 있는지 확인
        
        Parameters:
        -----------
        com : numpy array
            CoM 위치 [x, y, z]
        support_polygon : list
            지지 다각형 꼭짓점 [(x, z), ...]
        
        Returns:
        --------
        stable : bool
            안정성 여부
        margin : float
            안정성 마진 (다각형 중심까지의 거리)
        """
        if len(support_polygon) < 3:
            return False, 0.0
        
        polygon_path = Path(support_polygon)
        com_2d = [com[0], com[2]]  # X, Z 좌표
        
        is_inside = polygon_path.contains_point(com_2d)
        
        # 다각형 중심 계산
        center_x = sum(v[0] for v in support_polygon) / len(support_polygon)
        center_z = sum(v[1] for v in support_polygon) / len(support_polygon)
        
        # CoM에서 중심까지의 거리 (마진)
        margin = np.sqrt((com[0] - center_x)**2 + (com[2] - center_z)**2)
        
        return is_inside, margin


class TrottingGaitSimulator:
    """
    Trot Gait 시뮬레이터
    
    시간에 따라 다리의 phase 상태를 계산합니다.
    """
    
    def __init__(self):
        # Gait 파라미터 (ms)
        self.t0 = 300   # State 0: 대기
        self.t1 = 1200  # State 1: Stance
        self.t2 = 300   # State 2: 대기
        self.t3 = 200   # State 3: Swing
        
        # 총 주기
        self.Tt = self.t0 + self.t1 + self.t2 + self.t3  # 2000ms
        
        # 발 들어올리는 높이
        self.step_height = 40
        
    def get_stance_mask(self, t_ms):
        """
        시간에 따른 다리 상태 반환
        
        Returns:
        --------
        stance_mask : list
            [FL, FR, RL, RR] - True: Stance, False: Swing
        phase_info : dict
            각 다리의 phase 정보
        """
        # FL+RR과 FR+RL은 반 주기 차이
        t1 = t_ms % self.Tt
        t2 = (t_ms + self.Tt / 2) % self.Tt
        
        # State 판별 함수
        def get_state(t):
            if t < self.t0:
                return 0  # Wait on ground
            elif t < self.t0 + self.t1:
                return 1  # Ground move (Stance)
            elif t < self.t0 + self.t1 + self.t2:
                return 2  # Wait on ground
            else:
                return 3  # Swing
        
        state_fl = get_state(t1)  # FL
        state_rr = get_state(t1)  # RR (same as FL)
        state_fr = get_state(t2)  # FR
        state_rl = get_state(t2)  # RL (same as FR)
        
        # Swing phase (state 3)이면 False, 아니면 True (Stance)
        stance_mask = [
            state_fl != 3,  # FL
            state_fr != 3,  # FR
            state_rl != 3,  # RL
            state_rr != 3   # RR
        ]
        
        phase_info = {
            'FL': state_fl, 'FR': state_fr,
            'RL': state_rl, 'RR': state_rr
        }
        
        return stance_mask, phase_info


def visualize_com_tracking():
    """
    CoM 추적 시각화 메인 함수
    스페이스바를 눌러 다음 프레임으로 진행합니다.
    """
    com_calc = CoMCalculator()
    gait_sim = TrottingGaitSimulator()
    
    # 다리 기본 위치 (x, y, z) - 상단 뷰에서 X, Z 사용
    foot_positions = np.array([
        [120, -100, 87],    # FL
        [120, -100, -87],   # FR
        [-50, -100, 77],    # RL
        [-50, -100, -77]    # RR
    ])
    
    # 몸체 위치
    body_pos = (0, 100, 0)
    
    # 다리 색상 및 이름
    colors = ['blue', 'red', 'green', 'orange']
    leg_names = ['FL', 'FR', 'RL', 'RR']
    
    # Figure 설정
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 왼쪽: 상단 뷰 (X-Z)
    ax1 = axes[0]
    ax1.set_xlim(-200, 200)
    ax1.set_ylim(-150, 150)
    ax1.set_xlabel('X Position (mm) - Forward/Backward')
    ax1.set_ylabel('Z Position (mm) - Left/Right')
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 오른쪽: 상태 표시
    ax2 = axes[1]
    ax2.set_xlim(0, 4)
    ax2.set_ylim(0, 4)
    ax2.axis('off')
    
    # 현재 프레임 상태
    state = {'frame': 0, 'running': True}
    
    def draw_frame(frame):
        """프레임 그리기 함수"""
        # 시간 계산 (30 FPS 가정, ms 단위)
        t_ms = frame * 100  # 100ms per frame (더 느리게)
        
        # Phase 계산
        stance_mask, phase_info = gait_sim.get_stance_mask(t_ms)
        
        # CoM 계산
        com = com_calc.calculate_com(body_pos, foot_positions)
        
        # Support Polygon 계산
        support_poly = com_calc.get_support_polygon(foot_positions, stance_mask)
        
        # 안정성 확인
        is_stable, margin = com_calc.is_stable(com, support_poly)
        
        # 왼쪽 축 초기화
        ax1.clear()
        ax1.set_xlim(-200, 200)
        ax1.set_ylim(-150, 150)
        ax1.set_xlabel('X Position (mm) - Forward/Backward', fontsize=10)
        ax1.set_ylabel('Z Position (mm) - Left/Right', fontsize=10)
        ax1.set_aspect('equal')
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # 제목
        phase_str = "FL+RR" if stance_mask[0] and stance_mask[3] else "FR+RL"
        ax1.set_title(f'CoM Tracking - Stance: {phase_str}\n'
                     f'Time: {t_ms/1000:.2f}s | Frame: {frame}/60', fontsize=12)
        
        # 몸체 윤곽선 그리기
        body_vertices = [
            [70, 40], [70, -40], [-35, -40], [-35, 40], [70, 40]
        ]
        body_x = [v[0] for v in body_vertices]
        body_z = [v[1] for v in body_vertices]
        ax1.plot(body_x, body_z, 'k-', linewidth=2, label='Body')
        ax1.fill(body_x, body_z, 'lightgray', alpha=0.5)
        
        # Support Polygon 그리기
        if len(support_poly) >= 3:
            poly_closed = support_poly + [support_poly[0]]  # 닫힌 다각형
            poly_x = [v[0] for v in poly_closed]
            poly_z = [v[1] for v in poly_closed]
            
            poly_color = 'lightgreen' if is_stable else 'lightyellow'
            ax1.fill(poly_x, poly_z, poly_color, alpha=0.4, label='Support Polygon')
            ax1.plot(poly_x, poly_z, 'g-', linewidth=2)
        
        # 발끝 위치 그리기
        for i, (pos, color, name) in enumerate(zip(foot_positions, colors, leg_names)):
            is_stance = stance_mask[i]
            marker = 'o' if is_stance else 'x'
            size = 200 if is_stance else 150
            
            state_str = "Stance" if is_stance else "Swing"
            ax1.scatter(pos[0], pos[2], c=color, s=size, marker=marker, 
                       zorder=5, label=f'{name} ({state_str})')
            ax1.annotate(name, (pos[0], pos[2]), textcoords="offset points",
                        xytext=(10, 10), fontsize=9, color=color)
        
        # CoM 그리기
        stability_str = "STABLE" if is_stable else "UNSTABLE"
        com_color = 'green' if is_stable else 'red'
        ax1.scatter(com[0], com[2], c=com_color, s=400, marker='*', 
                   zorder=10, label=f'CoM ({stability_str})')
        ax1.annotate(f'CoM', (com[0], com[2]), textcoords="offset points",
                    xytext=(15, -15), fontsize=10, fontweight='bold', color=com_color)
        
        ax1.legend(loc='upper right', fontsize=8)
        
        # 오른쪽 축: 상태 정보 표시
        ax2.clear()
        ax2.set_xlim(0, 4)
        ax2.set_ylim(0, 5)
        ax2.axis('off')
        
        ax2.text(2, 4.5, 'Leg Status', fontsize=14, fontweight='bold', ha='center')
        
        # 다리 상태 표시
        state_names = ['Wait', 'Stance', 'Wait', 'Swing']
        y_pos = 3.5
        for i, (name, color) in enumerate(zip(leg_names, colors)):
            leg_state = phase_info[name]
            is_stance = stance_mask[i]
            
            # 배경색
            bg_color = 'lightgreen' if is_stance else 'lightyellow'
            rect = plt.Rectangle((0.5, y_pos - 0.3), 3, 0.6, 
                                 facecolor=bg_color, edgecolor='black', linewidth=2)
            ax2.add_patch(rect)
            
            # 텍스트
            ax2.text(1, y_pos, f'{name}:', fontsize=12, fontweight='bold', 
                    color=color, va='center')
            ax2.text(2.5, y_pos, f'{state_names[leg_state]}', fontsize=12, 
                    va='center', ha='center')
            
            y_pos -= 0.8
        
        # 안정성 정보
        ax2.text(2, 0.8, f'Stability: {stability_str}', fontsize=12, 
                fontweight='bold', ha='center',
                color='green' if is_stable else 'red')
        ax2.text(2, 0.3, f'Margin: {margin:.1f} mm', fontsize=10, ha='center')
        
        # 조작 안내
        ax2.text(2, -0.3, '[SPACE] Next  [Q] Quit  [A] Auto', fontsize=9, 
                ha='center', color='gray')
        
        fig.canvas.draw()
    
    def on_key(event):
        """키보드 이벤트 핸들러"""
        if event.key == ' ':  # 스페이스바: 다음 프레임
            state['frame'] = (state['frame'] + 1) % 60
            draw_frame(state['frame'])
            print(f"[Frame {state['frame']}/60] Press SPACE for next, Q to quit, A for auto-play")
        elif event.key == 'q':  # Q: 종료
            state['running'] = False
            plt.close(fig)
            print("[INFO] Visualization closed")
        elif event.key == 'a':  # A: 자동 재생 토글
            print("[INFO] Starting auto-play... Press Q to stop")
            for i in range(60):
                if not state['running']:
                    break
                state['frame'] = i
                draw_frame(state['frame'])
                plt.pause(0.1)
        elif event.key == 'left':  # 왼쪽 화살표: 이전 프레임
            state['frame'] = (state['frame'] - 1) % 60
            draw_frame(state['frame'])
            print(f"[Frame {state['frame']}/60]")
        elif event.key == 'right':  # 오른쪽 화살표: 다음 프레임
            state['frame'] = (state['frame'] + 1) % 60
            draw_frame(state['frame'])
            print(f"[Frame {state['frame']}/60]")
    
    # 키보드 이벤트 연결
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # 초기 프레임 그리기
    draw_frame(0)
    
    plt.tight_layout()
    plt.suptitle('Week 07: CoM (Center of Mass) Tracking Visualization\n'
                 '[SPACE] Next Frame | [←/→] Prev/Next | [A] Auto-play | [Q] Quit', 
                 fontsize=12, fontweight='bold', y=1.02)
    
    print("\n[INFO] Press SPACE to advance frames, Q to quit, A for auto-play")
    plt.show()


def main():
    """메인 함수"""
    print("=" * 60)
    print("CoM (Center of Mass) Tracking Visualization")
    print("-" * 60)
    print("이 시뮬레이션은 Trot Gait 동안 CoM과 Support Polygon의")
    print("변화를 시각화합니다.")
    print("")
    print("범례:")
    print("  ● (원)  : Stance (지면 접촉)")
    print("  × (X)   : Swing (공중)")
    print("  ★ (별)  : CoM (Center of Mass)")
    print("  녹색 영역: Support Polygon")
    print("=" * 60)
    
    visualize_com_tracking()


if __name__ == "__main__":
    main()
