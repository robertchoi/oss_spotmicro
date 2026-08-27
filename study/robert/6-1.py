"""
6-1.py: Swing/Stance Phase 발끝 궤적 시각화

Week 06 - 2.2절과 2.3절 코드를 통합한 완전한 실행 파일
Trot Gait의 기본이 되는 발끝 궤적(Foot Trajectory)을 시각화합니다.

실행 방법:
    python 6-1.py
"""

import numpy as np
import math
import matplotlib.pyplot as plt


class FootTrajectory:
    """
    한 발의 발끝 궤적을 생성하는 클래스
    
    Swing Phase (0% ~ 50%): 발을 들어올려 앞으로 이동
    Stance Phase (50% ~ 100%): 지면에 닿아 뒤로 밀기
    """
    
    def __init__(self, step_length=60, step_height=40, ground_height=-150):
        """
        Parameters:
        -----------
        step_length : float
            한 걸음의 보폭 (mm)
        step_height : float
            발을 들어올리는 높이 (mm)
        ground_height : float
            지면 높이 (몸체 기준, mm)
        """
        self.step_length = step_length
        self.step_height = step_height
        self.ground_height = ground_height

    def get_position(self, phase):
        """
        phase에 따른 발끝 위치 계산

        Parameters:
        -----------
        phase : float
            0.0 ~ 1.0 사이의 값
            0.0 ~ 0.5: Swing phase
            0.5 ~ 1.0: Stance phase

        Returns:
        --------
        x, z : float
            발끝의 X좌표(전후)와 Z좌표(상하)
        """
        Sl = self.step_length
        Sh = self.step_height
        ground_z = self.ground_height

        if phase < 0.5:
            # ===== Swing Phase =====
            # 반원 궤적으로 발을 앞으로 이동
            swing_phase = phase * 2  # 0 ~ 1로 정규화

            # X: 뒤에서 앞으로 이동
            x = -Sl/2 + Sl * swing_phase

            # Z: 반원 궤적 (sin 곡선)
            z = ground_z + Sh * math.sin(math.pi * swing_phase)
        else:
            # ===== Stance Phase =====
            # 지면에 닿아 직선으로 복귀
            stance_phase = (phase - 0.5) * 2  # 0 ~ 1로 정규화

            # X: 앞에서 뒤로 이동
            x = Sl/2 - Sl * stance_phase

            # Z: 지면 높이 유지
            z = ground_z

        return x, z

    def get_full_trajectory(self, num_points=100):
        """
        전체 궤적을 배열로 반환
        
        Returns:
        --------
        x_vals, z_vals : np.array
            X, Z 좌표 배열
        """
        phases = np.linspace(0, 1, num_points)
        positions = [self.get_position(p) for p in phases]
        x_vals = np.array([p[0] for p in positions])
        z_vals = np.array([p[1] for p in positions])
        return x_vals, z_vals


def visualize_foot_trajectory():
    """
    발끝 궤적을 시각화
    """
    # 파라미터 설정
    step_length = 60   # 보폭 (mm)
    step_height = 40   # 발 들어올리는 높이 (mm)
    ground_height = -150  # 지면 높이 (mm)
    
    # FootTrajectory 인스턴스 생성
    traj = FootTrajectory(
        step_length=step_length, 
        step_height=step_height, 
        ground_height=ground_height
    )

    # 궤적 계산
    phases = np.linspace(0, 1, 100)
    positions = [traj.get_position(p) for p in phases]
    x_vals = [p[0] for p in positions]
    z_vals = [p[1] for p in positions]

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 6))

    # Swing phase (파란색) - 0% ~ 50%
    swing_idx = 50
    ax.plot(x_vals[:swing_idx], z_vals[:swing_idx],
            'b-', linewidth=3, label='Swing Phase (발 들림)')

    # Stance phase (빨간색) - 50% ~ 100%
    ax.plot(x_vals[swing_idx:], z_vals[swing_idx:],
            'r-', linewidth=3, label='Stance Phase (지면 접촉)')

    # 시작/끝 점 표시
    ax.scatter(x_vals[0], z_vals[0], color='green', s=100, zorder=5, label='Start')
    ax.scatter(x_vals[-1], z_vals[-1], color='orange', s=100, zorder=5, label='End')

    # 화살표로 이동 방향 표시
    # Swing phase 방향 (앞으로)
    ax.annotate('', xy=(x_vals[25], z_vals[25]),
                xytext=(x_vals[20], z_vals[20]),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    # Stance phase 방향 (뒤로)
    ax.annotate('', xy=(x_vals[75], z_vals[75]),
                xytext=(x_vals[70], z_vals[70]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # 그래프 꾸미기
    ax.set_xlabel('X Position (mm)', fontsize=12)
    ax.set_ylabel('Z Position (mm)', fontsize=12)
    ax.set_title('Foot Trajectory - Swing/Stance Phase', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('equal')

    # 추가 정보 표시
    info_text = f'Step Length: {step_length}mm\nStep Height: {step_height}mm\nGround: {ground_height}mm'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()


def visualize_trot_gait_phases():
    """
    Trot Gait에서 대각선 다리 쌍의 phase 차이를 시각화
    """
    traj = FootTrajectory(step_length=60, step_height=40, ground_height=-150)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Trot Gait - 4 Legs Phase Timing', fontsize=16)
    
    leg_names = ['Front Left (FL)', 'Front Right (FR)', 'Rear Left (RL)', 'Rear Right (RR)']
    phase_offsets = [0.0, 0.5, 0.5, 0.0]  # Trot: FL+RR 동기화, FR+RL 동기화
    colors = ['blue', 'red', 'green', 'orange']
    
    for idx, (ax, name, offset, color) in enumerate(zip(axes.flatten(), leg_names, phase_offsets, colors)):
        # 해당 다리의 궤적 계산
        phases = np.linspace(0, 1, 100)
        adjusted_phases = [(p + offset) % 1.0 for p in phases]
        positions = [traj.get_position(p) for p in adjusted_phases]
        x_vals = [p[0] for p in positions]
        z_vals = [p[1] for p in positions]
        
        # 궤적 그리기
        ax.plot(x_vals, z_vals, color=color, linewidth=2)
        ax.scatter(x_vals[0], z_vals[0], color='green', s=80, zorder=5, label='t=0')
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Z (mm)')
        ax.set_title(f'{name}\nPhase Offset: {offset*100:.0f}%')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_aspect('equal')
        ax.legend()
    
    plt.tight_layout()
    plt.show()


def visualize_gait_timing():
    """
    시간에 따른 각 다리의 Swing/Stance 상태를 타임라인으로 시각화
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    leg_names = ['Front Left', 'Front Right', 'Rear Left', 'Rear Right']
    phase_offsets = [0.0, 0.5, 0.5, 0.0]  # Trot gait
    
    # 2 주기 표시
    total_time = 2.0
    
    for idx, (name, offset) in enumerate(zip(leg_names, phase_offsets)):
        y_pos = len(leg_names) - idx - 1
        
        # 시간에 따른 phase 계산
        for cycle in range(2):  # 2 cycle 표시
            cycle_start = cycle * 1.0
            
            # Swing phase (0 ~ 0.5)
            swing_start = cycle_start + offset * 1.0
            swing_end = swing_start + 0.5
            
            # Stance phase (0.5 ~ 1.0)
            stance_start = swing_end
            stance_end = swing_start + 1.0
            
            # 주기 범위 조정
            if swing_start >= total_time:
                continue
            swing_end = min(swing_end, total_time)
            stance_start = min(stance_start, total_time)
            stance_end = min(stance_end, total_time)
            
            # Swing phase 그리기 (파란색 - 발 들림)
            if swing_start < total_time and swing_start < swing_end:
                ax.barh(y_pos, swing_end - swing_start, left=swing_start, height=0.6, 
                       color='skyblue', edgecolor='blue', linewidth=2, label='Swing' if idx == 0 and cycle == 0 else '')
            
            # Stance phase 그리기 (빨간색 - 지면 접촉)
            if stance_start < total_time and stance_start < stance_end:
                ax.barh(y_pos, stance_end - stance_start, left=stance_start, height=0.6,
                       color='lightcoral', edgecolor='red', linewidth=2, label='Stance' if idx == 0 and cycle == 0 else '')
    
    ax.set_yticks(range(len(leg_names)))
    ax.set_yticklabels(reversed(leg_names), fontsize=12)
    ax.set_xlabel('Time (normalized)', fontsize=12)
    ax.set_title('Trot Gait Timing Diagram\n(대각선 다리 = 동시 Swing/Stance)', fontsize=14)
    ax.set_xlim(0, total_time)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # 범례
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', edgecolor='blue', label='Swing (발 들림)'),
        Patch(facecolor='lightcoral', edgecolor='red', label='Stance (지면 접촉)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # 대각선 쌍 표시
    ax.axhline(y=1.5, color='gray', linestyle=':', linewidth=1)
    ax.text(total_time + 0.05, 2.75, 'Pair 1\n(FL+RR)', fontsize=10, va='center')
    ax.text(total_time + 0.05, 1.25, 'Pair 2\n(FR+RL)', fontsize=10, va='center')
    
    plt.tight_layout()
    plt.show()


def main():
    """
    메인 함수 - 세 가지 시각화 실행
    """
    print("=" * 60)
    print("Week 06: Swing/Stance Phase 발끝 궤적 시각화")
    print("=" * 60)
    print()
    print("1. 단일 발 궤적 시각화")
    print("2. 4개 다리 궤적 비교")
    print("3. Trot Gait 타이밍 다이어그램")
    print()
    
    # 1. 기본 발끝 궤적 시각화
    print("[1/3] 단일 발 궤적 시각화...")
    visualize_foot_trajectory()
    
    # 2. 4개 다리 비교
    print("[2/3] 4개 다리 궤적 비교...")
    visualize_trot_gait_phases()
    
    # 3. 타이밍 다이어그램
    print("[3/3] Trot Gait 타이밍 다이어그램...")
    visualize_gait_timing()
    
    print()
    print("시각화 완료!")


if __name__ == "__main__":
    main()
