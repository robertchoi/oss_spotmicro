import mujoco
import mujoco.viewer
import numpy as np
import time
from collections import OrderedDict


# ==========================================
# 1. 수학 및 변환 유틸리티 (LieAlgebra 핵심)
# ==========================================
def RPY(roll, pitch, yaw):
    Roll = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Roll @ Pitch @ Yaw


def RpToTrans(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def TransToRp(T):
    return T[:3, :3], T[:3, 3]


def TransInv(T):
    R, p = TransToRp(T)
    Rt = R.T
    return RpToTrans(Rt, -Rt @ p)


# ==========================================
# 2. Inverse Kinematics (LegIK)
# ==========================================
# 단순히 0.3205만 쓰면 계산이 틀어집니다. 피타고라스 정리를 이용해 대각선 길이(sqrt(0.025^2 + 0.3205^2))인 약 0.3215를 써야 정확합니다.
class LegIK:
    def __init__(self, legtype="RIGHT", shoulder_length=0.110945, elbow_length=0.3215, wrist_length=0.37):
        self.legtype = legtype
        self.shoulder_length = shoulder_length
        self.elbow_length = elbow_length
        self.wrist_length = wrist_length

    def solve(self, xyz_coord):
        x, y, z = xyz_coord
        # Domain Calculation
        D = (y ** 2 + (-z) ** 2 - self.shoulder_length ** 2 + (
            -x) ** 2 - self.elbow_length ** 2 - self.wrist_length ** 2) / (2 * self.wrist_length * self.elbow_length)
        D = np.clip(D, -1.0, 1.0)

        wrist_angle = np.arctan2(-np.sqrt(1 - D ** 2), D)
        sqrt_component = np.maximum(y ** 2 + (-z) ** 2 - self.shoulder_length ** 2, 0.0)

        offset = self.shoulder_length if self.legtype == "LEFT" else -self.shoulder_length
        shoulder_angle = -np.arctan2(z, y) - np.arctan2(np.sqrt(sqrt_component), offset)
        elbow_angle = np.arctan2(-x, np.sqrt(sqrt_component)) - np.arctan2(
            self.wrist_length * np.sin(wrist_angle), self.elbow_length + self.wrist_length * np.cos(wrist_angle))

        return np.array([-shoulder_angle, elbow_angle, wrist_angle])


# ==========================================
# 3. Spot 모델 관리자
# ==========================================
class SpotModel:
    def __init__(self):
        # XML 규격에 맞춘 치수 (제공된 XML 기반)
        self.hip_x, self.hip_y = 0.29785 * 2, 0.055 * 2
        self.foot_x, self.foot_y = 0.29785 * 2, 0.2 * 2
        self.height = 0.45

        self.Legs = OrderedDict([
            ("FL", LegIK("LEFT")), ("FR", LegIK("RIGHT")),
            ("RL", LegIK("LEFT")), ("RR", LegIK("RIGHT"))
        ])

        # 기본 힙 위치 설정
        self.WorldToHip = {
            "FL": RpToTrans(np.eye(3), [self.hip_x / 2, self.hip_y / 2, 0]),
            "FR": RpToTrans(np.eye(3), [self.hip_x / 2, -self.hip_y / 2, 0]),
            "RL": RpToTrans(np.eye(3), [-self.hip_x / 2, self.hip_y / 2, 0]),
            "RR": RpToTrans(np.eye(3), [-self.hip_x / 2, -self.hip_y / 2, 0])
        }

    def IK(self, orn, pos, T_bf):
        Rb = RPY(*orn)
        T_wb = RpToTrans(Rb, pos)
        joint_angles = {}

        for key, T_wh in self.WorldToHip.items():
            T_bh = TransInv(T_wb) @ T_wh
            T_hf = TransInv(T_bh) @ T_bf[key]
            _, p_hf = TransToRp(T_hf)
            joint_angles[key] = self.Legs[key].solve(p_hf)
        return joint_angles


# ==========================================
# 4. Gait Generator (Bezier)
# ==========================================
# (제공된 BezierGait 클래스에서 핵심 로직만 간소화)
class SimpleGait:
    def __init__(self):
        self.phases = [0.0, 0.5, 0.5, 0.0]  # FL, FR, RL, RR (Trot)
        self.time = 0.0

    def get_foot_step(self, t, L, height):
        # 매우 단순화된 베지에 스윙 궤적
        phase = t % 1.0
        if phase < 0.5:  # Stance
            x = L * (1 - 4 * phase)
            z = 0
        else:  # Swing
            sw_ph = (phase - 0.5) * 2
            x = L * (4 * sw_ph - 3)
            z = height * np.sin(np.pi * sw_ph)
        return x, 0, z


# ==========================================
# 5. 메인 시뮬레이션 루프
# ==========================================
def main():
    # 모델 로드
    try:
        model = mujoco.MjModel.from_xml_path('scene.xml')
    except ValueError as e:
        print(f"모델 로드 실패: {e}")
        return
    data = mujoco.MjData(model)
    spot = SpotModel()
    gait = SimpleGait()

    # 관절 매핑 (XML의 액추에이터 순서와 일치해야 함)
    # XML: FL(3), FR(3), RL(3), RR(3) 순서
    leg_keys = ["FL", "FR", "RL", "RR"]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()

        while viewer.is_running():
            step_start = time.time()
            elapsed = time.time() - start_time  # 경과 시간 계산

            if elapsed < 5.0:
                # 5초 동안은 걷지 말고 제자리에 서 있어라!
                L = 0.0
                step_h = 0.0
                t = 0
            else:
                # 1. 보행 파라미터 업데이트
                t = (time.time() - start_time) * 1.5  # 속도 조절
                L = 0.08  # 보폭
                step_h = 0.06  # 발 들기 높이

            # 2. 각 다리의 목표 발 위치(T_bf) 계산
            T_bf = {}
            for i, key in enumerate(leg_keys):
                dx, dy, dz = gait.get_foot_step(t + gait.phases[i], L, step_h)
                # 기본 발 위치 (바닥)
                default_p = [spot.foot_x / 2 if "F" in key else -spot.foot_x / 2,
                             spot.foot_y / 2 if "L" in key else -spot.foot_y / 2,
                             -spot.height]
                T_bf[key] = RpToTrans(np.eye(3), np.array(default_p) + [dx, dy, dz])

            # 3. IK 풀기 (몸체는 수평 유지)
            angles = spot.IK(orn=[0, 0, 0], pos=[0, 0, 0], T_bf=T_bf)

            # 4. MuJoCo 액추에이터에 명령 전달
            # XML 순서: FL(hip, upper, lower), FR, RL, RR
            ctrl_idx = 0
            for key in leg_keys:
                data.ctrl[ctrl_idx:ctrl_idx + 3] = angles[key]
                ctrl_idx += 3

            # 시뮬레이션 진행
            mujoco.mj_step(model, data)
            viewer.sync()

            # 실시간 동기화
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()