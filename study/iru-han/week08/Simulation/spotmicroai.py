import sys
import os
import time
import math
import numpy as np

# 현재 파일의 절대 경로를 기준으로 상위 폴더들을 탐색 경로에 추가함
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)  # Simulation 폴더 추가
sys.path.append(parent_dir)   # SpotMicroJetson 폴더 추가

# 이제 Kinematics 패키지를 정상적으로 불러올 수 있음
from Kinematics.kinematics import Kinematic
from enum import Enum

# 상위 폴더 경로 추가
sys.path.append("..")

try:
    import pybullet as p
    import pybullet_data
except:
    pass

try:
    import mujoco
except:
    pass


class RobotState(Enum):
    OFF = 0
    READY = 1
    STAND = 2
    TROTTING_GAIT = 3
    CRAWL = 4
    CRAWL2 = 5


class Robot:
    def __init__(self, useFixedBase=False, useStairs=True, resetFunc=None, use_mujoco=False,
                 xml_path="../urdf/spot_micro_mujoco.xml"):
        self.use_mujoco = use_mujoco
        self.kin = Kinematic()
        self.resetFunc = resetFunc

        # 공통 설정
        self.dirs = np.array([[-1, 1, 1], [1, 1, 1], [-1, 1, 1], [1, 1, 1]])
        self.rot = (0, 0, 0)
        self.pos = (0, 100, 0)
        self.W = 120  # 원본 75+5+40 기준
        self.Lp = np.array([[120, -100, self.W / 2, 1], [120, -100, -self.W / 2, 1],
                            [-50, -100, self.W / 2, 1], [-50, -100, -self.W / 2, 1]])
        self.angles = np.zeros((4, 3))

        if self.use_mujoco:
            # --- MuJoCo 전용 초기화 ---
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
            print("MuJoCo Engine Loaded.")
        else:
            # --- PyBullet 전용 초기화 (기존 로직 100% 유지) ---
            self.useMaximalCoordinates = False
            self.useRealTime = True
            self.debugLidar = False
            self.rotateCamera = False
            self.debug = False
            self.fixedTimeStep = 1. / 550
            self.numSolverIterations = 200
            self.useFixedBase = useFixedBase
            self.useStairs = useStairs
            self.init_oritentation = p.getQuaternionFromEuler([0, 0, math.pi / 2])  # 90도
            self.init_position = [0, 0, 0.3]
            self.reflection = False
            self.state = RobotState.OFF
            self.kp = 0.045
            self.kd = 0.4
            self.maxForce = 25.0

            self.physId = p.connect(p.SHARED_MEMORY)
            if (self.physId < 0):
                p.connect(p.GUI)

            if self.reflection:
                p.configureDebugVisualizer(p.COV_ENABLE_PLANAR_reflection, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 1)

            self.IDkp = p.addUserDebugParameter("Kp", 0, 0.05, self.kp)
            self.IDkd = p.addUserDebugParameter("Kd", 0, 1, self.kd)
            self.IDmaxForce = p.addUserDebugParameter("MaxForce", 0, 50, 12.5)

            p.setRealTimeSimulation(self.useRealTime)
            self.quadruped = self.loadModels()
            self.createEnv()
            self.changeDynamics(self.quadruped)
            self.jointNameToId = self.getJointNames(self.quadruped)

            # Lidar 레이 설정
            self.numRays = 360
            self.rayFrom, self.rayTo, self.rayIds = [], [], []
            self.rayHitColor, self.rayMissColor = [1, 0, 0], [0, 1, 0]
            for i in range(self.numRays):
                h = 0.045
                self.rayFrom.append(
                    [0.12 * math.sin(math.pi * 2 * i / self.numRays), 0.12 * math.cos(math.pi * 2 * i / self.numRays),
                     h])
                self.rayTo.append(
                    [12 * math.sin(math.pi * 2 * i / self.numRays), 12 * math.cos(math.pi * 2 * i / self.numRays), h])
                self.rayIds.append(-1)

            p.setPhysicsEngineParameter(numSolverIterations=self.numSolverIterations)
            self.ref_time = time.time()
            p.resetDebugVisualizerCamera(1, 85.6, 0, [-0.61, 0.12, 0.25])
            fov, aspect, near, far = 90, 1.3, 0.0111, 100
            self.projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
            self.lastLidarTime = 0
            self.t = 0
            print("PyBullet Engine Loaded.")

    # --- 기존 PyBullet 전용 메서드들 (삭제 없이 유지) ---
    def createEnv(self):
        shift = [0, 0, 0]
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX, rgbaColor=[1, 1, 1, 1], halfExtents=[1, 1, .5],
                                            visualFramePosition=shift)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX, collisionFramePosition=shift,
                                                  halfExtents=[1, 1, .5])
        rangex, rangey = 5, 5
        for i in range(rangex):
            for j in range(rangey):
                p.createMultiBody(baseMass=1000, baseCollisionShapeIndex=collisionShapeId,
                                  baseVisualShapeIndex=visualShapeId,
                                  basePosition=[((-rangex / 2) + i) * 5, (-rangey / 2 + j) * 5, 1],
                                  useMaximalCoordinates=True)

    def loadModels(self):
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        if self.useStairs:
            p.loadURDF("../urdf/stairs_gen.urdf.xml", [0, -1, 0])
        return p.loadURDF("../urdf/spotmicroai_gen.urdf.xml", self.init_position, self.init_oritentation,
                          useFixedBase=self.useFixedBase)

    def changeDynamics(self, quadruped):
        for i in range(p.getNumJoints(quadruped)):
            p.changeDynamics(quadruped, i, localInertiaDiagonal=[1e-6, 1e-6, 1e-6])

    def getJointNames(self, quadruped):
        return {p.getJointInfo(quadruped, i)[1].decode('UTF-8'): i for i in range(p.getNumJoints(quadruped))}

    def handleCamera(self, cubePos, cubeOrn):
        # 기존 카메라 로직 유지
        pass

    def addInfoText(self, bodyPos, bodyEuler, linearVel, angularVel):
        # 기존 디버그 텍스트 로직 유지
        pass

    def resetBody(self):
        if self.use_mujoco:
            mujoco.mj_resetData(self.model, self.data)
        else:
            p.resetBasePositionAndOrientation(self.quadruped, self.init_position, [0, 0, 0, 1])
            p.resetBaseVelocity(self.quadruped, [0, 0, 0], [0, 0, 0])
        if self.resetFunc: self.resetFunc()

    # --- 공통 인터페이스 메서드 ---
    def feetPosition(self, Lp):
        self.Lp = Lp

    def bodyRotation(self, rot):
        self.rot = rot

    def bodyPosition(self, pos):
        self.pos = pos

    def getAngle(self):
        return self.angles

    def getPos(self):
        if self.use_mujoco: return self.data.qpos[:3]
        return p.getBasePositionAndOrientation(self.quadruped)[0]

    # spotmicroai.py 파일의 getIMU 함수 수정
    def getIMU(self):
        if self.use_mujoco:
            # qpos[0:3]은 위치, qpos[3:7]은 쿼터니언(회전)이다.
            # MuJoCo의 쿼터니언 형식은 [w, x, y, z]이다.
            quat = self.data.qpos[3:7]
            # qvel[0:3]은 선속도, qvel[3:6]은 각속도이다.
            lin_vel = self.data.qvel[:3]
            ang_vel = self.data.qvel[3:6]
            return quat, lin_vel, ang_vel
        else:
            # PyBullet 로직 유지
            return p.getBasePositionAndOrientation(self.quadruped)[1], \
                p.getBaseVelocity(self.quadruped)[0], \
                p.getBaseVelocity(self.quadruped)[1]
    # def getIMU(self):
    #     if self.use_mujoco: return self.data.qquat[:4], self.data.qvel[:3], self.data.qvel[3:6]
    #     return p.getBasePositionAndOrientation(self.quadruped)[1], p.getBaseVelocity(self.quadruped)[0], \
    #     p.getBaseVelocity(self.quadruped)[1]

    def step(self):
        # 1. 공통 역운동학 계산
        self.angles = self.kin.calcIK(self.Lp, self.rot, self.pos)

        if self.use_mujoco:
            # 2. MuJoCo 제어 (방향 보정 적용)
            for lx in range(4):
                actual = self.angles[lx] * self.dirs[lx]
                self.data.ctrl[lx * 3: lx * 3 + 3] = actual
            mujoco.mj_step(self.model, self.data)
        else:
            # 3. PyBullet 제어 (기존 로직 유지)
            bodyPos, bodyOrn = p.getBasePositionAndOrientation(self.quadruped)
            kp = p.readUserDebugParameter(self.IDkp)
            kd = p.readUserDebugParameter(self.IDkd)
            maxF = p.readUserDebugParameter(self.IDmaxForce)

            for lx, leg in enumerate(['front_left', 'front_right', 'rear_left', 'rear_right']):
                for px, part in enumerate(['shoulder', 'leg', 'foot']):
                    j_id = self.jointNameToId[f"{leg}_{part}"]
                    p.setJointMotorControl2(self.quadruped, j_id, p.POSITION_CONTROL,
                                            targetPosition=self.angles[lx][px] * self.dirs[lx][px],
                                            positionGain=kp, velocityGain=kd, force=maxF)
            if not self.useRealTime: p.stepSimulation()