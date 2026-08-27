import mujoco_automatic_gait

# 1. URDF 파일 읽기
# stairs_gen
try:
    model = mujoco.MjModel.from_xml_path("../urdf/spotmicroai_gen.urdf.xml")

    # 2. MJCF(MuJoCo용 XML)로 저장하기
    mujoco.mj_saveLastXML("../urdf/spot_micro_mujoco.xml", model)
    print("성공! spot_micro_mujoco.xml 파일이 생성되었습니다.")
except Exception as e:
    print(f"변환 실패: {e}")
