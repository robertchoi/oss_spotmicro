import mujoco_automatic_gait

# 계단 URDF 변환
try:
    stairs_model = mujoco.MjModel.from_xml_path("../urdf/stairs_gen.urdf.xml")
    mujoco.mj_saveLastXML("../urdf/stairs_mujoco.xml", stairs_model)
    print("성공! stairs_mujoco.xml 파일이 생성되었습니다.")
except Exception as e:
    print(f"계단 변환 실패: {e}")