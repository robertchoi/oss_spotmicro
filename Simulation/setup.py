from setuptools import setup, find_packages

setup(
    name='gym_spotmicroai',
    version='0.1',
    packages=find_packages(include=['gym_spotmicroai', 'gym_spotmicroai.*']),
    install_requires=['numpy', 'pybullet>=3.0.6', 'matplotlib>=3.5.0', 'scipy>=1.7.0']
    # gym 제거: 빌드 문제로 인해 선택적 의존성으로 변경
)
