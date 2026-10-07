import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'vision_pkg'

setup(
    name=package_name,
    version='0.1.0',
    # tests/ 沒有 __init__.py，find_packages 本來就不會收；exclude 只是明確表達
    packages=find_packages(exclude=['tests', 'tests.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # launch/ 與 config/ 安裝到 share/vision_pkg/ 底下
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vision',
    maintainer_email='azeemlala0786@gmail.com',
    description='雙機械手臂協同搬運系統的視覺套件（OpenCV，不使用深度學習）',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 三個視覺節點（各自獨立的行程）
            'env_vision_node = vision_pkg.env_vision_node:main',
            'arm1_vision_node = vision_pkg.arm1_vision_node:main',
            'arm2_vision_node = vision_pkg.arm2_vision_node:main',
            # 工具程式
            'hsv_tuner = vision_pkg.tools.hsv_tuner:main',
            'calibrate_intrinsics = vision_pkg.tools.calibrate_intrinsics:main',
            'calibrate_homography = vision_pkg.tools.calibrate_homography:main',
        ],
    },
)
