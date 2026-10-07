"""一次啟動三個 usb_cam 與三個視覺節點。

    ros2 launch vision_pkg vision.launch.py \
        env_cam_device:=/dev/v4l/by-id/usb-XXXX-video-index0 \
        arm1_cam_device:=/dev/v4l/by-id/usb-YYYY-video-index0 \
        arm2_cam_device:=/dev/v4l/by-id/usb-ZZZZ-video-index0 \
        show_debug:=true

鏡頭裝置路徑：launch 參數 > config/cameras.yaml。兩處皆留空或路徑不存在時，印出錯誤提示與查詢指令
（ls /dev/v4l/by-id/），略過該顆 usb_cam；其他鏡頭與視覺節點照常啟動（該鏡頭對應的 service 會回報「無影像」）。
視覺節點只訂閱 /env_cam|/arm1_cam|/arm2_cam/image_raw，絕不自行開啟 /dev/video。
"""
import os
import sys
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from vision_pkg import calib_io

# (命名空間/相機名稱, launch 參數名, 視覺節點 executable, 視覺節點參數檔)
CAMERAS = [
    ('env_cam', 'env_cam_device', 'env_vision_node', 'env_vision.yaml'),
    ('arm1_cam', 'arm1_cam_device', 'arm1_vision_node', 'arm1_vision.yaml'),
    ('arm2_cam', 'arm2_cam_device', 'arm2_vision_node', 'arm2_vision.yaml'),
]
LIST_CMD = 'ls /dev/v4l/by-id/'


def _truthy(s: str) -> bool:
    return str(s).strip().lower() in ('1', 'true', 'yes', 'on')


def _launch_setup(context, *args, **kwargs):
    share = Path(get_package_share_directory('vision_pkg'))
    cfg = share / 'config'
    arg = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731

    calib_dir = calib_io.resolve_calib_dir(arg('calib_dir'))
    show_debug = _truthy(arg('show_debug'))
    respawn = _truthy(arg('respawn'))
    undistort = arg('undistort')   # 空字串 = 沿用各節點 YAML 的設定

    try:
        cam_cfg = yaml.safe_load((cfg / 'cameras.yaml').read_text(encoding='utf-8')) or {}
    except Exception as e:
        cam_cfg = {}
        print(f'[vision.launch] 讀取 cameras.yaml 失敗: {e}', file=sys.stderr)

    actions = [LogInfo(msg=f'[vision.launch] 校正檔目錄: {calib_dir}')]
    problems = []
    for cam, dev_arg, vision_exe, vision_yaml in CAMERAS:
        device = arg(dev_arg).strip() or str((cam_cfg.get(cam) or {}).get('video_device', '')).strip()
        if not device:
            problems.append(f'{cam}: 未設定裝置路徑（launch 參數 {dev_arg}:= 或 config/cameras.yaml）')
        elif not os.path.exists(device):
            problems.append(f'{cam}: 裝置不存在 {device}（相機沒插好，或路徑打錯）')
        else:
            params = [str(cfg / 'usb_cam.yaml'),
                      {'video_device': device, 'camera_name': cam, 'frame_id': cam}]
            info = calib_io.intrinsics_path(cam, calib_dir)
            if info.exists():
                params[1]['camera_info_url'] = f'file://{info}'
            actions.append(Node(package='usb_cam', executable='usb_cam_node_exe', namespace=cam, name='usb_cam',
                                output='screen', parameters=params, respawn=respawn, respawn_delay=2.0))

        node_params = [str(cfg / vision_yaml),
                       {'show_debug': show_debug, 'calib_dir': str(calib_dir)}]
        if undistort:
            node_params[1]['undistort'] = _truthy(undistort)
        actions.append(Node(package='vision_pkg', executable=vision_exe, name=vision_exe, output='screen',
                            emulate_tty=True, parameters=node_params, respawn=respawn, respawn_delay=2.0))

    if problems:
        msg = ('[vision.launch] 以下鏡頭無法啟動（已略過 usb_cam，其他節點照常啟動）:\n  - ' + '\n  - '.join(problems) +
               f'\n  查詢可用裝置: {LIST_CMD}\n  範例: ros2 launch vision_pkg vision.launch.py '
               'env_cam_device:=/dev/v4l/by-id/<名稱> arm1_cam_device:=... arm2_cam_device:=...')
        print(msg, file=sys.stderr)
        actions.insert(1, LogInfo(msg=msg))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('env_cam_device', default_value='', description='環境相機 /dev/v4l/by-id/ 路徑（待實測）'),
        DeclareLaunchArgument('arm1_cam_device', default_value='', description='手眼相機 1 /dev/v4l/by-id/ 路徑（待實測）'),
        DeclareLaunchArgument('arm2_cam_device', default_value='', description='手眼相機 2 /dev/v4l/by-id/ 路徑（待實測）'),
        DeclareLaunchArgument('show_debug', default_value='true',
                              description='三個視覺節點是否開 debug 視窗（沒有 DISPLAY 時自動停用）'),
        DeclareLaunchArgument('calib_dir', default_value='',
                              description='校正檔目錄（空 = $VISION_CALIB_DIR 或 ~/.ros/vision_calib）'),
        DeclareLaunchArgument('undistort', default_value='',
                              description='true/false 覆寫節點的去畸變設定；空字串 = 沿用 config'),
        DeclareLaunchArgument('respawn', default_value='true', description='節點/相機異常結束時是否自動重啟'),
        OpaqueFunction(function=_launch_setup),
    ])
