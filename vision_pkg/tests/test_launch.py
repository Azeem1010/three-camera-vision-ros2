"""launch/vision.launch.py 的邏輯測試：用 stub 取代 launch / launch_ros / ament_index_python。

驗證：鏡頭裝置路徑檢查與錯誤提示、usb_cam 與視覺節點的命名/參數、show_debug / undistort 傳遞、
camera_info_url 只在內參檔存在時帶入。
"""
import importlib.util
import sys
import types
from pathlib import Path

from vision_pkg import calib_io

PKG_ROOT = Path(__file__).resolve().parents[1]


class _Node:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Arg:
    def __init__(self, name, default_value='', description=''):
        self.name, self.default = name, default_value


class _LogInfo:
    def __init__(self, msg=''):
        self.msg = msg


class _Opaque:
    def __init__(self, function):
        self.function = function


class _LC:
    values = {}

    def __init__(self, name):
        self.name = name

    def perform(self, context):
        return _LC.values[self.name]


class _LD:
    def __init__(self, actions):
        self.actions = actions


def _load_launch_module(monkeyed):
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__path__ = []
        m.__dict__.update(attrs)
        monkeyed[name] = sys.modules.get(name)
        sys.modules[name] = m
        return m

    mod('ament_index_python')
    mod('ament_index_python.packages', get_package_share_directory=lambda name: str(PKG_ROOT))
    mod('launch', LaunchDescription=_LD)
    mod('launch.actions', DeclareLaunchArgument=_Arg, LogInfo=_LogInfo, OpaqueFunction=_Opaque)
    mod('launch.substitutions', LaunchConfiguration=_LC)
    mod('launch_ros')
    mod('launch_ros.actions', Node=_Node)
    spec = importlib.util.spec_from_file_location('vision_launch', PKG_ROOT / 'launch' / 'vision.launch.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _setup(tmp_path, **values):
    saved = {}
    try:
        m = _load_launch_module(saved)
        _LC.values = {'env_cam_device': '', 'arm1_cam_device': '', 'arm2_cam_device': '', 'show_debug': 'true',
                      'calib_dir': str(tmp_path), 'undistort': '', 'respawn': 'true'}
        _LC.values.update(values)
        return m, m._launch_setup(None)
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _nodes(actions):
    return [a for a in actions if isinstance(a, _Node)]


def test_no_devices_configured_skips_usb_cam_but_still_starts_vision_nodes(tmp_path):
    m, actions = _setup(tmp_path)
    nodes = _nodes(actions)
    assert [n.executable for n in nodes] == ['env_vision_node', 'arm1_vision_node', 'arm2_vision_node']
    msgs = [a.msg for a in actions if isinstance(a, _LogInfo)]
    err = '\n'.join(msgs)
    assert 'ls /dev/v4l/by-id/' in err and 'env_cam' in err and 'arm2_cam' in err
    assert all(n.respawn is True for n in nodes)


def test_existing_device_starts_usb_cam_in_its_namespace(tmp_path):
    dev = tmp_path / 'fake-video-index0'
    dev.write_text('', encoding='utf-8')
    m, actions = _setup(tmp_path, env_cam_device=str(dev))
    nodes = _nodes(actions)
    cams = [n for n in nodes if n.package == 'usb_cam']
    assert len(cams) == 1
    cam = cams[0]
    assert (cam.executable, cam.namespace, cam.name) == ('usb_cam_node_exe', 'env_cam', 'usb_cam')
    assert cam.parameters[0].endswith('usb_cam.yaml')
    assert cam.parameters[1]['video_device'] == str(dev) and cam.parameters[1]['camera_name'] == 'env_cam'
    assert 'camera_info_url' not in cam.parameters[1]            # 還沒標內參
    err = '\n'.join(a.msg for a in actions if isinstance(a, _LogInfo))
    assert 'arm1_cam' in err and 'arm2_cam' in err and 'env_cam:' not in err


def test_nonexistent_device_path_is_reported(tmp_path):
    m, actions = _setup(tmp_path, arm1_cam_device='/dev/v4l/by-id/does-not-exist')
    assert not [n for n in _nodes(actions) if n.package == 'usb_cam']
    err = '\n'.join(a.msg for a in actions if isinstance(a, _LogInfo))
    assert '裝置不存在' in err and 'does-not-exist' in err


def test_intrinsics_file_is_passed_as_camera_info_url_when_present(tmp_path):
    dev = tmp_path / 'cam0'
    dev.write_text('', encoding='utf-8')
    calib_io.intrinsics_path('arm2_cam', tmp_path).write_text('image_width: 640\n', encoding='utf-8')
    m, actions = _setup(tmp_path, arm2_cam_device=str(dev))
    cam = [n for n in _nodes(actions) if n.package == 'usb_cam'][0]
    assert cam.namespace == 'arm2_cam'
    assert cam.parameters[1]['camera_info_url'].startswith('file://')


def test_vision_node_parameters_show_debug_calib_dir_and_undistort(tmp_path):
    m, actions = _setup(tmp_path, show_debug='false', undistort='true', respawn='false')
    nodes = _nodes(actions)
    for n in nodes:
        assert n.parameters[0].endswith(('env_vision.yaml', 'arm1_vision.yaml', 'arm2_vision.yaml'))
        assert n.parameters[1] == {'show_debug': False, 'calib_dir': str(tmp_path), 'undistort': True}
        assert n.respawn is False
    m, actions = _setup(tmp_path)                                  # undistort 留空 -> 沿用各節點 YAML
    assert all('undistort' not in n.parameters[1] for n in _nodes(actions))
