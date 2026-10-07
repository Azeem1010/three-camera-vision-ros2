"""校正 / 設定檔 I/O（不含 ROS），供三個節點與工具程式共用。

檔案存放位置（calib_dir）：參數 calib_dir > 環境變數 VISION_CALIB_DIR > ~/.ros/vision_calib
  zones.yaml                 環境相機 A/B/C 角點（節點自動寫入）
  hsv.yaml                   hsv_tuner 調整後的 HSV（逐色覆寫套件預設 config/hsv.yaml）
  arm1_homography.yaml       像素 -> 手臂 1 基座座標（m）
  a_station_homography.yaml  像素 -> A 站平面座標（mm）
  arm2_homography.yaml       像素 -> 手臂 2 基座座標（m）
  <cam>_intrinsics.yaml      相機內參（camera_info YAML 格式，usb_cam 可直接讀）
讀取時 calib_dir 優先，找不到才退回套件內建的範本（config/，通常是唯讀的 install 目錄）。
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import yaml

from .vision_core.color_zone import HsvRange
from .vision_core.geometry import build_undistort_maps, undistort

CALIB_ENV_VAR = 'VISION_CALIB_DIR'
CAMERA_NAMES = {'env': 'env_cam', 'arm1': 'arm1_cam', 'arm2': 'arm2_cam'}


# --------------------------------------------------------------------------- #
# 路徑
# --------------------------------------------------------------------------- #
def package_config_dir() -> Path:
    """套件內建 config 目錄：已安裝時為 share/vision_pkg/config，直接從原始碼執行時為專案 config/。"""
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory('vision_pkg')) / 'config'
    except Exception:
        return Path(__file__).resolve().parent.parent / 'config'


def resolve_calib_dir(param_value: str = '') -> Path:
    p = param_value or os.environ.get(CALIB_ENV_VAR, '') or '~/.ros/vision_calib'
    return Path(os.path.expanduser(p))


def intrinsics_path(camera_name: str, calib_dir: Path) -> Path:
    return Path(calib_dir) / f'{camera_name}_intrinsics.yaml'


# --------------------------------------------------------------------------- #
# YAML 基本讀寫
# --------------------------------------------------------------------------- #
def to_builtin(obj: Any) -> Any:
    """numpy 型別轉成 YAML 可安全序列化的 Python 內建型別。"""
    if isinstance(obj, np.ndarray):
        return to_builtin(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    return obj


def load_yaml(path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def save_yaml(path, data: Dict[str, Any], header: str = '') -> None:
    """原子寫入（先寫暫存檔再取代），避免節點剛好在讀時讀到半個檔案。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = header + yaml.safe_dump(to_builtin(data), allow_unicode=True, sort_keys=False,
                                   default_flow_style=None)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# --------------------------------------------------------------------------- #
# 監看檔案變更並自動重新載入
# --------------------------------------------------------------------------- #
class ReloadableConfig:
    """監看一組檔案的 mtime，變動時以 loader 重新載入。載入失敗時保留舊值並把原因放在 .error。"""

    def __init__(self, loader: Callable[[], Any], paths: Sequence[Path], period_s: float):
        self._loader = loader
        self._paths = [Path(p) for p in paths]
        self._period = float(period_s)
        self._last_check = 0.0
        self._sig = None
        self.value: Any = None
        self.error: str = ''
        self.poll(force=True)

    def _signature(self):
        sig = []
        for p in self._paths:
            try:
                st = p.stat()
                sig.append((st.st_mtime_ns, st.st_size))
            except OSError:
                sig.append(None)
        return tuple(sig)

    def poll(self, force: bool = False) -> bool:
        """回傳 True 代表剛成功重新載入。"""
        now = time.monotonic()
        if not force and now - self._last_check < self._period:
            return False
        self._last_check = now
        sig = self._signature()
        if not force and sig == self._sig:
            return False
        self._sig = sig
        try:
            self.value = self._loader()
            self.error = ''
            return True
        except Exception as e:  # 檔案寫到一半 / 格式錯誤：保留舊值
            self.error = f'{type(e).__name__}: {e}'
            return False


# --------------------------------------------------------------------------- #
# HSV
# --------------------------------------------------------------------------- #
def hsv_paths(calib_dir: Path, hsv_file: str = '') -> Tuple[Path, Path]:
    """(套件預設 hsv.yaml, 覆寫檔)。hsv_file 非空時用它當覆寫檔。"""
    over = Path(hsv_file) if hsv_file else Path(calib_dir) / 'hsv.yaml'
    return package_config_dir() / 'hsv.yaml', over


def load_hsv_ranges(calib_dir: Path, hsv_file: str = '') -> Dict[str, HsvRange]:
    base_p, over_p = hsv_paths(calib_dir, hsv_file)
    merged = dict(load_yaml(base_p))
    if over_p.exists():
        merged.update(load_yaml(over_p))
    return {name: HsvRange.from_dict(v) for name, v in merged.items()
            if isinstance(v, dict) and 'lower' in v and 'upper' in v}


def save_hsv_range(calib_dir: Path, name: str, rng: HsvRange, hsv_file: str = '') -> Path:
    """把單一顏色寫進覆寫檔（保留其他顏色）。"""
    _, over_p = hsv_paths(calib_dir, hsv_file)
    data = load_yaml(over_p) if over_p.exists() else {}
    data[name] = rng.to_dict()
    save_yaml(over_p, data, header='# 由 hsv_tuner 產生；逐色覆寫套件預設 hsv.yaml（OpenCV HSV：H 0-179, S/V 0-255）\n')
    return over_p


# --------------------------------------------------------------------------- #
# Homography
# --------------------------------------------------------------------------- #
def load_homography(name: str, calib_dir: Path) -> Dict[str, Any]:
    """name 例如 'arm1_homography'。回傳 {'valid','H','unit','path','rms_error','image_size'}。"""
    for p in (Path(calib_dir) / f'{name}.yaml', package_config_dir() / f'{name}.yaml'):
        if p.exists():
            d = load_yaml(p)
            H = np.array(d['H'], dtype=np.float64).reshape(3, 3)
            return {'valid': bool(d.get('valid', False)), 'H': H, 'unit': d.get('unit', ''),
                    'path': str(p), 'rms_error': d.get('rms_error'), 'image_size': d.get('image_size')}
    raise FileNotFoundError(f'找不到 {name}.yaml（{calib_dir} 與套件 config 皆無）')


def save_homography(path, H: np.ndarray, unit: str, image_size: Sequence[int], pixel_pts, world_pts,
                    rms_error: float, undistorted: bool, header: str = '') -> None:
    pts = [{'pixel': [float(p[0]), float(p[1])], 'world': [float(w[0]), float(w[1])]}
           for p, w in zip(pixel_pts, world_pts)]
    save_yaml(path, {
        'valid': True,
        'unit': unit,
        'image_size': [int(image_size[0]), int(image_size[1])],
        'undistorted': bool(undistorted),
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'rms_error': float(rms_error),
        'H': np.asarray(H, dtype=np.float64),
        'points': pts,
    }, header=header)


# --------------------------------------------------------------------------- #
# 站台角點
# --------------------------------------------------------------------------- #
def load_zones(calib_dir: Path) -> Optional[Dict[str, Dict[str, Any]]]:
    """讀上次存檔的站台。回傳 {'a': {'corners': (4,2) ndarray, 'baseline': float}, ...}；無有效存檔回傳 None。"""
    p = Path(calib_dir) / 'zones.yaml'
    if not p.exists():
        return None
    d = load_yaml(p)
    if not d.get('valid', True) or not d.get('zones'):
        return None
    out: Dict[str, Dict[str, Any]] = {}
    for name, z in d['zones'].items():
        c = np.array(z['corners_px'], dtype=np.float32)
        if c.shape != (4, 2):
            raise ValueError(f'zones.yaml 的 {name} 角點形狀不是 (4,2)')
        out[name] = {'corners': c, 'baseline': float(z['baseline_visible_ratio'])}
    return out


def save_zones(calib_dir: Path, image_size: Sequence[int], zones: Dict[str, Dict[str, Any]]) -> Path:
    p = Path(calib_dir) / 'zones.yaml'
    save_yaml(p, {
        'version': 1,
        'valid': True,
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'image_size': [int(image_size[0]), int(image_size[1])],
        'zones': {n: {'corners_px': z['corners'], 'baseline_visible_ratio': float(z['baseline'])}
                  for n, z in zones.items()},
    }, header='# 環境相機站台角點（tl,tr,br,bl；影像像素）。由 env_vision_node 自動產生。\n')
    return p


# --------------------------------------------------------------------------- #
# 相機內參（camera_info YAML，usb_cam 的 camera_info_url 可直接讀取）
# --------------------------------------------------------------------------- #
def save_camera_info_yaml(path, camera_name: str, size_wh: Sequence[int], K: np.ndarray,
                          dist: np.ndarray, rms: Optional[float] = None) -> None:
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    P = np.hstack([K, np.zeros((3, 1))])
    header = '# 相機內參（camera_info 格式）。usb_cam: camera_info_url: file://<此檔絕對路徑>\n'
    if rms is not None:
        header += f'# 標定 RMS 重投影誤差: {rms:.4f} px\n'
    save_yaml(path, {
        'image_width': int(size_wh[0]),
        'image_height': int(size_wh[1]),
        'camera_name': camera_name,
        'camera_matrix': {'rows': 3, 'cols': 3, 'data': K.reshape(-1)},
        'distortion_model': 'plumb_bob',
        'distortion_coefficients': {'rows': 1, 'cols': int(dist.size), 'data': dist},
        'rectification_matrix': {'rows': 3, 'cols': 3, 'data': np.eye(3).reshape(-1)},
        'projection_matrix': {'rows': 3, 'cols': 4, 'data': P.reshape(-1)},
    }, header=header)


def load_camera_info_yaml(path) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    d = load_yaml(path)
    K = np.array(d['camera_matrix']['data'], dtype=np.float64).reshape(3, 3)
    dist = np.array(d['distortion_coefficients']['data'], dtype=np.float64).reshape(-1)
    return K, dist, (int(d['image_width']), int(d['image_height']))


class Undistorter:
    """依 camera_info YAML 去畸變；影像尺寸與標定尺寸不同時按比例縮放內參。"""

    def __init__(self, yaml_path):
        self.K, self.dist, self.calib_size = load_camera_info_yaml(yaml_path)
        self._maps = None
        self._size = None

    def apply(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        if self._maps is None or self._size != (w, h):
            sx, sy = w / self.calib_size[0], h / self.calib_size[1]
            K = self.K.copy()
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy
            self._maps = build_undistort_maps(K, self.dist, (w, h))
            self._size = (w, h)
        return undistort(bgr, self._maps)
