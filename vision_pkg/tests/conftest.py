"""pytest 共用設定與 fixture。

不需要 ROS：只依賴 numpy / OpenCV / PyYAML。從 vision_pkg/ 目錄執行  python3 -m pytest tests
"""
import sys
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))            # 讓 import vision_pkg 可用（不必先 colcon build）
sys.path.insert(0, str(Path(__file__).parent))  # 讓 import synthetic 可用

import synthetic as S  # noqa: E402
from make_test_images import ensure_images  # noqa: E402
from vision_pkg.calib_io import load_hsv_ranges  # noqa: E402
from vision_pkg.vision_core.color_zone import (MaskParams, OccupancyParams,  # noqa: E402
                                               ZoneFindParams)
from vision_pkg.vision_core.params_util import (dataclass_from_flat,  # noqa: E402
                                                load_ros_params_yaml)
from vision_pkg.vision_core.square_detect import SquareParams  # noqa: E402

CONFIG = PKG_ROOT / 'config'


@pytest.fixture(scope='session')
def data_dir():
    return ensure_images()


@pytest.fixture(scope='session')
def hsv(tmp_path_factory):
    """套件預設 hsv.yaml（calib_dir 指向空資料夾，確保不受本機調色結果影響）。"""
    return load_hsv_ranges(tmp_path_factory.mktemp('calib'))


@pytest.fixture(scope='session')
def env_flat():
    """env_vision.yaml + 舊合成場景用的 color 模式覆寫（三塊彩色站台）。"""
    return {**load_ros_params_yaml(CONFIG / 'env_vision.yaml'), **S.LEGACY_ENV_PARAMS}


@pytest.fixture(scope='session')
def field_flat():
    """env_vision.yaml 原樣（field 模式：黑色外框場地）。"""
    return load_ros_params_yaml(CONFIG / 'env_vision.yaml')


@pytest.fixture(scope='session')
def arm2_flat():
    return load_ros_params_yaml(CONFIG / 'arm2_vision.yaml')


@pytest.fixture(scope='session')
def env_cfg(env_flat):
    """env 節點相關參數物件（直接讀 config/env_vision.yaml，確保 YAML 與程式一致）。"""
    f = env_flat
    return {
        'zone_mp': dataclass_from_flat(MaskParams, f, 'zone_mask'),
        'zone_find': dataclass_from_flat(ZoneFindParams, f, 'zone_find'),
        'orange_mp': dataclass_from_flat(MaskParams, f, 'orange_mask'),
        'black_mp': dataclass_from_flat(MaskParams, f, 'black_mask'),
        'occupancy': dataclass_from_flat(OccupancyParams, f, 'occupancy'),
        'square': dataclass_from_flat(SquareParams, f, 'square'),
        'ppm': float(f['snapshot.warp_px_per_mm']),
        'erode': int(f['roi_erode_px']),
        'min_baseline': float(f['min_baseline_ratio']),
        'min_det': int(f['boot_min_detections']),
        'margin_px': float(f['ranker.confidence_margin_px']),
        'match_mm': float(f['match_threshold_mm']),
        'count_min_mm2': float(f['c_count.min_blob_area_mm2']),
    }


@pytest.fixture(scope='session')
def env_scenes(data_dir):
    """從 tests/data 讀入的靜態 PNG（缺少時 ensure_images 會先生成）。"""
    return {
        'empty': S.imread(data_dir / 'env_empty.png'),
        'a8': S.imread(data_dir / 'env_a_8objects.png'),
        'a7': S.imread(data_dir / 'env_a_7objects.png'),
        'b_arm': S.imread(data_dir / 'env_b_arm.png'),
    }
