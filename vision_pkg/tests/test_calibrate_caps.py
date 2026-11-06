"""calibrate_caps：尺量 -> 基座座標、瓶蓋編號、多影格穩定中心、合成影像端到端。"""
import sys

import cv2
import numpy as np
import pytest

from vision_pkg.tools import calibrate_caps as cc
from vision_pkg.vision_core.geometry import apply_homography, compute_homography


def test_field_to_base_uses_model_geometry():
    # 前緣上、A 區外側角落 -> (0.020, -0.200)
    assert cc.field_to_base(0, 0) == pytest.approx((0.020, -0.200))
    # A/B 分界、離前緣 100mm -> 正前方 0.12 m
    assert cc.field_to_base(100, 200) == pytest.approx((0.120, 0.0))


def test_order_caps_rows_then_columns():
    c = np.array([[300, 52], [100, 200], [100, 50], [300, 205]], float)
    assert list(cc.order_caps(c)) == [2, 0, 1, 3]


def test_stable_centers_median_and_warnings():
    a = np.array([[100, 100], [200, 200]], float)
    frames = [a + np.random.default_rng(i).normal(0, 0.3, a.shape) for i in range(9)] + [a[:1]]
    c, warn = cc.stable_centers(frames)
    assert c.shape == (2, 2) and np.allclose(c, a, atol=0.5)
    assert '不一致' in warn


def test_read_measure(tmp_path):
    p = tmp_path / 'm.txt'
    p.write_text('# front side\n10 20\n30,40  # 註解\n\n', encoding='utf-8')
    assert cc.read_measure(str(p)) == [(10, 20), (30, 40)]
    p.write_text('1 2 3\n', encoding='utf-8')
    with pytest.raises(ValueError):
        cc.read_measure(str(p))


def _scene(H_px_to_base, caps_base, size=(640, 480)):
    """白底藍色圓：依已知 homography 把基座座標的瓶蓋畫到影像上。"""
    img = np.full((size[1], size[0], 3), 225, np.uint8)
    px = apply_homography(np.linalg.inv(H_px_to_base), np.array(caps_base))
    for x, y in px:
        cv2.circle(img, (int(round(x)), int(round(y))), 25, (200, 90, 20), -1)   # BGR 藍
    return img


def test_end_to_end_offline_recovers_homography(tmp_path, monkeypatch):
    # 真值：俯視、0.4 mm/px，影像軸與基座軸互換且反向
    s = 0.0004
    H_true = np.array([[0, -s, 0.25], [-s, 0, -0.03], [0, 0, 1]], float)
    meas = [(60, 40), (60, 140), (150, 30), (150, 100), (210, 60), (210, 150)]   # front side mm
    base = [cc.field_to_base(f, sd) for f, sd in meas]
    img = _scene(H_true, base)
    cv2.imwrite(str(tmp_path / 'scene.png'), img)
    # 量測檔要依畫面編號排列：先算出每顆的像素位置與編號順序
    px = apply_homography(np.linalg.inv(H_true), np.array(base))
    order = cc.order_caps(px)
    (tmp_path / 'm.txt').write_text('\n'.join(f'{meas[i][0]} {meas[i][1]}' for i in order), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['calibrate_caps', '--image', str(tmp_path / 'scene.png'),
                                      '--measure', str(tmp_path / 'm.txt'), '--calib-dir', str(tmp_path),
                                      '--max-area-px', '40000'])
    assert cc.main() == 0
    from vision_pkg import calib_io
    h = calib_io.load_homography('arm1_homography', tmp_path)
    assert h['valid'] and h['unit'] == 'm'
    test_px = np.array([[320, 240], [100, 400]], float)
    assert np.allclose(apply_homography(h['H'], test_px), apply_homography(H_true, test_px), atol=0.0005)
    assert (tmp_path / 'arm1_caps_numbered.png').exists()
