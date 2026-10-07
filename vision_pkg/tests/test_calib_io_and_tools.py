"""校正檔 I/O，以及標定工具的離線（無視窗）模式。"""
import sys
import time

import cv2
import numpy as np
import pytest

from vision_pkg import calib_io
from vision_pkg.vision_core.color_zone import HsvRange
from vision_pkg.vision_core.geometry import apply_homography


def test_resolve_calib_dir_priority():
    import os
    old = os.environ.pop(calib_io.CALIB_ENV_VAR, None)
    try:
        assert str(calib_io.resolve_calib_dir('/tmp/x')).replace('\\', '/').endswith('/tmp/x')
        os.environ[calib_io.CALIB_ENV_VAR] = '/tmp/from_env'
        assert str(calib_io.resolve_calib_dir('')).replace('\\', '/').endswith('/tmp/from_env')
        assert str(calib_io.resolve_calib_dir('/tmp/x')).replace('\\', '/').endswith('/tmp/x')
        del os.environ[calib_io.CALIB_ENV_VAR]
        assert str(calib_io.resolve_calib_dir('')).replace('\\', '/').endswith('.ros/vision_calib')
    finally:
        os.environ.pop(calib_io.CALIB_ENV_VAR, None)
        if old is not None:
            os.environ[calib_io.CALIB_ENV_VAR] = old


def test_homography_roundtrip_and_template_fallback(tmp_path):
    H = np.array([[1.1, 0.02, 3.0], [0.01, 0.9, -4.0], [1e-4, 0, 1.0]])
    px = [[0, 0], [100, 0], [100, 100], [0, 100]]
    calib_io.save_homography(tmp_path / 'arm1_homography.yaml', H, 'm', (640, 480), px,
                             apply_homography(H, px), 0.0012, False)
    got = calib_io.load_homography('arm1_homography', tmp_path)
    assert got['valid'] and got['unit'] == 'm' and np.allclose(got['H'], H) and got['rms_error'] == 0.0012
    # calib_dir 沒有 -> 退回套件內的空範本（valid=false）
    tpl = calib_io.load_homography('arm2_homography', tmp_path)
    assert not tpl['valid'] and np.allclose(tpl['H'], np.eye(3))
    with pytest.raises(FileNotFoundError):
        calib_io.load_homography('nope', tmp_path)


def test_zones_roundtrip(tmp_path):
    assert calib_io.load_zones(tmp_path) is None
    zones = {'a': {'corners': np.float32([[1, 2], [3, 4], [5, 6], [7, 8]]), 'baseline': 0.93},
             'b': {'corners': np.float32([[10, 20], [30, 40], [50, 60], [70, 80]]), 'baseline': 0.9}}
    calib_io.save_zones(tmp_path, (640, 480), zones)
    got = calib_io.load_zones(tmp_path)
    assert set(got) == {'a', 'b'} and np.allclose(got['a']['corners'], zones['a']['corners'])
    assert got['b']['baseline'] == 0.9
    (tmp_path / 'zones.yaml').write_text('valid: false\nzones: {}\n', encoding='utf-8')
    assert calib_io.load_zones(tmp_path) is None


def test_hsv_override_is_per_color_and_reloads(tmp_path):
    base = calib_io.load_hsv_ranges(tmp_path)
    assert set(base) >= {'blue', 'green', 'c_zone', 'orange', 'black'}
    new = HsvRange((1, 2, 3), (4, 5, 6))
    calib_io.save_hsv_range(tmp_path, 'orange', new)
    over = calib_io.load_hsv_ranges(tmp_path)
    assert over['orange'] == new and over['blue'] == base['blue']        # 只覆寫 orange
    calib_io.save_hsv_range(tmp_path, 'blue', HsvRange((9, 9, 9), (10, 10, 10)))
    assert calib_io.load_hsv_ranges(tmp_path)['orange'] == new            # 先前的覆寫不會被洗掉


def test_reloadable_config_detects_changes_and_keeps_old_value_on_error(tmp_path):
    f = tmp_path / 'v.yaml'
    f.write_text('x: 1\n', encoding='utf-8')
    rc = calib_io.ReloadableConfig(lambda: calib_io.load_yaml(f), [f], period_s=0.0)
    assert rc.value == {'x': 1} and rc.error == ''
    assert rc.poll() is False                                              # 沒變
    time.sleep(0.02)
    f.write_text('x: 2\ny: [3]\n', encoding='utf-8')
    assert rc.poll() is True and rc.value == {'x': 2, 'y': [3]}
    time.sleep(0.02)
    f.write_text('x: [unclosed\n', encoding='utf-8')                       # 壞掉的 YAML
    assert rc.poll() is False and rc.value == {'x': 2, 'y': [3]} and rc.error
    missing = calib_io.ReloadableConfig(lambda: calib_io.load_yaml(tmp_path / 'none.yaml'),
                                        [tmp_path / 'none.yaml'], 0.0)
    assert missing.value is None and missing.error


def test_save_yaml_is_atomic_and_leaves_no_temp_files(tmp_path):
    for i in range(3):
        calib_io.save_yaml(tmp_path / 'a.yaml', {'i': i, 'arr': np.arange(3), 'f': np.float32(0.5)})
    assert calib_io.load_yaml(tmp_path / 'a.yaml') == {'i': 2, 'arr': [0, 1, 2], 'f': 0.5}
    assert [p.name for p in tmp_path.iterdir()] == ['a.yaml']


# --------------------------------------------------------------------------- #
# 標定工具（離線模式）
# --------------------------------------------------------------------------- #
def _run(module_main, argv):
    old = sys.argv
    sys.argv = ['tool'] + argv
    try:
        module_main()
        return 0
    except SystemExit as e:
        return 0 if e.code in (None, 0) else e.code
    finally:
        sys.argv = old


def test_calibrate_homography_batch_mode_recovers_known_mapping(tmp_path):
    from vision_pkg.tools import calibrate_homography as ch
    H_true = np.array([[0.0012, 0.0001, 0.05], [-0.0001, 0.0011, -0.08], [1e-7, 2e-7, 1.0]])
    rng = np.random.RandomState(3)
    px = rng.uniform([40, 40], [600, 440], size=(10, 2))
    world = apply_homography(H_true, px) + rng.randn(10, 2) * 0.0003      # 0.3mm 量測雜訊
    (tmp_path / 'px.txt').write_text('# u v\n' + '\n'.join(f'{u:.2f} {v:.2f}' for u, v in px), encoding='utf-8')
    (tmp_path / 'w.txt').write_text('\n'.join(f'{x:.5f}, {y:.5f}' for x, y in world), encoding='utf-8')
    code = _run(ch.main, ['--target', 'arm1', '--pixels', str(tmp_path / 'px.txt'), '--points',
                          str(tmp_path / 'w.txt'), '--calib-dir', str(tmp_path)])
    assert code == 0
    got = calib_io.load_homography('arm1_homography', tmp_path)
    assert got['valid'] and got['unit'] == 'm' and got['rms_error'] < 0.001
    probe = np.array([[320.0, 240.0]])
    assert np.allclose(apply_homography(got['H'], probe), apply_homography(H_true, probe), atol=0.001)


def test_calibrate_homography_a_station_corners_and_dry_run(tmp_path):
    from vision_pkg.tools import calibrate_homography as ch
    (tmp_path / 'px.txt').write_text('50 250\n250 256\n246 448\n44 442\n', encoding='utf-8')
    assert _run(ch.main, ['--target', 'a_station', '--corners', '--station-size-mm', '200', '200',
                          '--pixels', str(tmp_path / 'px.txt'), '--calib-dir', str(tmp_path), '--dry-run']) == 0
    assert not (tmp_path / 'a_station_homography.yaml').exists()          # dry-run 不存檔
    assert _run(ch.main, ['--target', 'a_station', '--corners', '--station-size-mm', '200', '200',
                          '--pixels', str(tmp_path / 'px.txt'), '--calib-dir', str(tmp_path)]) == 0
    got = calib_io.load_homography('a_station_homography', tmp_path)
    assert got['unit'] == 'mm'
    assert np.allclose(apply_homography(got['H'], [[246, 448]]), [[200, 200]], atol=0.01)
    assert _run(ch.main, ['--target', 'arm1', '--corners']) != 0           # --corners 只給 a_station


def test_calibrate_homography_rejects_mismatched_counts(tmp_path):
    from vision_pkg.tools import calibrate_homography as ch
    (tmp_path / 'px.txt').write_text('1 1\n2 2\n3 3\n4 4\n', encoding='utf-8')
    (tmp_path / 'w.txt').write_text('1 1\n2 2\n3 3\n', encoding='utf-8')
    assert _run(ch.main, ['--target', 'arm2', '--pixels', str(tmp_path / 'px.txt'), '--points',
                          str(tmp_path / 'w.txt'), '--calib-dir', str(tmp_path)]) != 0


def _chessboard_views(n, K, cols=9, rows=6, square_mm=25.0, seed=0):
    """用已知內參 K 合成 n 張不同姿態的棋盤格影像（無鏡頭畸變）。"""
    ppm = 4                                                    # 棋盤格影像解析度 px/mm
    nx, ny = cols + 1, rows + 1                                # 方格數
    m = 2                                                      # 外圍白邊（方格數）
    board = np.full(((ny + 2 * m) * 25 * ppm, (nx + 2 * m) * 25 * ppm), 255, np.uint8)
    s = int(square_mm * ppm)
    for r in range(ny):
        for c in range(nx):
            if (r + c) % 2 == 0:
                y0, x0 = (r + m) * s, (c + m) * s
                board[y0:y0 + s, x0:x0 + s] = 0
    rng = np.random.RandomState(seed)
    center_mm = np.array([nx * square_mm / 2, ny * square_mm / 2, 0.0])   # 棋盤格中心（board 座標，mm）
    views = []
    for _ in range(n):
        rvec = rng.uniform(-0.45, 0.45, 3)
        R, _ = cv2.Rodrigues(rvec)
        t = np.array([rng.uniform(-40, 40), rng.uniform(-30, 30), rng.uniform(550, 750)])
        # board 平面 (X,Y,0)（原點在左上內角） -> 相機座標：R @ (P - center) + t
        origin_cam = R @ (-center_mm) + t
        Hm = K @ np.column_stack([R[:, 0], R[:, 1], origin_cam])           # board mm -> pixel
        S = np.array([[1.0 / ppm, 0, -m * square_mm], [0, 1.0 / ppm, -m * square_mm], [0, 0, 1]])  # 影像 px -> board mm
        img = cv2.warpPerspective(board, Hm @ S, (640, 480), flags=cv2.INTER_LINEAR, borderValue=255)
        views.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return views


def test_calibrate_intrinsics_offline_recovers_focal_length_and_saves_camera_info(tmp_path):
    from vision_pkg.tools import calibrate_intrinsics as ci
    K = np.array([[620.0, 0, 322.0], [0, 618.0, 238.0], [0, 0, 1.0]])
    img_dir = tmp_path / 'chess'
    img_dir.mkdir()
    for i, v in enumerate(_chessboard_views(14, K)):
        ok, buf = cv2.imencode('.png', v)
        buf.tofile(str(img_dir / f'{i:02d}.png'))
    code = _run(ci.main, ['--camera', 'arm1', '--cols', '9', '--rows', '6', '--square-mm', '25',
                          '--image-dir', str(img_dir), '--calib-dir', str(tmp_path)])
    assert code == 0
    out = tmp_path / 'arm1_cam_intrinsics.yaml'
    assert out.exists()
    d = calib_io.load_yaml(out)                                           # camera_info 標準欄位
    assert d['camera_name'] == 'arm1_cam' and (d['image_width'], d['image_height']) == (640, 480)
    assert d['distortion_model'] == 'plumb_bob' and len(d['camera_matrix']['data']) == 9
    assert d['projection_matrix']['cols'] == 4 and d['rectification_matrix']['data'][0] == 1.0
    Kg, dist, size = calib_io.load_camera_info_yaml(out)
    assert size == (640, 480)
    assert abs(Kg[0, 0] - 620.0) / 620.0 < 0.03 and abs(Kg[1, 1] - 618.0) / 618.0 < 0.03
    assert abs(Kg[0, 2] - 322.0) < 6.0 and abs(Kg[1, 2] - 238.0) < 6.0
    assert np.abs(dist).max() < 0.15                                      # 合成影像沒有畸變

    und = calib_io.Undistorter(out)                                       # 去畸變器可載入並套用
    frame = np.random.RandomState(0).randint(0, 255, (480, 640, 3), np.uint8)
    res = und.apply(frame)
    assert res.shape == frame.shape
    assert np.mean(np.abs(res[100:380, 100:540].astype(int) - frame[100:380, 100:540].astype(int))) < 40


def test_calibrate_intrinsics_needs_enough_images(tmp_path):
    from vision_pkg.tools import calibrate_intrinsics as ci
    K = np.array([[620.0, 0, 322.0], [0, 618.0, 238.0], [0, 0, 1.0]])
    d = tmp_path / 'few'
    d.mkdir()
    for i, v in enumerate(_chessboard_views(3, K)):
        cv2.imencode('.png', v)[1].tofile(str(d / f'{i}.png'))
    assert _run(ci.main, ['--camera', 'env', '--cols', '9', '--rows', '6', '--square-mm', '25',
                          '--image-dir', str(d), '--calib-dir', str(tmp_path)]) != 0
    assert not (tmp_path / 'env_cam_intrinsics.yaml').exists()
