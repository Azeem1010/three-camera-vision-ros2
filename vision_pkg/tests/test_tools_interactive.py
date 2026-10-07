"""hsv_tuner / calibrate_homography 互動流程測試：以「腳本化的假 GUI」取代 cv2 視窗、滑桿、滑鼠與鍵盤。"""
import builtins
import sys

import cv2
import numpy as np
import synthetic as S

from vision_pkg import calib_io
from vision_pkg.vision_core.geometry import apply_homography


class FakeGui:
    """script 的每一項代表一次 cv2.waitKey()：int = 按鍵；('click',x,y) = 滑鼠左鍵；('set',名稱,值) = 拖滑桿。
    script 用完後自動回傳 'q'（離開）。"""

    NAMES = ('namedWindow', 'imshow', 'waitKey', 'createTrackbar', 'getTrackbarPos', 'setTrackbarPos',
             'setMouseCallback', 'destroyAllWindows', 'destroyWindow', 'resize')

    def __init__(self, script):
        self.script = list(script)
        self.trackbars = {}
        self.mouse_cb = None
        self.frames = 0
        self.last_view = None

    def __enter__(self):
        self.saved = {n: getattr(cv2, n) for n in self.NAMES}
        cv2.namedWindow = lambda *a, **k: None
        cv2.destroyAllWindows = lambda *a, **k: None
        cv2.destroyWindow = lambda *a, **k: None
        cv2.createTrackbar = lambda name, win, v, mx, cb: self.trackbars.__setitem__(name, v)
        cv2.getTrackbarPos = lambda name, win: self.trackbars[name]
        cv2.setTrackbarPos = lambda name, win, v: self.trackbars.__setitem__(name, v)
        cv2.setMouseCallback = lambda win, cb: setattr(self, 'mouse_cb', cb)
        cv2.imshow = self._imshow
        cv2.waitKey = self._wait
        return self

    def __exit__(self, *exc):
        for n, f in self.saved.items():
            setattr(cv2, n, f)

    def _imshow(self, win, img):
        self.frames += 1
        self.last_view = img

    def _wait(self, ms=0):
        if not self.script:
            return ord('q')
        item = self.script.pop(0)
        if isinstance(item, int):
            return item
        if item[0] == 'click':
            self.mouse_cb(cv2.EVENT_LBUTTONDOWN, item[1], item[2], 0, None)
        elif item[0] == 'set':
            self.trackbars[item[1]] = item[2]
        return -1


def _argv(*args):
    old = sys.argv
    sys.argv = ['tool'] + list(args)
    return old


def test_hsv_tuner_switches_colors_and_saves_only_what_changed(tmp_path):
    from vision_pkg.tools import hsv_tuner
    img_path = tmp_path / 'env.png'
    S.imwrite(img_path, S.make_env_scene(a_objs=S.a_objects()))
    hsv_tuner.require_display = lambda: None
    base = calib_io.load_hsv_ranges(tmp_path)
    script = [-1, ('set', 'H low', 8), -1, ord('s'),          # 調 orange 並存
              ord('2'), -1,                                    # 切到 green：滑桿應換成 green 的值
              ('set', 'H high', 90), -1, ('click', 120, 90), -1, ord('m'), ord('p'),
              ord('a'),                                        # 只存「有改過」的顏色
              ord('3'), -1, ord('1'), -1]                      # 切換但不修改 -> 不該被存
    old = _argv('--image', str(img_path), '--color', 'orange', '--calib-dir', str(tmp_path))
    try:
        with FakeGui(script) as gui:
            seen = {}
            orig = gui._wait

            def spy(ms=0):
                k = orig(ms)
                if k == ord('2'):
                    seen['switch'] = True
                elif seen.get('switch') and 'green_low_h' not in seen:
                    seen['green_low_h'] = gui.trackbars['H low']
                return k
            cv2.waitKey = spy
            hsv_tuner.main()
            assert gui.frames > 5 and gui.last_view is not None
    finally:
        sys.argv = old
    assert seen['green_low_h'] == base['green'].lower[0]       # 切色時滑桿已換成 green 的數值
    got = calib_io.load_hsv_ranges(tmp_path)
    assert got['orange'].lower[0] == 8 and got['orange'].upper == base['orange'].upper
    assert got['green'].upper[0] == 90
    saved = calib_io.load_yaml(tmp_path / 'hsv.yaml')
    assert set(saved) == {'orange', 'green'}                    # blue / c_zone / black 沒改就不寫入


def test_calibrate_homography_interactive_points_and_save(tmp_path):
    from vision_pkg.tools import calibrate_homography as ch
    img_path = tmp_path / 'home.png'
    S.imwrite(img_path, S.make_env_scene())
    ch.require_display = lambda: None
    H_true = np.array([[0.0010, 0.0001, 0.05], [-0.0001, 0.0012, -0.08], [1e-7, 2e-7, 1.0]])
    px = [(100, 100), (500, 100), (500, 400), (100, 400), (300, 250), (200, 330)]
    world = apply_homography(H_true, px)
    answers = iter([f'{x:.6f} {y:.6f}' for x, y in world] + ['garbage'])
    real_input = builtins.input
    builtins.input = lambda prompt='': next(answers)
    script = [-1, ord(' ')] + [('click', x, y) for x, y in px[:3]] + [-1, ord('c'),    # 3 點：不夠 4 點
                                                                       ] + [('click', x, y) for x, y in px[3:]] + [
        -1, ord('u'), ('click', *px[5]), -1,                                          # 刪掉最後一點再重點
        ord('c'), ord('s')]
    old = _argv('--target', 'arm1', '--image', str(img_path), '--calib-dir', str(tmp_path))
    try:
        with FakeGui(script) as gui:
            ch.main()
    finally:
        sys.argv = old
        builtins.input = real_input
    got = calib_io.load_homography('arm1_homography', tmp_path)
    assert got['valid'] and got['unit'] == 'm' and got['rms_error'] < 1e-4
    assert np.allclose(apply_homography(got['H'], [(320, 240)]), apply_homography(H_true, [(320, 240)]), atol=1e-4)


def test_calibrate_homography_interactive_corner_mode_needs_no_typing(tmp_path):
    from vision_pkg.tools import calibrate_homography as ch
    img_path = tmp_path / 'home.png'
    S.imwrite(img_path, S.make_env_scene())
    ch.require_display = lambda: None
    corners = [(50, 250), (250, 256), (246, 448), (44, 442)]
    real_input = builtins.input
    builtins.input = lambda prompt='': (_ for _ in ()).throw(AssertionError('--corners 模式不該要求輸入座標'))
    script = [-1, ord(' ')] + [('click', x, y) for x, y in corners] + [-1, ord('s')]
    old = _argv('--target', 'a_station', '--corners', '--station-size-mm', '200', '200', '--image',
                str(img_path), '--calib-dir', str(tmp_path))
    try:
        with FakeGui(script):
            ch.main()
    finally:
        sys.argv = old
        builtins.input = real_input
    got = calib_io.load_homography('a_station_homography', tmp_path)
    assert got['valid'] and got['unit'] == 'mm'
    assert np.allclose(apply_homography(got['H'], [(246, 448), (50, 250)]), [(200, 200), (0, 0)], atol=0.05)
