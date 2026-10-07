"""node_common：去彈跳、影格緩衝、最新結果容器、影像訊息轉換、參數讀取、service 內呼叫。"""
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from fake_ros import FakeClient
from vision_pkg.node_common import (Debouncer, FlatParams, FrameBuffer, LatestValue, call_service,
                                    image_msg_to_bgr)
from vision_pkg.vision_core.params_util import dataclass_from_flat
from vision_pkg.vision_core.square_detect import SquareParams


def test_debouncer_requires_consecutive_samples():
    d = Debouncer(n_to_true=3, n_to_false=3)
    assert d.update(False) is False            # 第一筆直接採用
    assert [d.update(True) for _ in range(3)] == [False, False, True]
    assert d.update(False) is True and d.update(True) is True   # 中途被打斷，計數重來
    assert [d.update(False) for _ in range(3)] == [True, True, False]


def test_debouncer_asymmetric_and_conservative_initial_state():
    arm = Debouncer(n_to_true=1, n_to_false=5, initial=True)    # 手臂：預設在場、離開要連續確認
    assert arm.update(False) is True
    assert [arm.update(False) for _ in range(4)] == [True, True, True, False]
    assert arm.update(True) is True                              # 再出現：立刻採信


def test_frame_buffer_wait_fresh_only_returns_frames_after_the_call():
    fb = FrameBuffer(maxlen=6)
    old = np.zeros((4, 4, 3), np.uint8)
    fb.push(old)
    assert fb.wait_fresh(1, 0.05) == []                         # 舊影格不算
    out = {}

    def waiter():
        out['f'] = fb.wait_fresh(3, 2.0)

    t = threading.Thread(target=waiter)
    t.start()
    for i in range(3):
        time.sleep(0.05)
        fb.push(np.full((4, 4, 3), i + 1, np.uint8))
    t.join(3)
    assert [int(f[0, 0, 0]) for f in out['f']] == [1, 2, 3]
    assert fb.latest()[0] == 4 and len(fb.latest_n(2)) == 2 and fb.age() < 1.0


def test_latest_value_fresh_stale_and_empty():
    lv = LatestValue()
    v, why = lv.wait_fresh(1.0, 0.05)
    assert v is None and '無結果' in why
    lv.set('x')
    assert lv.wait_fresh(1.0, 0.05) == ('x', '')
    time.sleep(0.15)
    v, why = lv.wait_fresh(0.1, 0.05)
    assert v is None and '過期' in why
    threading.Timer(0.1, lambda: lv.set('y')).start()           # 等待期間有新結果進來
    assert lv.wait_fresh(0.05, 1.0) == ('y', '')
    lv.clear()
    assert lv.peek() == (None, None)


def _msg(arr, enc, step=None):
    h, w = arr.shape[:2]
    ch = 1 if arr.ndim == 2 else arr.shape[2]
    return SimpleNamespace(height=h, width=w, encoding=enc, step=step or w * ch, data=arr.tobytes())


def test_image_conversion_rgb_bgr_mono_yuyv_and_padding():
    bgr = np.random.RandomState(0).randint(0, 255, (6, 8, 3), np.uint8)
    assert np.array_equal(image_msg_to_bgr(_msg(bgr, 'bgr8')), bgr)
    assert np.array_equal(image_msg_to_bgr(_msg(np.ascontiguousarray(bgr[:, :, ::-1]), 'rgb8')), bgr)
    gray = np.random.RandomState(1).randint(0, 255, (6, 8), np.uint8)
    out = image_msg_to_bgr(_msg(gray, 'mono8'))
    assert out.shape == (6, 8, 3) and np.array_equal(out[:, :, 0], gray)
    # step 比 width*3 大（每列有 padding）
    padded = np.zeros((6, 8 * 3 + 5), np.uint8)
    padded[:, :24] = bgr.reshape(6, 24)
    m = SimpleNamespace(height=6, width=8, encoding='bgr8', step=29, data=padded.tobytes())
    assert np.array_equal(image_msg_to_bgr(m), bgr)
    yuyv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)                  # 只驗證形狀與可轉換（YUYV 有色度子取樣）
    packed = np.zeros((6, 8, 2), np.uint8)
    packed[:, :, 0] = yuyv[:, :, 0]
    packed[:, 0::2, 1] = yuyv[:, 0::2, 1]
    packed[:, 1::2, 1] = yuyv[:, 0::2, 2]
    assert image_msg_to_bgr(_msg(packed, 'yuv422_yuy2')).shape == (6, 8, 3)


def test_image_conversion_errors():
    with pytest.raises(ValueError):
        image_msg_to_bgr(SimpleNamespace(height=2, width=2, encoding='16UC1', step=4, data=bytes(8)))
    with pytest.raises(ValueError):
        image_msg_to_bgr(SimpleNamespace(height=4, width=4, encoding='bgr8', step=12, data=bytes(10)))
    with pytest.raises(ValueError):
        image_msg_to_bgr(SimpleNamespace(height=0, width=0, encoding='bgr8', step=0, data=b''))


def test_flat_params_reports_missing_names():
    p = FlatParams({'a': 1, 'b': '2.5', 'c': [1, 2], 'd': 'true', 'square.min_area_px': 1.0})
    assert (p.i('a'), p.f('b'), p.fl('c'), p.b('d')) == (1, 2.5, [1.0, 2.0], True)
    with pytest.raises(KeyError, match='zzz'):
        p.f('zzz')
    with pytest.raises(KeyError, match='max_area_px'):
        p.group(SquareParams, 'square')


def test_dataclass_from_flat_coerces_types():
    flat = {'s.min_area_px': 5, 's.max_area_px': '9', 's.approx_eps_ratio': 0.04, 's.max_aspect': 1.15,
            's.min_fill': 0.85, 's.reject_border_touch': 'true', 's.border_px': 2.0, 's.side_method': 'rect'}
    sp = dataclass_from_flat(SquareParams, flat, 's')
    assert sp.min_area_px == 5.0 and sp.max_area_px == 9.0 and sp.reject_border_touch is True
    assert sp.border_px == 2 and isinstance(sp.border_px, int)


def test_call_service_success_not_ready_and_timeout():
    ok = FakeClient(None, '/x', None)
    ok.handler = lambda req: 'resp'
    assert call_service(ok, 'req', 1.0) == ('resp', '')

    down = FakeClient(None, '/x', None)
    r, err = call_service(down, 'req', 0.15)
    assert r is None and '未就緒' in err

    slow = FakeClient(None, '/x', None)
    slow.handler = lambda req: time.sleep(0.6) or 'late'
    t0 = time.time()
    r, err = call_service(slow, 'req', 0.2)
    assert r is None and '逾時' in err and time.time() - t0 < 0.5 and slow.removed == 1

    boom = FakeClient(None, '/x', None)

    def raiser(req):
        raise RuntimeError('boom')
    boom.handler = raiser
    r, err = call_service(boom, 'req', 1.0)
    assert r is None and 'boom' in err
