"""藍色圓形瓶蓋偵測。"""
import cv2
import numpy as np

from vision_pkg.vision_core.cap_detect import (CapParams, analyze_caps, describe_rejected, detect_caps,
                                               nearest_neighbor_dist)

P = CapParams(min_area_px=50, max_area_px=3000, min_circularity=0.85, min_fill=0.7,
              reject_border_touch=True, border_px=2)


def _mask(draw):
    m = np.zeros((200, 300), np.uint8)
    draw(m)
    return m


def test_circle_is_a_cap_with_correct_diameter():
    caps, rej = detect_caps(_mask(lambda m: cv2.circle(m, (100, 80), 15, 255, -1)), P)
    assert len(caps) == 1 and not rej
    c = caps[0]
    assert abs(c.cx - 100) < 0.5 and abs(c.cy - 80) < 0.5
    assert abs(c.diameter - 30) < 1.0 and c.circularity > 0.95


def test_square_bar_touching_and_merged_blobs_are_rejected_with_reasons():
    def draw(m):
        cv2.rectangle(m, (20, 20), (50, 50), 255, -1)           # 正方形：圓度 ~0.785
        cv2.rectangle(m, (80, 20), (160, 30), 255, -1)          # 細長條
        cv2.circle(m, (298, 100), 12, 255, -1)                  # 被影像邊緣截斷
        cv2.circle(m, (100, 150), 30, 255, -1)                  # 太大（兩顆相連）
        cv2.circle(m, (140, 150), 30, 255, -1)
    caps, rej = detect_caps(_mask(draw), P)
    assert not caps and len(rej) == 4
    text = describe_rejected(rej, max_items=4)
    assert '圓度' in text and '碰到邊界' in text and '面積過大' in text


def test_tiny_noise_is_ignored():
    caps = analyze_caps(_mask(lambda m: cv2.circle(m, (50, 50), 2, 255, -1)), P)
    assert caps == []


def test_nearest_neighbor_dist():
    d = nearest_neighbor_dist(np.array([[0, 0], [3, 4], [100, 0]], float))
    assert np.allclose(d, [5, 5, np.hypot(97, 4)])
    assert np.isinf(nearest_neighbor_dist(np.array([[1.0, 1.0]]))).all()
