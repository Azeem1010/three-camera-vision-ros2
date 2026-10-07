"""geometry：角點排序、homography、遮罩。"""
import itertools

import cv2
import numpy as np
import pytest

from vision_pkg.vision_core.geometry import (apply_homography, compute_homography, order_corners,
                                             plane_homography, polygon_area, polygon_mask, rotate_corners)


def test_order_corners_is_independent_of_input_order():
    expected = np.float32([[50, 250], [250, 256], [246, 448], [44, 442]])  # tl,tr,br,bl
    for perm in itertools.permutations(range(4)):
        got = order_corners(expected[list(perm)])
        assert np.allclose(got, expected), f'perm={perm}'


def test_order_corners_rotated_quad_starts_from_top_left():
    # 順時針轉 30° 的矩形：排序後第一點應是 x+y 最小者，且依序為順時針
    c = np.float32([[0, 0], [100, 0], [100, 60], [0, 60]]) - np.float32([50, 30])
    a = np.deg2rad(30)
    R = np.float32([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = c @ R.T + np.float32([200, 200])
    out = order_corners(pts)
    assert out.sum(axis=1).argmin() == 0
    cross = np.cross(out[1] - out[0], out[2] - out[1])
    assert cross > 0  # 影像座標（y 向下）順時針


def test_rotate_corners_shifts_origin():
    c = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    assert np.allclose(rotate_corners(c, 0), c)
    assert np.allclose(rotate_corners(c, 1)[0], c[1])
    assert np.allclose(rotate_corners(c, 2)[0], c[2])
    assert np.allclose(rotate_corners(c, 4), c)


def test_polygon_area():
    assert abs(polygon_area([[0, 0], [10, 0], [10, 5], [0, 5]]) - 50.0) < 1e-9


def test_plane_homography_maps_corners_to_plane_rectangle():
    corners = np.float32([[50, 250], [250, 256], [246, 448], [44, 442]])
    H = plane_homography(corners, 200.0, 100.0)
    out = apply_homography(H, corners)
    assert np.allclose(out, [[0, 0], [200, 0], [200, 100], [0, 100]], atol=1e-3)


def test_compute_homography_recovers_known_mapping():
    H_true = np.array([[0.9, 0.05, 12.0], [-0.04, 1.1, -7.0], [1e-4, -2e-4, 1.0]])
    rng = np.random.RandomState(1)
    src = rng.uniform(0, 600, size=(12, 2))
    dst = apply_homography(H_true, src) + rng.randn(12, 2) * 0.05
    H, err = compute_homography(src, dst)
    assert err.max() < 0.3
    probe = np.array([[320.0, 240.0]])
    assert np.allclose(apply_homography(H, probe), apply_homography(H_true, probe), atol=0.3)


def test_compute_homography_needs_four_points():
    with pytest.raises(ValueError):
        compute_homography([[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 0], [1, 1]])


def test_polygon_mask_erode_shrinks_area():
    poly = [[10, 10], [90, 10], [90, 90], [10, 90]]
    full = cv2.countNonZero(polygon_mask((100, 100), poly, 0))
    eroded = cv2.countNonZero(polygon_mask((100, 100), poly, 5))
    assert eroded < full
    assert abs(eroded - 71 * 71) < 150  # 81x81 的方塊，四邊各內縮 5px -> 71x71
