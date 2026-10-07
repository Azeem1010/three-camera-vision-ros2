"""幾何工具：角點排序、homography、多邊形遮罩、去畸變（純 numpy / OpenCV）。"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def order_corners(pts) -> np.ndarray:
    """把 4 個角點排成 [左上, 右上, 右下, 左下]（影像座標，順時針）。

    以質心角度排序（影像 y 向下，角度遞增 = 順時針），再把「x+y 最小」者旋轉到第一個。
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    ordered = pts[np.argsort(ang)]
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0).astype(np.float32)


def rotate_corners(corners: np.ndarray, k: int) -> np.ndarray:
    """把角點順序轉 k 個位置（k=0..3）。用來決定「透視校正後影像的左上角」對應哪個實體角落。"""
    return np.roll(np.asarray(corners, dtype=np.float32), -(int(k) % 4), axis=0)


def polygon_area(pts) -> float:
    """多邊形面積（shoelace 公式）。"""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def apply_homography(H: np.ndarray, pts) -> np.ndarray:
    """對 (N,2) 點套用 3x3 homography，回傳 (N,2) float64。"""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, np.asarray(H, dtype=np.float64)).reshape(-1, 2)


def plane_homography(corners: np.ndarray, width_mm: float, height_mm: float) -> np.ndarray:
    """影像像素 -> 站台平面座標（mm，原點為左上角，x 向右、y 向下）。corners 須為 tl,tr,br,bl。"""
    dst = np.float32([[0, 0], [width_mm, 0], [width_mm, height_mm], [0, height_mm]])
    return cv2.getPerspectiveTransform(np.asarray(corners, dtype=np.float32), dst)


def compute_homography(src, dst, ransac_thresh: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """由對應點求 homography，回傳 (H, 每點重投影誤差，單位同 dst)。

    ransac_thresh=None 用最小平方（點少時較穩）；給數值則用 RANSAC。至少 4 點。
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(src) != len(dst) or len(src) < 4:
        raise ValueError(f'至少需要 4 組對應點（目前 {len(src)} 組）')
    if ransac_thresh is None:
        H, _ = cv2.findHomography(src, dst, 0)
    else:
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, float(ransac_thresh))
    if H is None:
        raise ValueError('findHomography 失敗（點共線或退化）')
    err = np.linalg.norm(apply_homography(H, src) - dst, axis=1)
    return H, err


def polygon_mask(shape_hw, poly, erode_px: int) -> np.ndarray:
    """多邊形（凸）填滿遮罩，可向內收縮 erode_px 像素以避開邊緣雜訊。"""
    m = np.zeros(tuple(shape_hw[:2]), np.uint8)
    cv2.fillConvexPoly(m, np.round(np.asarray(poly)).astype(np.int32).reshape(-1, 1, 2), 255)
    if erode_px > 0:
        k = 2 * int(erode_px) + 1
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return m


def build_undistort_maps(K: np.ndarray, dist: np.ndarray, size_wh) -> Tuple[np.ndarray, np.ndarray]:
    """建立去畸變 remap 表（保留原 K，視角不縮放；邊緣可能出現少量黑邊）。"""
    w, h = int(size_wh[0]), int(size_wh[1])
    return cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_16SC2)


def undistort(bgr: np.ndarray, maps) -> np.ndarray:
    return cv2.remap(bgr, maps[0], maps[1], cv2.INTER_LINEAR)
