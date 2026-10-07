"""debug 視窗的繪圖小工具（純 OpenCV；cv2.putText 不支援中文，畫面文字一律用 ASCII）。"""
from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np

Color = Tuple[int, int, int]


def put_text(img: np.ndarray, text: str, org: Tuple[float, float], color: Color = (255, 255, 255),
             scale: float = 0.5, thick: int = 1) -> None:
    """帶深色描邊的文字，在任何背景上都看得清楚。"""
    p = (int(org[0]), int(org[1]))
    cv2.putText(img, text, p, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, p, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_poly(img: np.ndarray, pts: Sequence, color: Color, thick: int = 2) -> None:
    p = np.round(np.asarray(pts, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [p], True, color, thick, cv2.LINE_AA)


def draw_corner_ids(img: np.ndarray, corners: np.ndarray, color: Color = (0, 255, 255)) -> None:
    """在四角點旁標 0..3（0 = 校正後影像的左上角），用來確認 warp_rotation_k。"""
    for i, c in enumerate(corners):
        cv2.circle(img, (int(c[0]), int(c[1])), 4, color, -1, cv2.LINE_AA)
        put_text(img, str(i), (c[0] + 6, c[1] - 6), color, 0.5, 1)
