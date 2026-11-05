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


def put_text_centered(img: np.ndarray, text: str, center: Tuple[float, float], color: Color = (255, 255, 255),
                      scale: float = 0.4, thick: int = 1) -> None:
    """以 center 為中心畫文字（用來把座標標在物件正下方/正中央）。"""
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    put_text(img, text, (center[0] - w / 2.0, center[1] + h / 2.0), color, scale, thick)
