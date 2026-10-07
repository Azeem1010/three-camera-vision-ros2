"""橘色正方形偵測與邊長量測（純函式，不含 ROS）。

判定條件（皆由參數傳入）：
  1. 凸包 approxPolyDP 後剛好 4 個頂點
  2. minAreaRect 長寬比 <= max_aspect
  3. 填滿率（輪廓面積 / minAreaRect 面積）>= min_fill
  4. 面積在 [min_area_px, max_area_px]，且不碰到遮罩邊界（可關閉）
不符合者保留在 rejected 清單並附上原因，方便上層回報而不是默默猜測。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class SquareParams:
    """偵測所在影像的單位：env/arm2 為「校正後像素」，arm1 為「原始影像像素」。"""
    min_area_px: float          # 小於此面積視為雜訊，直接忽略
    max_area_px: float          # 大於此面積視為多個物件黏在一起 -> 排除
    approx_eps_ratio: float     # approxPolyDP epsilon（佔凸包周長比例）
    max_aspect: float           # minAreaRect 長寬比上限（1.15）
    min_fill: float             # 填滿率下限（0.85）
    reject_border_touch: bool   # 碰到影像邊界（被截斷）的區塊是否排除
    border_px: int              # 遮罩邊緣這幾像素視為無效（避免站台邊緣的桌面/陰影雜訊）
    side_method: str            # 'rect' = minAreaRect 長寬平均；'area' = sqrt(面積)


@dataclass
class Blob:
    """一個橘色連通塊的量測結果。"""
    cx: float
    cy: float
    side: float                 # 邊長（影像像素）
    aspect: float
    fill: float
    n_vertices: int
    area: float
    box: np.ndarray             # minAreaRect 四角 (4,2)
    angle: float
    touches_border: bool
    too_large: bool             # 面積超過 max_area_px（疑似多個物件相連）
    is_square: bool
    reject_reason: str          # is_square=True 時為空字串
    contour: np.ndarray = field(repr=False, default=None)


def clear_border(mask: np.ndarray, border_px: int) -> np.ndarray:
    """把遮罩最外圈 border_px 像素清為 0（回傳副本）。"""
    b = int(border_px)
    if b <= 0:
        return mask
    m = mask.copy()
    m[:b, :] = 0
    m[-b:, :] = 0
    m[:, :b] = 0
    m[:, -b:] = 0
    return m


def analyze_blobs(mask: np.ndarray, p: SquareParams) -> List[Blob]:
    """量測遮罩內所有面積 >= min_area_px 的連通塊（含不合格者），依 (y, x) 由上到下、左到右排序。"""
    h, w = mask.shape[:2]
    b = max(0, int(p.border_px))
    cnts, _ = cv2.findContours(clear_border(mask, b), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs: List[Blob] = []
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < p.min_area_px:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        touches = x <= b or y <= b or (x + bw) >= (w - b) or (y + bh) >= (h - b)

        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        n_vert = len(cv2.approxPolyDP(hull, p.approx_eps_ratio * peri, True))

        rect = cv2.minAreaRect(c)
        (_, _), (rw, rh), angle = rect
        long_side, short_side = max(rw, rh), min(rw, rh)
        aspect = long_side / short_side if short_side > 1e-6 else float('inf')
        fill = area / (rw * rh) if rw * rh > 1e-6 else 0.0
        side = math.sqrt(area) if p.side_method == 'area' else (rw + rh) / 2.0

        m = cv2.moments(c)
        if m['m00'] > 0:
            cx, cy = m['m10'] / m['m00'], m['m01'] / m['m00']
        else:
            cx, cy = x + bw / 2.0, y + bh / 2.0

        reasons = []
        if area > p.max_area_px:
            reasons.append(f'面積過大({area:.0f}>{p.max_area_px:.0f}，疑似物件相連)')
        if p.reject_border_touch and touches:
            reasons.append('碰到邊界(被截斷)')
        if n_vert != 4:
            reasons.append(f'頂點數={n_vert}')
        if aspect > p.max_aspect:
            reasons.append(f'長寬比={aspect:.2f}>{p.max_aspect:.2f}')
        if fill < p.min_fill:
            reasons.append(f'填滿率={fill:.2f}<{p.min_fill:.2f}')

        blobs.append(Blob(cx=cx, cy=cy, side=float(side), aspect=float(aspect), fill=float(fill),
                          n_vertices=n_vert, area=area, box=cv2.boxPoints(rect),
                          angle=float(angle), touches_border=bool(touches),
                          too_large=bool(area > p.max_area_px), is_square=not reasons, reject_reason='；'.join(reasons), contour=c))
    blobs.sort(key=lambda t: (round(t.cy / 8.0), t.cx))
    return blobs


def detect_squares(mask: np.ndarray, p: SquareParams) -> Tuple[List[Blob], List[Blob]]:
    """回傳 (合格正方形, 被排除的區塊)。"""
    blobs = analyze_blobs(mask, p)
    return [t for t in blobs if t.is_square], [t for t in blobs if not t.is_square]


def describe_rejected(rejected: List[Blob], unit_scale: float = 1.0, unit: str = 'px',
                      max_items: int = 3) -> str:
    """把被排除的區塊整理成人看得懂的一段文字（供 service message 使用）。"""
    if not rejected:
        return ''
    parts = []
    for i, t in enumerate(rejected[:max_items]):
        parts.append(f'#{i + 1}@({t.cx * unit_scale:.0f},{t.cy * unit_scale:.0f}){unit}: {t.reject_reason}')
    more = f' …另有 {len(rejected) - max_items} 個' if len(rejected) > max_items else ''
    return '；'.join(parts) + more
