"""藍色圓形瓶蓋偵測（純函式，不含 ROS）。

判定條件（皆由參數傳入）：
  1. 面積在 [min_area_px, max_area_px]（超過上限視為多顆相連）
  2. 圓度 4πA/P² >= min_circularity（凸包周長，不受邊緣鋸齒影響）
  3. 填滿率（輪廓面積 / 最小外接圓面積）>= min_fill
  4. 不碰到遮罩邊界（被截斷；可關閉）
不符合者保留在 rejected 清單並附上原因，方便上層回報而不是默默猜測。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class CapParams:
    """偵測所在影像的單位：env 為「A 區校正後像素」，arm1 為「原始影像像素」。"""
    min_area_px: float          # 小於此面積視為雜訊，直接忽略
    max_area_px: float          # 大於此面積視為多顆瓶蓋黏在一起 -> 排除
    min_circularity: float      # 圓度下限（正圓 = 1；正方形約 0.785）
    min_fill: float             # 輪廓面積 / 最小外接圓面積 的下限
    reject_border_touch: bool   # 碰到影像邊界（被截斷）的區塊是否排除
    border_px: int              # 遮罩邊緣這幾像素視為無效


@dataclass
class Cap:
    """一個物件色連通塊的量測結果。"""
    cx: float
    cy: float
    diameter: float             # 等面積圓直徑（影像像素）
    circularity: float
    fill: float
    area: float
    radius: float               # 最小外接圓半徑（繪圖用）
    touches_border: bool
    too_large: bool             # 面積超過 max_area_px（疑似多顆相連）
    is_cap: bool
    reject_reason: str          # is_cap=True 時為空字串
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


def analyze_caps(mask: np.ndarray, p: CapParams) -> List[Cap]:
    """量測遮罩內所有面積 >= min_area_px 的連通塊（含不合格者），依 (y, x) 由上到下、左到右排序。"""
    h, w = mask.shape[:2]
    b = max(0, int(p.border_px))
    cnts, _ = cv2.findContours(clear_border(mask, b), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Cap] = []
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < p.min_area_px:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        touches = x <= b or y <= b or (x + bw) >= (w - b) or (y + bh) >= (h - b)
        peri = cv2.arcLength(cv2.convexHull(c), True)
        circ = 4.0 * math.pi * area / (peri * peri) if peri > 1e-6 else 0.0
        (ex, ey), r = cv2.minEnclosingCircle(c)
        fill = area / (math.pi * r * r) if r > 1e-6 else 0.0
        m = cv2.moments(c)
        cx, cy = (m['m10'] / m['m00'], m['m01'] / m['m00']) if m['m00'] > 0 else (ex, ey)

        reasons = []
        if area > p.max_area_px:
            reasons.append(f'面積過大({area:.0f}>{p.max_area_px:.0f}，疑似瓶蓋相連)')
        if p.reject_border_touch and touches:
            reasons.append('碰到邊界(被截斷)')
        if circ < p.min_circularity:
            reasons.append(f'圓度={circ:.2f}<{p.min_circularity:.2f}')
        if fill < p.min_fill:
            reasons.append(f'填滿率={fill:.2f}<{p.min_fill:.2f}')
        out.append(Cap(cx=float(cx), cy=float(cy), diameter=2.0 * math.sqrt(area / math.pi),
                       circularity=float(circ), fill=float(fill), area=area, radius=float(r),
                       touches_border=bool(touches), too_large=bool(area > p.max_area_px),
                       is_cap=not reasons, reject_reason='；'.join(reasons), contour=c))
    out.sort(key=lambda t: (round(t.cy / 8.0), t.cx))
    return out


def detect_caps(mask: np.ndarray, p: CapParams) -> Tuple[List[Cap], List[Cap]]:
    """回傳 (合格瓶蓋, 被排除的區塊)。"""
    caps = analyze_caps(mask, p)
    return [t for t in caps if t.is_cap], [t for t in caps if not t.is_cap]


def describe_rejected(rejected: List[Cap], unit_scale: float = 1.0, unit: str = 'px',
                      max_items: int = 3) -> str:
    """把被排除的區塊整理成人看得懂的一段文字（供 service message 使用）。"""
    if not rejected:
        return ''
    parts = [f'#{i + 1}@({t.cx * unit_scale:.0f},{t.cy * unit_scale:.0f}){unit}: {t.reject_reason}'
             for i, t in enumerate(rejected[:max_items])]
    more = f' …另有 {len(rejected) - max_items} 個' if len(rejected) > max_items else ''
    return '；'.join(parts) + more


def nearest_neighbor_dist(pts: np.ndarray) -> np.ndarray:
    """每個點到其他點的最近距離（只有一個點時為 inf）。pts: (N,2)。"""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    n = len(pts)
    if n < 2:
        return np.full(n, np.inf)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)
