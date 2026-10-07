"""C 區透視校正後切成 rows x cols 格，計算每格橘色比例與各格物件邊長（純函式，不含 ROS）。

格位編號（1 起算，列優先）：上排（靠近 B 區那一排）1..cols，下排接續；每排由左至右。
2x4 時：上排 1–4、下排 5–8。
「哪一排靠近 B 區」由校正時的角點旋轉（warp_rotation_k）決定，不在本模組內判斷。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .color_zone import HsvRange, MaskParams, hsv_mask
from .square_detect import Blob, SquareParams, analyze_blobs

Rect = Tuple[int, int, int, int]  # x0, y0, x1, y1（x1,y1 為開區間）


@dataclass(frozen=True)
class GridParams:
    rows: int
    cols: int
    margin_ratio: float         # 計算占用比例時，每格四邊各內縮的比例（0~0.4），避開格線/邊緣雜訊
    occupied_threshold: float   # 橘色像素比例 >= 此值 -> 該格已占用


@dataclass
class GridAnalysis:
    rects: List[Rect]
    ratios: List[float]
    occupied: List[bool]
    sides_px: List[Optional[float]]      # 已占用且可量測者為邊長（校正後像素），其餘 None
    approx_slots: List[int]              # 已量測但形狀不是標準正方形的格位（量測值僅供參考）
    unmeasurable_slots: List[int]        # 已占用但無法量測（物件相連 / 被截斷 / 找不到區塊）
    blobs: List[Blob] = field(default_factory=list)
    mask: Optional[np.ndarray] = None


def cell_rects(width_px: int, height_px: int, rows: int, cols: int) -> List[Rect]:
    """列優先的格位矩形清單，索引 i 對應格位 i+1。"""
    rects: List[Rect] = []
    for r in range(rows):
        for c in range(cols):
            rects.append((int(round(c * width_px / cols)), int(round(r * height_px / rows)),
                          int(round((c + 1) * width_px / cols)), int(round((r + 1) * height_px / rows))))
    return rects


def cell_centers(rects: List[Rect]) -> List[Tuple[float, float]]:
    """各格中心（校正後影像像素，浮點）。"""
    return [((x0 + x1) / 2.0, (y0 + y1) / 2.0) for (x0, y0, x1, y1) in rects]


def orange_ratios(mask: np.ndarray, rects: List[Rect], margin_ratio: float) -> List[float]:
    """每格（內縮 margin_ratio 後）的橘色像素比例。"""
    out: List[float] = []
    for (x0, y0, x1, y1) in rects:
        mx = int(round((x1 - x0) * margin_ratio))
        my = int(round((y1 - y0) * margin_ratio))
        roi = mask[y0 + my:y1 - my, x0 + mx:x1 - mx]
        out.append(float(cv2.countNonZero(roi)) / roi.size if roi.size else 0.0)
    return out


def occupancy(ratios: List[float], threshold: float) -> List[bool]:
    return [r >= threshold for r in ratios]


def assign_blobs(blobs: List[Blob], rects: List[Rect]) -> Dict[int, Blob]:
    """把每個連通塊指派給「中心所在的格位」；同格多塊時取面積最大者。回傳 {格位編號(1起): Blob}。"""
    by_slot: Dict[int, Blob] = {}
    for b in blobs:
        for i, (x0, y0, x1, y1) in enumerate(rects):
            if x0 <= b.cx < x1 and y0 <= b.cy < y1:
                slot = i + 1
                if slot not in by_slot or b.area > by_slot[slot].area:
                    by_slot[slot] = b
                break
    return by_slot


def analyze_grid(warped_bgr: np.ndarray, orange: HsvRange, orange_mp: MaskParams,
                 sp: SquareParams, gp: GridParams) -> GridAnalysis:
    """對已透視校正的 C 區影像做整套格位分析。"""
    h, w = warped_bgr.shape[:2]
    mask = hsv_mask(warped_bgr, orange, orange_mp)
    rects = cell_rects(w, h, gp.rows, gp.cols)
    ratios = orange_ratios(mask, rects, gp.margin_ratio)
    occ = occupancy(ratios, gp.occupied_threshold)

    blobs = analyze_blobs(mask, sp)
    by_slot = assign_blobs(blobs, rects)

    sides: List[Optional[float]] = [None] * len(rects)
    approx: List[int] = []
    unmeasurable: List[int] = []
    for i, is_occ in enumerate(occ):
        if not is_occ:
            continue
        b = by_slot.get(i + 1)
        if b is None or b.too_large or b.touches_border:
            unmeasurable.append(i + 1)
            continue
        sides[i] = b.side
        if not b.is_square:
            approx.append(i + 1)
    return GridAnalysis(rects, ratios, occ, sides, approx, unmeasurable, blobs, mask)
