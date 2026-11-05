"""C 區 8 個方孔：偵測、固定編號、每孔是否已放瓶蓋（純函式，不含 ROS）。

場地是粉紅色泡棉板，C 區挖了 8 個方孔（2 行 x 4 列），孔底是白紙/桌面。
做法：粉紅遮罩的「內部洞」= 方孔（孔裡的瓶蓋不是粉紅色，所以不影響孔的輪廓）。

孔位編號（固定，與使用者約定）：以「使用者視角」——人站在場地長邊前、B 區在右手邊——
最下面一排（靠近人）左邊 1、右邊 2，往上一排 3、4，… 最上面一排 7、8：

      [7] [8]
      [5] [6]
      [3] [4]
      [1] [2]
    ── 使用者這側 ──

手眼相機 2 的影像不一定是這個方向：參數 view_rotation_k（0..3）= 影像要「順時針」轉幾個 90° 才會變成
上面的使用者視角。程式只用孔的相對位置排序，不需要看到 C 區外框。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .geometry import order_corners

N_SLOTS = 8
ROWS, COLS = 4, 2      # 使用者視角：4 排（由下往上）x 2 個（由左至右）


@dataclass(frozen=True)
class HoleParams:
    min_area_px: float          # 方孔在影像中的最小面積（像素²）
    max_area_px: float          # 最大面積（大於此值多半是手臂/陰影等大塊非粉紅區域）
    max_aspect: float           # minAreaRect 長寬比上限
    min_fill: float             # 輪廓面積 / minAreaRect 面積 下限（方孔約 0.9 以上）
    margin_ratio: float         # 計算占用比例時，孔四邊各內縮的比例（避開孔壁陰影）
    occupied_threshold: float   # 孔內物件色比例 >= 此值 -> 已放置


@dataclass
class Hole:
    corners: np.ndarray         # (4,2) [tl,tr,br,bl]（影像像素）
    cx: float
    cy: float
    area: float
    aspect: float
    fill: float
    ok: bool
    reason: str = ''


@dataclass
class HoleLayout:
    ok: bool
    message: str
    holes: List[Hole] = field(default_factory=list)       # ok=True 時依格位 1..8 排序
    candidates: List[Hole] = field(default_factory=list)  # 所有候選（含不合格者），debug 用


def find_hole_candidates(field_mask: np.ndarray, hp: HoleParams) -> List[Hole]:
    """場地色遮罩裡的「洞」（被場地色包圍的非場地色區域），附合格與否。"""
    cnts, hier = cv2.findContours(field_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Hole] = []
    if hier is None:
        return out
    for i, c in enumerate(cnts):
        if hier[0][i][3] < 0:          # 最外層輪廓（場地本身），不是洞
            continue
        area = float(cv2.contourArea(c))
        if area < hp.min_area_px * 0.5:   # 小雜點不列入候選
            continue
        (rx, ry), (rw, rh), _ = rect = cv2.minAreaRect(c)
        aspect = max(rw, rh) / max(min(rw, rh), 1e-6)
        fill = area / max(rw * rh, 1e-6)
        reasons = []
        if not hp.min_area_px <= area <= hp.max_area_px:
            reasons.append(f'面積 {area:.0f} 不在 [{hp.min_area_px:.0f}, {hp.max_area_px:.0f}]')
        if aspect > hp.max_aspect:
            reasons.append(f'長寬比 {aspect:.2f} > {hp.max_aspect:.2f}')
        if fill < hp.min_fill:
            reasons.append(f'填滿率 {fill:.2f} < {hp.min_fill:.2f}')
        out.append(Hole(order_corners(cv2.boxPoints(rect)), float(rx), float(ry), area, float(aspect),
                        float(fill), not reasons, '；'.join(reasons)))
    return out


def _to_view(pts: np.ndarray, k: int) -> np.ndarray:
    """影像座標 -> 使用者視角座標（只看方向）：順時針轉 k 個 90°，(x, y) -> (-y, x)。"""
    p = np.asarray(pts, np.float64).reshape(-1, 2).copy()
    for _ in range(int(k) % 4):
        p = np.stack([-p[:, 1], p[:, 0]], axis=1)
    return p


def order_slots(centers: Sequence[Sequence[float]], view_rotation_k: int) -> Tuple[Optional[List[int]], str]:
    """把 8 個孔中心排成格位 1..8。回傳 (索引清單：第 i 項 = 格位 i+1 的孔索引, 訊息)；失敗時 (None, 原因)。"""
    c = np.asarray(centers, np.float64).reshape(-1, 2)
    if len(c) != N_SLOTS:
        return None, f'孔數 {len(c)} != {N_SLOTS}'
    v = _to_view(c, view_rotation_k)
    v = v - v.mean(axis=0)
    w, vecs = np.linalg.eigh(np.cov(v.T))
    u = vecs[:, int(np.argmax(w))]                       # 長軸（4 排的方向）
    if abs(u[1]) < abs(u[0]):
        return None, (f'孔位排列的長軸在使用者視角中是橫的（view_rotation_k={int(view_rotation_k) % 4}），'
                      '請把 view_rotation_k 加 1 或減 1，並看 debug 視窗確認 1 號在左下')
    if u[1] < 0:
        u = -u                                           # u 指向「下」（靠近使用者）
    s = np.array([u[1], -u[0]])                          # 短軸，指向「右」
    pu, ps = v @ u, v @ s
    by_u = list(np.argsort(-pu))                         # 由下往上
    rows = [by_u[2 * r:2 * r + 2] for r in range(ROWS)]
    # 檢查是否像 4x2 格子：同一排的兩孔在長軸上的差，要明顯小於排與排之間的間距
    row_pos = [float(np.mean(pu[r])) for r in rows]
    gaps = [row_pos[r] - row_pos[r + 1] for r in range(ROWS - 1)]
    spread = max(abs(float(pu[r[0]] - pu[r[1]])) for r in rows)
    if min(gaps) <= 0 or spread > 0.5 * min(gaps):
        return None, f'孔位不像 4 排 x 2 個的格子（同排長軸差 {spread:.1f}px，排距最小 {min(gaps):.1f}px）'
    order: List[int] = []
    for r in rows:
        order.extend(sorted(r, key=lambda i: ps[i]))     # 左 -> 右
    return [int(i) for i in order], 'ok'


def locate_holes(field_mask: np.ndarray, hp: HoleParams, view_rotation_k: int) -> HoleLayout:
    """找出 8 個方孔並依固定編號排序。數量不對或排列不像格子時 ok=False（不猜測）。"""
    cands = find_hole_candidates(field_mask, hp)
    good = [h for h in cands if h.ok]
    if len(good) != N_SLOTS:
        bad = [h for h in cands if not h.ok]
        msg = f'找到 {len(good)} 個方孔，預期 {N_SLOTS} 個'
        if bad:
            msg += '；被排除的候選：' + '；'.join(f'@({h.cx:.0f},{h.cy:.0f}) {h.reason}' for h in bad[:3])
        return HoleLayout(False, msg, [], cands)
    order, why = order_slots([(h.cx, h.cy) for h in good], view_rotation_k)
    if order is None:
        return HoleLayout(False, why, [], cands)
    return HoleLayout(True, 'ok', [good[i] for i in order], cands)


def slot_ratios(obj_mask: np.ndarray, hole_corners: Sequence[np.ndarray], margin_ratio: float) -> List[float]:
    """每個孔（向中心內縮 margin_ratio 後）內的物件色比例。"""
    out: List[float] = []
    for corners in hole_corners:
        q = np.asarray(corners, np.float32)
        ctr = q.mean(axis=0)
        q = ctr + (q - ctr) * (1.0 - 2.0 * float(margin_ratio))
        m = np.zeros(obj_mask.shape[:2], np.uint8)
        cv2.fillConvexPoly(m, np.round(q).astype(np.int32), 255)
        area = cv2.countNonZero(m)
        out.append(cv2.countNonZero(cv2.bitwise_and(obj_mask, m)) / area if area else 0.0)
    return out


@dataclass
class FillOrderCheck:
    order_ok: bool
    wrong_slots: List[int]       # 已放置、但前面還有空格的格位（跳號放置）
    next_slot: int               # 下一個應該放的格位（0 = 已放滿）
    message: str


def check_fill_order(occupied: Sequence[bool]) -> FillOrderCheck:
    """瓶蓋大小都一樣，「順序正確」= 依 1, 2, 3, … 的順序放：已放置的格位必須是 1..n 連續。"""
    occ = [bool(o) for o in occupied]
    first_empty = next((i for i, o in enumerate(occ) if not o), None)
    if first_empty is None:
        return FillOrderCheck(True, [], 0, '8 格已放滿')
    wrong = [i + 1 for i in range(first_empty + 1, len(occ)) if occ[i]]
    nxt = first_empty + 1
    if wrong:
        return FillOrderCheck(False, wrong, nxt, f'格位 {wrong} 已放置，但格位 {nxt} 還是空的（跳號放置）')
    return FillOrderCheck(True, [], nxt, f'已依序放置 {first_empty} 格，下一格 {nxt}')

