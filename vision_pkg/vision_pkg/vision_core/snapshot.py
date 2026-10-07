"""A 區快照（偵測 + 排序）與座標配對（純函式，不含 ROS）。

capture_snapshot：A 站四角點透視校正 -> 橘色遮罩 -> 正方形偵測 -> 量邊長 -> 相對排序，
                  以 A 站平面座標（mm，原點左上、x 向右、y 向下）保存各物件中心。
match_object    ：輸入 A 站平面座標，以最近距離配對快照物件（略過已取走者）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Set

import numpy as np

from .color_zone import HsvRange, MaskParams, hsv_mask, warp_to_plane
from .geometry import apply_homography
from .size_ranker import SizeRanker
from .square_detect import SquareParams, analyze_blobs, describe_rejected


@dataclass
class ObjectInfo:
    rank: int
    x_mm: float                 # A 站平面座標
    y_mm: float
    side_mm: float
    px: float                   # 在原始（env 相機）影像中的像素位置，僅供 debug 繪圖
    py: float


@dataclass
class SnapshotResult:
    success: bool
    message: str
    count: int = 0
    low_confidence: bool = False
    objects: List[ObjectInfo] = field(default_factory=list)


@dataclass
class MatchResult:
    success: bool
    message: str
    index: int = -1             # 在 objects 清單中的索引
    rank: int = 0
    distance_mm: float = 0.0


def capture_snapshot(bgr: np.ndarray, corners: np.ndarray, size_mm, px_per_mm: float,
                     orange: HsvRange, orange_mp: MaskParams, sp: SquareParams,
                     ranker: SizeRanker, expected_count: int) -> SnapshotResult:
    """拍快照。偵測到的正方形數量 != expected_count 時回傳失敗（不猜測）。"""
    warped, M = warp_to_plane(bgr, corners, size_mm[0], size_mm[1], px_per_mm)
    mask = hsv_mask(warped, orange, orange_mp)
    blobs = analyze_blobs(mask, sp)
    squares = [b for b in blobs if b.is_square]
    rejected = [b for b in blobs if not b.is_square]

    n = len(squares)
    if n != expected_count:
        detail = describe_rejected(rejected, 1.0 / px_per_mm, 'mm')
        msg = f'偵測到 {n} 個正方形，預期 {expected_count} 個'
        if rejected:
            msg += f'；另有 {len(rejected)} 個橘色區塊被排除（{detail}）'
        elif n < expected_count:
            msg += '；沒有被排除的區塊，請確認物件都在 A 區內且 HSV(orange) 正確'
        return SnapshotResult(False, msg, count=n)

    sides_mm = [b.side / px_per_mm for b in squares]
    rr = ranker.rank(sides_mm)
    inv = np.linalg.inv(M)
    raw_pts = apply_homography(inv, [(b.cx, b.cy) for b in squares])
    objs = [ObjectInfo(rank=rr.ranks[i], x_mm=b.cx / px_per_mm, y_mm=b.cy / px_per_mm,
                       side_mm=sides_mm[i], px=float(raw_pts[i][0]), py=float(raw_pts[i][1]))
            for i, b in enumerate(squares)]
    objs.sort(key=lambda o: o.rank)
    msg = '快照完成；' + rr.message
    if rr.low_confidence:
        msg += '；低信心：相鄰物件尺寸太接近'
    return SnapshotResult(True, msg, count=n, low_confidence=rr.low_confidence, objects=objs)


def match_object(objects: List[ObjectInfo], taken: Iterable[int], x_mm: float, y_mm: float,
                 threshold_mm: float) -> MatchResult:
    """以最近距離配對「尚未取走」的物件（taken 為已取走物件的索引集合，不參與配對，避免重複配對）。

    找不到門檻內的未取走物件時，若該位置附近其實有已取走的物件，會在訊息中點出，方便除錯。
    """
    if not objects:
        return MatchResult(False, '尚無快照物件（請先呼叫 capture_ranking）')
    taken_set: Set[int] = set(taken)
    d = [float(np.hypot(o.x_mm - x_mm, o.y_mm - y_mm)) for o in objects]
    free = [i for i in range(len(objects)) if i not in taken_set]
    if not free:
        return MatchResult(False, '快照中的物件皆已被標記為取走', distance_mm=min(d))
    nearest = min(free, key=lambda i: d[i])
    if d[nearest] > threshold_mm:
        near_taken = [i for i in taken_set if d[i] <= threshold_mm]
        if near_taken:
            t = min(near_taken, key=lambda i: d[i])
            return MatchResult(False, f'該位置的物件（rank {objects[t].rank}）已被標記為取走，距離 {d[t]:.1f} mm；'
                                      '若是重試同一物件，請先呼叫 clear_taken 或改用 peek_only',
                               distance_mm=d[nearest])
        return MatchResult(False, f'配對失敗：最近的未取走物件距離 {d[nearest]:.1f} mm > 門檻 {threshold_mm:.1f} mm',
                           distance_mm=d[nearest])
    o = objects[nearest]
    return MatchResult(True, f'配對 rank {o.rank}，距離 {d[nearest]:.1f} mm',
                       index=nearest, rank=o.rank, distance_mm=d[nearest])
