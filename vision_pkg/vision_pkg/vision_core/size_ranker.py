"""物件尺寸排序（純函式，不含 ROS）。

節點只依賴 SizeRanker 介面：
    ranker = make_ranker(kind, ...)           # kind 由 config 決定
    result = ranker.rank(sides_mm)            # 8 個邊長 -> rank 1..8 與信心
    check  = ranker.check_order(slot_sides)   # C 區各格邊長是否隨編號遞增

目前實作 RelativeRanker（相對排序）。日後改成「絕對尺寸對照表」時，只要把 config 的
ranker.type 改成 'absolute_table' 並填入 table_mm，節點程式碼完全不用動
（AbsoluteTableRanker 已提供，待實測對照表後即可啟用）。
所有長度單位由呼叫端統一（本專案節點一律傳 mm）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_left
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class RankResult:
    ranks: List[int]            # 與輸入同順序；1 = 最小
    low_confidence: bool        # 相鄰排序（或與對照表）差距小於門檻
    min_gap: float              # 排序後相鄰邊長的最小差距（對照表模式為最大偏差）
    message: str = ''


@dataclass
class OrderCheck:
    order_ok: bool
    wrong_slots: List[int]      # 排錯的格位編號（沿用呼叫端給的編號）
    low_confidence: bool
    message: str = ''


class SizeRanker(ABC):
    """尺寸排序介面。"""

    @abstractmethod
    def rank(self, sides: Sequence[float]) -> RankResult:
        """把一批邊長轉成 rank（1..n，越小越前面）。"""

    @abstractmethod
    def sort_keys(self, sides: Sequence[float]) -> List[float]:
        """回傳可直接比較大小的鍵值（相對排序用邊長本身；對照表用類別編號）。"""

    @abstractmethod
    def ambiguous(self, a: float, b: float) -> bool:
        """兩個邊長的差距是否小到無法可靠比較。"""

    def check_order(self, slot_sides: Sequence[Optional[float]], slot_ids: Optional[Sequence[int]] = None) -> OrderCheck:
        """檢查「已占用格位」的尺寸是否隨格位編號嚴格遞增。

        slot_sides[i] 為第 i 格的邊長，None 代表該格空著（略過）。
        排錯者 = 不在「最長嚴格遞增子序列」內的格位。
        """
        ids = list(slot_ids) if slot_ids is not None else list(range(1, len(slot_sides) + 1))
        occ = [(ids[i], float(s)) for i, s in enumerate(slot_sides) if s is not None]
        if len(occ) <= 1:
            return OrderCheck(True, [], False, '已占用格位不足 2 個，無需比較')
        keys = self.sort_keys([s for _, s in occ])
        keep = _longest_increasing(keys)
        wrong = [occ[i][0] for i in range(len(occ)) if i not in keep]
        low_conf = any(self.ambiguous(occ[i][1], occ[i + 1][1]) for i in range(len(occ) - 1))
        msg = ''
        if wrong:
            msg = '格位 ' + '、'.join(str(w) for w in wrong) + ' 的尺寸與編號順序不符'
        if low_conf:
            msg = (msg + '；' if msg else '') + '部分相鄰格位邊長差距小於信心門檻（低信心）'
        return OrderCheck(not wrong, wrong, low_conf, msg)


def _longest_increasing(keys: Sequence[float]) -> set:
    """最長嚴格遞增子序列，回傳被保留的索引集合（patience sorting，O(n log n)）。"""
    tails: List[float] = []
    tails_idx: List[int] = []
    prev = [-1] * len(keys)
    for i, k in enumerate(keys):
        pos = bisect_left(tails, k)
        if pos == len(tails):
            tails.append(k)
            tails_idx.append(i)
        else:
            tails[pos] = k
            tails_idx[pos] = i
        prev[i] = tails_idx[pos - 1] if pos > 0 else -1
    keep = set()
    i = tails_idx[-1] if tails_idx else -1
    while i != -1:
        keep.add(i)
        i = prev[i]
    return keep


class RelativeRanker(SizeRanker):
    """相對排序：依邊長由小到大給 1..n；相鄰邊長差 < margin 時標記低信心。"""

    def __init__(self, confidence_margin: float):
        self.margin = float(confidence_margin)

    def rank(self, sides: Sequence[float]) -> RankResult:
        n = len(sides)
        if n == 0:
            return RankResult([], False, float('inf'), '沒有物件')
        order = sorted(range(n), key=lambda i: (sides[i], i))
        ranks = [0] * n
        for r, i in enumerate(order, start=1):
            ranks[i] = r
        gaps = [sides[order[j + 1]] - sides[order[j]] for j in range(n - 1)]
        min_gap = min(gaps) if gaps else float('inf')
        low = bool(gaps) and min_gap < self.margin
        msg = f'最小相鄰邊長差 {min_gap:.2f}（門檻 {self.margin:.2f}）' if gaps else ''
        return RankResult(ranks, low, float(min_gap), msg)

    def sort_keys(self, sides: Sequence[float]) -> List[float]:
        return [float(s) for s in sides]

    def ambiguous(self, a: float, b: float) -> bool:
        return abs(a - b) < self.margin


class AbsoluteTableRanker(SizeRanker):
    """絕對尺寸對照表：每個邊長歸類到最接近的標稱尺寸（table 由小到大，index+1 = rank）。

    與標稱值偏差 > tolerance，或兩個物件歸到同一類別時標記低信心。
    """

    def __init__(self, table: Sequence[float], tolerance: float):
        self.table = sorted(float(t) for t in table)
        if not self.table:
            raise ValueError('absolute_table 需要非空的 table_mm')
        self.tol = float(tolerance)

    def _classify(self, side: float) -> int:
        return min(range(len(self.table)), key=lambda i: abs(self.table[i] - side)) + 1

    def rank(self, sides: Sequence[float]) -> RankResult:
        ranks = [self._classify(s) for s in sides]
        worst = max((abs(self.table[r - 1] - s) for r, s in zip(ranks, sides)), default=0.0)
        dup = len(set(ranks)) != len(ranks)
        low = worst > self.tol or dup
        msg = f'與標稱尺寸最大偏差 {worst:.2f}（容許 {self.tol:.2f}）' + ('；有物件歸到同一尺寸類別' if dup else '')
        return RankResult(ranks, low, float(worst), msg)

    def sort_keys(self, sides: Sequence[float]) -> List[float]:
        return [float(self._classify(s)) for s in sides]

    def ambiguous(self, a: float, b: float) -> bool:
        return self._classify(a) == self._classify(b)


def make_ranker(kind: str, confidence_margin: float, table: Optional[Sequence[float]] = None,
                table_tolerance: float = 0.0) -> SizeRanker:
    """依 config 的 ranker.type 建立排序器：'relative' 或 'absolute_table'。"""
    kind = str(kind).strip().lower()
    if kind == 'relative':
        return RelativeRanker(confidence_margin)
    if kind == 'absolute_table':
        return AbsoluteTableRanker(table or [], table_tolerance)
    raise ValueError(f"未知的 ranker.type '{kind}'（可用：relative / absolute_table）")
