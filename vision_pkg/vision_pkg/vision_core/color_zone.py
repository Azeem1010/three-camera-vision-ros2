"""HSV 分割、最大輪廓、四邊形角點、透視校正、站台占用判斷（純函式，不含 ROS）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from .geometry import apply_homography, order_corners, polygon_area, polygon_mask


# --------------------------------------------------------------------------- #
# 參數 / 結果資料結構（欄位名稱即 YAML 參數名稱，數值一律由呼叫端傳入）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HsvRange:
    """OpenCV HSV 範圍（H 0-179, S/V 0-255）。H 下限 > 上限表示色相環繞（例如紅色）。"""
    lower: Tuple[int, int, int]
    upper: Tuple[int, int, int]

    @classmethod
    def from_dict(cls, d: Dict) -> 'HsvRange':
        return cls(tuple(int(x) for x in d['lower']), tuple(int(x) for x in d['upper']))

    def to_dict(self) -> Dict:
        return {'lower': [int(x) for x in self.lower], 'upper': [int(x) for x in self.upper]}


@dataclass(frozen=True)
class MaskParams:
    """遮罩前處理：高斯模糊核、開運算核（去雜點）、閉運算核（補洞）。核大小 0 表示略過。"""
    blur_ksize: int
    open_ksize: int
    close_ksize: int


@dataclass(frozen=True)
class ZoneFindParams:
    """站台四邊形偵測參數。"""
    min_area_px: float          # 最小色塊面積（像素²）
    approx_eps_min: float       # approxPolyDP 的 epsilon 掃描範圍（佔周長比例）
    approx_eps_max: float
    approx_eps_steps: int
    min_quad_fill: float        # 輪廓面積 / 四邊形面積 的下限，太低代表形狀不像矩形
    allow_rect_fallback: bool   # approxPolyDP 湊不出 4 點時是否改用 minAreaRect


@dataclass(frozen=True)
class FieldParams:
    """外框定位參數（zone_locate_mode=field：找整個場地的黑色外框，再沿長邊均分成各區）。"""
    edge: str                   # 'outer' = 黑框外緣即場地邊界；'inner' = 黑框內緣（框內白色區域）即場地邊界
    min_aspect: float           # 外框 寬/高 下限；場地在影像中必須是橫向（A|B|C 由左至右）
    reverse: bool               # true = 影像中由左至右為 C|B|A


@dataclass(frozen=True)
class OccupancyParams:
    """站台占用 / 手臂判斷參數。"""
    occ_ratio_thresh: float     # 站台顏色被遮擋比例 >= 此值 -> 判定有東西（主要依據）
    orange_ratio_thresh: float  # 區內橘色比例 >= 此值 -> 判定有物件（輔助）
    black_ratio_thresh: float   # 區內黑色比例 >= 此值 -> 視為大面積黑色（疑似手臂）
    green_drop_ratio: float     # B 區綠色可見面積 / 基準 <= 此值 -> 視為驟降


@dataclass
class ZoneResult:
    found: bool
    corners: Optional[np.ndarray]   # (4,2) float32，[tl,tr,br,bl]
    area_px: float
    method: str                     # 'approx' | 'minrect' | ''
    reason: str                     # 失敗原因（found=True 時為空）


@dataclass
class ZoneMetrics:
    visible: float   # 站台顏色可見比例（ROI 內）
    orange: float    # 橘色比例
    black: float     # 黑色比例


# --------------------------------------------------------------------------- #
# HSV 分割
# --------------------------------------------------------------------------- #
def _odd(k: int) -> int:
    k = int(k)
    return k if k % 2 == 1 else k + 1


def to_hsv(bgr: np.ndarray, blur_ksize: int) -> np.ndarray:
    """可選的高斯模糊後轉 HSV。"""
    img = bgr
    if int(blur_ksize) >= 3:
        k = _odd(blur_ksize)
        img = cv2.GaussianBlur(bgr, (k, k), 0)
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def in_range_hsv(hsv: np.ndarray, rng: HsvRange) -> np.ndarray:
    """HSV 範圍二值化，支援色相環繞（H 下限 > 上限）。"""
    lo, hi = rng.lower, rng.upper
    if lo[0] <= hi[0]:
        return cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    m1 = cv2.inRange(hsv, np.array((lo[0], lo[1], lo[2]), np.uint8),
                     np.array((179, hi[1], hi[2]), np.uint8))
    m2 = cv2.inRange(hsv, np.array((0, lo[1], lo[2]), np.uint8),
                     np.array((hi[0], hi[1], hi[2]), np.uint8))
    return cv2.bitwise_or(m1, m2)


def morph_clean(mask: np.ndarray, open_ksize: int, close_ksize: int) -> np.ndarray:
    """先開後閉：去除雜點、填補高光造成的小洞。"""
    out = mask
    if int(open_ksize) >= 2:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(open_ksize),) * 2)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if int(close_ksize) >= 2:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(close_ksize),) * 2)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    return out


def mask_from_hsv(hsv: np.ndarray, rng: HsvRange, mp: MaskParams) -> np.ndarray:
    return morph_clean(in_range_hsv(hsv, rng), mp.open_ksize, mp.close_ksize)


def hsv_mask(bgr: np.ndarray, rng: HsvRange, mp: MaskParams) -> np.ndarray:
    """BGR -> 單一顏色的二值遮罩（255 = 符合）。"""
    return mask_from_hsv(to_hsv(bgr, mp.blur_ksize), rng, mp)


# --------------------------------------------------------------------------- #
# 輪廓 / 四邊形
# --------------------------------------------------------------------------- #
def largest_contour(mask: np.ndarray, min_area: float) -> Optional[np.ndarray]:
    """回傳面積最大且 >= min_area 的外輪廓，沒有則 None。"""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, float(min_area)
    for c in cnts:
        a = cv2.contourArea(c)
        if a >= best_area:
            best, best_area = c, a
    return best


def quad_from_contour(contour: np.ndarray, zp: ZoneFindParams) -> Tuple[Optional[np.ndarray], str]:
    """以 approxPolyDP 取四邊形角點。對凸包掃描 epsilon，直到剛好 4 點且為凸；

    湊不出時（allow_rect_fallback）退回 minAreaRect。回傳 (角點[tl,tr,br,bl] 或 None, 方法名稱)。
    """
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    steps = max(1, int(zp.approx_eps_steps))
    for eps in np.linspace(zp.approx_eps_min, zp.approx_eps_max, steps):
        approx = cv2.approxPolyDP(hull, float(eps) * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_corners(approx.reshape(4, 2)), 'approx'
    if zp.allow_rect_fallback:
        box = cv2.boxPoints(cv2.minAreaRect(hull))
        return order_corners(box), 'minrect'
    return None, ''


def find_zone(bgr: np.ndarray, rng: HsvRange, mp: MaskParams, zp: ZoneFindParams,
              extra_mask: Optional[np.ndarray] = None) -> Tuple[ZoneResult, np.ndarray]:
    """找出單一顏色站台的四個角點。回傳 (結果, 顏色遮罩)；遮罩供 debug 使用。

    extra_mask：與站台色「聯集」後再找輪廓的遮罩。站台上已放了物件時（例如 C 區放滿橘色物件），
    貼近站台邊緣的物件會把站台色輪廓咬掉一角；把物件遮罩併進來即可還原完整外框。
    """
    mask = hsv_mask(bgr, rng, mp)
    if extra_mask is not None:
        mask = cv2.bitwise_or(mask, extra_mask)
    c = largest_contour(mask, zp.min_area_px)
    if c is None:
        return ZoneResult(False, None, 0.0, '', f'找不到面積 >= {zp.min_area_px:.0f} px² 的色塊'), mask
    quad, method = quad_from_contour(c, zp)
    if quad is None:
        return ZoneResult(False, None, 0.0, '', 'approxPolyDP 無法取得 4 個角點'), mask
    area_c = float(cv2.contourArea(c))
    area_q = polygon_area(quad)
    fill = area_c / area_q if area_q > 0 else 0.0
    if fill < zp.min_quad_fill:
        return ZoneResult(False, None, area_c, method,
                          f'色塊不像矩形（輪廓/四邊形面積比 {fill:.2f} < {zp.min_quad_fill:.2f}）'), mask
    return ZoneResult(True, quad, area_c, method, ''), mask


def fuse_corners(corner_list: Sequence[np.ndarray], min_count: int) -> Optional[np.ndarray]:
    """多張影格的角點取逐點中位數（抗雜訊）。有效影格數不足 min_count 回傳 None。"""
    if len(corner_list) < max(1, int(min_count)):
        return None
    return np.median(np.stack([np.asarray(c, np.float32) for c in corner_list]), axis=0).astype(np.float32)


# --------------------------------------------------------------------------- #
# 透視校正
# --------------------------------------------------------------------------- #
def warp_to_plane(bgr: np.ndarray, corners: np.ndarray, width_mm: float, height_mm: float,
                  px_per_mm: float) -> Tuple[np.ndarray, np.ndarray]:
    """把站台四角點（tl,tr,br,bl）校正成正視圖。

    輸出尺寸 = (width_mm*px_per_mm, height_mm*px_per_mm)；回傳 (校正圖, 3x3 矩陣 M)，
    M 把「原影像像素」映射到「校正圖像素」。校正圖像素 / px_per_mm = 站台平面 mm。
    """
    w = max(1, int(round(width_mm * px_per_mm)))
    h = max(1, int(round(height_mm * px_per_mm)))
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(np.asarray(corners, dtype=np.float32), dst)
    return cv2.warpPerspective(bgr, M, (w, h), flags=cv2.INTER_LINEAR), M


# --------------------------------------------------------------------------- #
# 占用判斷
# --------------------------------------------------------------------------- #
def compute_zone_metrics(color_mask: np.ndarray, orange_mask: np.ndarray, black_mask: np.ndarray,
                         roi_mask: np.ndarray, roi_area: int) -> ZoneMetrics:
    """在 ROI（多邊形遮罩）內統計 站台色 / 橘色 / 黑色 的面積比例。"""
    area = float(max(1, roi_area))

    def ratio(m: np.ndarray) -> float:
        return cv2.countNonZero(cv2.bitwise_and(m, roi_mask)) / area

    return ZoneMetrics(ratio(color_mask), ratio(orange_mask), ratio(black_mask))


def occlusion_ratio(visible: float, baseline: float) -> float:
    """站台顏色被遮擋的比例（0 = 完全可見，1 = 全被遮擋）。baseline 為桌面淨空時的可見比例。"""
    return float(np.clip(1.0 - visible / max(baseline, 1e-6), 0.0, 1.0))


def judge_object_present(m: ZoneMetrics, baseline: float, p: OccupancyParams) -> Tuple[bool, str]:
    """區內是否有物件。

    以「站台顏色被遮擋」為主、「區內橘色」為輔；大面積黑色而沒有橘色時視為手臂而非物件。
    """
    occ = occlusion_ratio(m.visible, baseline)
    primary = occ >= p.occ_ratio_thresh
    support = m.orange >= p.orange_ratio_thresh
    arm_like = m.black >= p.black_ratio_thresh
    present = (primary or support) and not (arm_like and not support)
    return present, f'occ={occ:.2f} orange={m.orange:.3f} black={m.black:.2f}'


def judge_arm_present(m: ZoneMetrics, baseline: float, p: OccupancyParams) -> Tuple[bool, str]:
    """B 區是否有手臂：綠色可見面積相對基準驟降，且出現大面積黑色。"""
    keep = m.visible / max(baseline, 1e-6)
    arm = keep <= p.green_drop_ratio and m.black >= p.black_ratio_thresh
    return arm, f'green_keep={keep:.2f} black={m.black:.2f}'


def count_blobs(mask: np.ndarray, min_area_px: float) -> int:
    """遮罩內面積 >= min_area_px 的連通塊數。"""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in cnts if cv2.contourArea(c) >= min_area_px)


# --------------------------------------------------------------------------- #
# 站台 ROI、開機定位、逐區量測（env_vision_node 使用）
# --------------------------------------------------------------------------- #
@dataclass
class ZoneRoi:
    corners: np.ndarray     # (4,2) tl,tr,br,bl
    mask: np.ndarray        # 站台多邊形遮罩（已內縮），uint8 0/255
    area: int               # 遮罩內像素數


def make_roi(shape_hw, corners: np.ndarray, erode_px: int) -> ZoneRoi:
    m = polygon_mask(shape_hw, corners, erode_px)
    return ZoneRoi(np.asarray(corners, np.float32), m, int(cv2.countNonZero(m)))


@dataclass
class LocateResult:
    ok: bool
    corners: Optional[np.ndarray]
    baseline: float          # 淨空時站台顏色佔 ROI 的比例（之後用來換算「被遮擋比例」）
    message: str


def locate_zones(frames: Sequence[np.ndarray], zone_ranges: Dict[str, HsvRange], mp: MaskParams,
                 zp: ZoneFindParams, min_detections: int, roi_erode_px: int,
                 min_baseline: float) -> Dict[str, LocateResult]:
    """用多張影格（桌面淨空）逐區定位站台：角點取逐點中位數，並算出基準可見比例。

    每個站台獨立判斷成功與否，失敗的站台 ok=False 並附原因（呼叫端可改用上次存檔）。
    """
    out: Dict[str, LocateResult] = {}
    for name, rng in zone_ranges.items():
        found, reasons = [], []
        for f in frames:
            zr, _ = find_zone(f, rng, mp, zp)
            if zr.found:
                found.append(zr.corners)
            else:
                reasons.append(zr.reason)
        corners = fuse_corners(found, min_detections)
        if corners is None:
            why = reasons[0] if reasons else '無影格'
            out[name] = LocateResult(False, None, 0.0,
                                     f'{len(found)}/{len(frames)} 張偵測到（至少需 {min_detections}）：{why}')
            continue
        out[name] = _with_baseline(frames, corners, rng, mp, roi_erode_px, min_baseline,
                                   f'{len(found)}/{len(frames)} 張')
    return out


def _with_baseline(frames: Sequence[np.ndarray], corners: np.ndarray, rng: HsvRange, mp: MaskParams,
                   roi_erode_px: int, min_baseline: float, note: str) -> LocateResult:
    """算出淨空時站台顏色佔 ROI 的基準比例；太低代表 HSV 不對或有東西遮擋。"""
    roi = make_roi(frames[0].shape, corners, roi_erode_px)
    vis = [cv2.countNonZero(cv2.bitwise_and(hsv_mask(f, rng, mp), roi.mask)) / max(1, roi.area)
           for f in frames]
    baseline = float(np.median(vis))
    if baseline < min_baseline:
        return LocateResult(False, corners, baseline,
                            f'淨空基準可見比例 {baseline:.2f} < {min_baseline:.2f}'
                            '（HSV 設定不對，或站台上有東西遮擋，請淨空後重試）')
    return LocateResult(True, corners, baseline, f'ok（{note}，基準 {baseline:.2f}）')


# --------------------------------------------------------------------------- #
# 外框定位：整個場地一個黑色外框，沿長邊均分成 A|B|C
# --------------------------------------------------------------------------- #
def find_field(bgr: np.ndarray, black: HsvRange, mp: MaskParams, zp: ZoneFindParams,
               edge: str) -> Tuple[ZoneResult, np.ndarray]:
    """找出場地黑色外框的四個角點。回傳 (結果, 黑色遮罩)。

    edge='outer'：取最大黑色輪廓的外緣（框內有分隔線也不影響）。
    edge='inner'：取該輪廓所有內洞（框內白色區域，有分隔線時為多個）的凸包。
    """
    mask = hsv_mask(bgr, black, mp)
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = -1, float(zp.min_area_px)
    for i, c in enumerate(cnts):
        a = cv2.contourArea(c)
        if hier[0][i][3] == -1 and a >= best_area:
            best, best_area = i, a
    if best < 0:
        return ZoneResult(False, None, 0.0, '', f'找不到面積 >= {zp.min_area_px:.0f} px² 的黑色外框'), mask
    if edge == 'inner':
        holes = [cnts[i] for i in range(len(cnts))
                 if hier[0][i][3] == best and cv2.contourArea(cnts[i]) >= 0.02 * best_area]
        if not holes:
            return ZoneResult(False, None, 0.0, '', '黑色外框沒有封閉的內部區域（框線有斷開？可改用 edge=outer）'), mask
        contour = cv2.convexHull(np.vstack(holes))
        area_c = float(sum(cv2.contourArea(h) for h in holes))
    else:
        contour = cnts[best]
        area_c = float(best_area)
    quad, method = quad_from_contour(contour, zp)
    if quad is None:
        return ZoneResult(False, None, 0.0, '', 'approxPolyDP 無法取得外框的 4 個角點'), mask
    area_q = polygon_area(quad)
    fill = area_c / area_q if area_q > 0 else 0.0
    if fill < zp.min_quad_fill:
        return ZoneResult(False, None, area_c, method,
                          f'外框不像矩形（輪廓/四邊形面積比 {fill:.2f} < {zp.min_quad_fill:.2f}）'), mask
    return ZoneResult(True, quad, area_c, method, ''), mask


def split_field(corners: np.ndarray, field_size_mm: Sequence[float], names: Sequence[str]) -> Dict[str, np.ndarray]:
    """把場地四角點（tl,tr,br,bl）沿寬度方向均分成 len(names) 區，回傳各區影像角點（tl,tr,br,bl）。

    以透視轉換在場地平面上等分，所以斜拍時各區在影像中的寬度不同也沒關係。
    """
    W, H = float(field_size_mm[0]), float(field_size_mm[1])
    to_img = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]),
                                         np.asarray(corners, np.float32))
    n = len(names)
    out: Dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        x0, x1 = W * i / n, W * (i + 1) / n
        out[name] = apply_homography(to_img, [[x0, 0], [x1, 0], [x1, H], [x0, H]]).astype(np.float32)
    return out


def locate_field_zones(frames: Sequence[np.ndarray], black: HsvRange, zone_ranges: Dict[str, HsvRange],
                       field_mp: MaskParams, zone_mp: MaskParams, zp: ZoneFindParams, fp: FieldParams,
                       field_size_mm: Sequence[float], min_detections: int, roi_erode_px: int,
                       min_baseline: float) -> Dict[str, LocateResult]:
    """用多張影格（桌面淨空）找場地外框，均分成各區，並算出各區基準可見比例。

    zone_ranges 的順序即影像中由左至右的順序（fp.reverse=True 時反過來）。
    外框找不到時所有區一起失敗；基準比例則逐區判斷。
    """
    names = list(zone_ranges)
    found, reasons = [], []
    for f in frames:
        zr, _ = find_field(f, black, field_mp, zp, fp.edge)
        if zr.found:
            found.append(zr.corners)
        else:
            reasons.append(zr.reason)
    corners = fuse_corners(found, min_detections)
    if corners is None:
        why = reasons[0] if reasons else '無影格'
        msg = f'外框 {len(found)}/{len(frames)} 張偵測到（至少需 {min_detections}）：{why}'
        return {n: LocateResult(False, None, 0.0, msg) for n in names}
    tl, tr, br, bl = corners
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    if w < fp.min_aspect * h:
        msg = (f'外框在影像中不是橫向（寬/高 {w / max(h, 1e-6):.2f} < {fp.min_aspect:.2f}）；'
               '請轉正環境相機，讓 A|B|C 在影像中由左至右排列')
        return {n: LocateResult(False, corners, 0.0, msg) for n in names}
    zones = split_field(corners, field_size_mm, names[::-1] if fp.reverse else names)
    note = f'外框 {len(found)}/{len(frames)} 張'
    return {n: _with_baseline(frames, zones[n], zone_ranges[n], zone_mp, roi_erode_px, min_baseline, note)
            for n in names}


def measure_zones(bgr: np.ndarray, rois: Dict[str, ZoneRoi], zone_ranges: Dict[str, HsvRange],
                  orange: HsvRange, black: HsvRange, zone_mp: MaskParams, orange_mp: MaskParams,
                  black_mp: MaskParams) -> Tuple[Dict[str, ZoneMetrics], np.ndarray]:
    """一張影像內，各站台 ROI 的 站台色可見 / 橘色 / 黑色 比例。另回傳整張影像的橘色遮罩（供計數用）。"""
    hsv_z = to_hsv(bgr, zone_mp.blur_ksize)
    hsv_o = hsv_z if orange_mp.blur_ksize == zone_mp.blur_ksize else to_hsv(bgr, orange_mp.blur_ksize)
    hsv_k = hsv_z if black_mp.blur_ksize == zone_mp.blur_ksize else to_hsv(bgr, black_mp.blur_ksize)
    orange_mask = mask_from_hsv(hsv_o, orange, orange_mp)
    black_mask = mask_from_hsv(hsv_k, black, black_mp)
    metrics: Dict[str, ZoneMetrics] = {}
    for name, roi in rois.items():
        color_mask = mask_from_hsv(hsv_z, zone_ranges[name], zone_mp)
        metrics[name] = compute_zone_metrics(color_mask, orange_mask, black_mask, roi.mask, roi.area)
    return metrics, orange_mask
