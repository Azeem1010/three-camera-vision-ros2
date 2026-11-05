"""HSV 分割、最大輪廓、四邊形角點、透視校正、場地定位與各區占用判斷（純函式，不含 ROS）。"""
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
    """場地外框偵測參數。"""
    min_area_px: float          # 場地色塊最小面積（像素²）
    min_quad_fill: float        # 輪廓面積 / 四邊形面積 的下限，太低代表形狀不像矩形


@dataclass(frozen=True)
class FieldParams:
    """場地定位參數：找出整塊場地（場地色的最大色塊），再沿長邊均分成各區。"""
    color: str                  # 場地顏色（hsv.yaml 的名稱，預設 pink）
    min_aspect: float           # 場地 寬/高 下限；場地在影像中必須是橫向
    reverse: bool               # false = 影像中由左至右為 C|B|A（目前場地）；true = A|B|C


@dataclass(frozen=True)
class OccupancyParams:
    """站台占用 / 手臂判斷參數。"""
    occ_ratio_thresh: float     # 場地顏色被遮擋比例 >= 此值 -> 判定有東西（主要依據）
    object_ratio_thresh: float  # 區內物件色（瓶蓋藍）比例 >= 此值 -> 判定有物件（一顆瓶蓋就要能觸發）
    black_ratio_thresh: float   # 區內黑色比例 >= 此值 -> 視為大面積黑色（疑似手臂）
    visible_drop_ratio: float   # B 區場地色可見面積 / 基準 <= 此值 -> 視為驟降


@dataclass
class ZoneResult:
    found: bool
    corners: Optional[np.ndarray]   # (4,2) float32，[tl,tr,br,bl]
    area_px: float
    method: str                     # 'maxquad' | ''
    reason: str                     # 失敗原因（found=True 時為空）


@dataclass
class ZoneMetrics:
    visible: float   # 場地顏色可見比例（ROI 內）
    obj: float       # 物件色比例
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
def compute_zone_metrics(color_mask: np.ndarray, obj_mask: np.ndarray, black_mask: np.ndarray,
                         roi_mask: np.ndarray, roi_area: int) -> ZoneMetrics:
    """在 ROI（多邊形遮罩）內統計 場地色 / 物件色 / 黑色 的面積比例。"""
    area = float(max(1, roi_area))

    def ratio(m: np.ndarray) -> float:
        return cv2.countNonZero(cv2.bitwise_and(m, roi_mask)) / area

    return ZoneMetrics(ratio(color_mask), ratio(obj_mask), ratio(black_mask))


def occlusion_ratio(visible: float, baseline: float) -> float:
    """場地顏色被遮擋的比例（0 = 完全可見，1 = 全被遮擋）。baseline 為場地淨空時的可見比例。"""
    return float(np.clip(1.0 - visible / max(baseline, 1e-6), 0.0, 1.0))


def judge_object_present(m: ZoneMetrics, baseline: float, p: OccupancyParams) -> Tuple[bool, str]:
    """區內是否有物件。

    以「區內物件色」與「場地顏色被遮擋」判斷；大面積黑色而沒有物件色時視為手臂而非物件。
    瓶蓋很小（直徑 20 mm 只佔 20x30 cm 區域約 0.5%），所以物件色比例是主要依據。
    """
    occ = occlusion_ratio(m.visible, baseline)
    primary = occ >= p.occ_ratio_thresh
    support = m.obj >= p.object_ratio_thresh
    arm_like = m.black >= p.black_ratio_thresh
    present = (primary or support) and not (arm_like and not support)
    return present, f'occ={occ:.2f} obj={m.obj:.4f} black={m.black:.2f}'


def judge_arm_present(m: ZoneMetrics, baseline: float, p: OccupancyParams) -> Tuple[bool, str]:
    """B 區是否有手臂：場地色可見面積相對基準驟降，且出現大面積黑色。"""
    keep = m.visible / max(baseline, 1e-6)
    arm = keep <= p.visible_drop_ratio and m.black >= p.black_ratio_thresh
    return arm, f'visible_keep={keep:.2f} black={m.black:.2f}'


def count_blobs(mask: np.ndarray, min_area_px: float) -> int:
    """遮罩內面積 >= min_area_px 的連通塊數。"""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in cnts if cv2.contourArea(c) >= min_area_px)


# --------------------------------------------------------------------------- #
# 各區 ROI、開機定位、逐區量測（env_vision_node 使用）
# --------------------------------------------------------------------------- #
@dataclass
class ZoneRoi:
    corners: np.ndarray     # (4,2) tl,tr,br,bl
    mask: np.ndarray        # 區域多邊形遮罩（已內縮），uint8 0/255
    area: int               # 遮罩內像素數


def make_roi(shape_hw, corners: np.ndarray, erode_px: int) -> ZoneRoi:
    m = polygon_mask(shape_hw, corners, erode_px)
    return ZoneRoi(np.asarray(corners, np.float32), m, int(cv2.countNonZero(m)))


@dataclass
class LocateResult:
    ok: bool
    corners: Optional[np.ndarray]
    baseline: float          # 淨空時場地色佔 ROI 的比例（之後用來換算「被遮擋比例」）
    message: str


def _with_baseline(frames: Sequence[np.ndarray], corners: np.ndarray, rng: HsvRange, mp: MaskParams,
                   roi_erode_px: int, min_baseline: float, note: str) -> LocateResult:
    """算出淨空時場地色佔 ROI 的基準比例；太低代表 HSV 不對或有東西遮擋。"""
    roi = make_roi(frames[0].shape, corners, roi_erode_px)
    vis = [cv2.countNonZero(cv2.bitwise_and(hsv_mask(f, rng, mp), roi.mask)) / max(1, roi.area)
           for f in frames]
    baseline = float(np.median(vis))
    if baseline < min_baseline:
        return LocateResult(False, corners, baseline,
                            f'淨空基準可見比例 {baseline:.2f} < {min_baseline:.2f}'
                            '（HSV 設定不對，或場地上有東西遮擋，請淨空後重試）')
    return LocateResult(True, corners, baseline, f'ok（{note}，基準 {baseline:.2f}）')


# --------------------------------------------------------------------------- #
# 場地定位：場地色（粉紅）的最大色塊 -> 四角點 -> 沿長邊均分成 C|B|A
# --------------------------------------------------------------------------- #
def max_area_quad(contour: np.ndarray, simplify_ratio: float = 0.005) -> Optional[np.ndarray]:
    """從輪廓凸包的頂點中挑 4 個、使四邊形面積最大，回傳 [tl,tr,br,bl]。

    場地邊上有小凸出物（例如手臂底座的泡棉板、邊緣的小缺口）時，approxPolyDP 會被拉歪；
    「面積最大的內接四邊形」仍會落在真正的四個角上，因為凸出物的頂點換進來會讓面積變小。
    """
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    eps = simplify_ratio
    while True:   # 頂點太多就逐步加大簡化程度（限制組合數；角點是凸包上最「尖」的點，不會被簡化掉）
        pts = cv2.approxPolyDP(hull, max(1.0, eps * peri), True).reshape(-1, 2).astype(np.float64)
        if len(pts) <= 16 or eps > 0.05:
            break
        eps *= 1.5
    n = len(pts)
    if n < 4:
        return None
    best, best_area = None, -1.0
    # 凸包頂點已依序排列，挑 i<j<k<l 即為凸四邊形
    for i in range(n - 3):
        for j in range(i + 1, n - 2):
            for k in range(j + 1, n - 1):
                for l in range(k + 1, n):
                    a = polygon_area(pts[[i, j, k, l]])
                    if a > best_area:
                        best, best_area = (i, j, k, l), a
    return order_corners(pts[list(best)])


def find_field(bgr: np.ndarray, field: HsvRange, mp: MaskParams, zp: ZoneFindParams) -> Tuple[ZoneResult, np.ndarray]:
    """找出整塊場地（場地色最大色塊）的四個角點。回傳 (結果, 場地色遮罩)。

    只看最外層輪廓：場地內的方孔、瓶蓋、兩塊板子的接縫都不影響外框。
    """
    mask = hsv_mask(bgr, field, mp)
    c = largest_contour(mask, zp.min_area_px)
    if c is None:
        return ZoneResult(False, None, 0.0, '', f'找不到面積 >= {zp.min_area_px:.0f} px² 的場地色塊（HSV pink？）'), mask
    quad = max_area_quad(c)
    if quad is None:
        return ZoneResult(False, None, 0.0, '', '場地輪廓無法取得 4 個角點'), mask
    area_c = float(cv2.contourArea(c))
    area_q = polygon_area(quad)
    fill = area_c / area_q if area_q > 0 else 0.0
    if fill < zp.min_quad_fill:
        return ZoneResult(False, None, area_c, 'maxquad',
                          f'場地不像矩形（輪廓/四邊形面積比 {fill:.2f} < {zp.min_quad_fill:.2f}；'
                          '場地被大面積遮住，或有其他粉紅色物體連在場地上）'), mask
    return ZoneResult(True, quad, area_c, 'maxquad', ''), mask


def split_field(corners: np.ndarray, field_size_mm: Sequence[float], names: Sequence[str]) -> Dict[str, np.ndarray]:
    """把場地四角點（tl,tr,br,bl）沿寬度方向均分成 len(names) 區（names 為影像中由左至右的順序），
    回傳各區影像角點（tl,tr,br,bl）。

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


def locate_field_zones(frames: Sequence[np.ndarray], field: HsvRange, names_left_to_right: Sequence[str],
                       field_mp: MaskParams, zone_mp: MaskParams, zp: ZoneFindParams, fp: FieldParams,
                       field_size_mm: Sequence[float], min_detections: int, roi_erode_px: int,
                       min_baseline: float) -> Dict[str, LocateResult]:
    """用多張影格（場地淨空）找場地，均分成各區，並算出各區基準可見比例。

    names_left_to_right：fp.reverse=False 時影像中由左至右的區名（目前場地為 c, b, a）。
    場地找不到時所有區一起失敗；基準比例則逐區判斷。
    """
    names = list(names_left_to_right)
    found, reasons = [], []
    for f in frames:
        zr, _ = find_field(f, field, field_mp, zp)
        if zr.found:
            found.append(zr.corners)
        else:
            reasons.append(zr.reason)
    corners = fuse_corners(found, min_detections)
    if corners is None:
        why = reasons[0] if reasons else '無影格'
        msg = f'場地 {len(found)}/{len(frames)} 張偵測到（至少需 {min_detections}）：{why}'
        return {n: LocateResult(False, None, 0.0, msg) for n in names}
    tl, tr, br, bl = corners
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    if w < fp.min_aspect * h:
        msg = (f'場地在影像中不是橫向（寬/高 {w / max(h, 1e-6):.2f} < {fp.min_aspect:.2f}）；'
               '請轉正環境相機，讓三區在影像中左右排列')
        return {n: LocateResult(False, corners, 0.0, msg) for n in names}
    zones = split_field(corners, field_size_mm, names[::-1] if fp.reverse else names)
    note = f'場地 {len(found)}/{len(frames)} 張'
    return {n: _with_baseline(frames, zones[n], field, zone_mp, roi_erode_px, min_baseline, note)
            for n in names}


def measure_zones(bgr: np.ndarray, rois: Dict[str, ZoneRoi], field: HsvRange, obj: HsvRange, black: HsvRange,
                  zone_mp: MaskParams, obj_mp: MaskParams,
                  black_mp: MaskParams) -> Tuple[Dict[str, ZoneMetrics], np.ndarray]:
    """一張影像內，各區 ROI 的 場地色可見 / 物件色 / 黑色 比例。另回傳整張影像的物件色遮罩（供計數用）。"""
    hsv_z = to_hsv(bgr, zone_mp.blur_ksize)
    hsv_o = hsv_z if obj_mp.blur_ksize == zone_mp.blur_ksize else to_hsv(bgr, obj_mp.blur_ksize)
    hsv_k = hsv_z if black_mp.blur_ksize == zone_mp.blur_ksize else to_hsv(bgr, black_mp.blur_ksize)
    field_mask = mask_from_hsv(hsv_z, field, zone_mp)
    obj_mask = mask_from_hsv(hsv_o, obj, obj_mp)
    black_mask = mask_from_hsv(hsv_k, black, black_mp)
    metrics = {name: compute_zone_metrics(field_mask, obj_mask, black_mask, roi.mask, roi.area)
               for name, roi in rois.items()}
    return metrics, obj_mask
