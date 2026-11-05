"""合成靜態場景（含透視、灰色桌面、噪點），供 pytest 與 hsv_tuner 離線練習使用。

直接執行  python tests/make_test_images.py  會把這些場景輸出成 PNG 到 tests/data/。

場地（與實物一致）：60x30 cm 粉紅色泡棉板，由左至右 C | B | A 各 20x30 cm；C 區有 8 個方孔（2 行 x 4 列，
孔底白色），場地下緣在 C 區有一塊凸出的手臂底座泡棉板。物件：藍色圓形瓶蓋，直徑 20 mm。
座標：場地平面 mm，原點為場地左上角（C 區那一側），x 向右、y 向下（y 大 = 靠近使用者）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

# 顏色（BGR）
DESK = (182, 184, 178)       # 灰色桌面（低飽和度）
PINK = (200, 170, 240)       # 粉紅泡棉（HSV 約 H167 S74 V240）
WHITE = (235, 235, 235)      # 孔底白紙
BLUE = (200, 90, 30)         # 藍色瓶蓋（HSV 約 H109 S217 V200）
BLACK = (25, 25, 25)

IMG_SIZE = (640, 480)  # (w, h)

FIELD_SIZE_MM = (600.0, 300.0)
ZONE_W_MM = 200.0
A_X0_MM = 400.0                                  # A 區左緣（場地座標）
FIELD_CORNERS = np.float32([[40, 112], [606, 100], [612, 396], [32, 404]])   # 環境相機中的場地 [tl,tr,br,bl]
TAB_MM = [[150, 300], [210, 300], [210, 335], [150, 335]]                    # 下緣凸出的手臂底座（粉紅）

CAP_D_MM = 20.0
HOLE_MM = 28.0
# 8 個方孔中心（場地座標 mm），依格位 1..8：最下排（y 大）左 1、右 2，往上 3、4 …
HOLE_COLS_X = (132.0, 166.0)
HOLE_ROWS_Y = (207.0, 162.0, 117.0, 72.0)        # 由下往上
SLOT_CENTERS_MM = [(x, y) for y in HOLE_ROWS_Y for x in HOLE_COLS_X]

# A 區 8 顆瓶蓋（場地座標 mm；中間 (500,150) 空著）
A_CAPS_MM = [(440, 60), (500, 60), (560, 60), (440, 150), (560, 150), (440, 240), (500, 240), (560, 240)]

# 手眼相機 2：俯看 C 區（含一點 B 區），場地平面這塊矩形 -> 影像四邊形（約 2 px/mm、帶透視）
ARM2_PLANE_MM = np.float32([[60, 30], [250, 30], [250, 250], [60, 250]])
ARM2_IMG = np.float32([[110, 30], [530, 24], [548, 456], [96, 462]])

Pt = Tuple[float, float]


# --------------------------------------------------------------------------- #
# 影像 I/O（Windows 上 cv2.imread 不吃中文路徑，改走 imdecode / tofile）
# --------------------------------------------------------------------------- #
def imread(path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f'讀不到影像 {path}')
    return img


def imwrite(path, img: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode('.png', img)
    if not ok:
        raise IOError(f'編碼失敗 {path}')
    buf.tofile(str(path))


# --------------------------------------------------------------------------- #
# 座標轉換
# --------------------------------------------------------------------------- #
def env_field_to_img() -> np.ndarray:
    """場地平面 mm -> 環境相機影像像素。"""
    W, H = FIELD_SIZE_MM
    return cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]), FIELD_CORNERS)


def arm2_field_to_img() -> np.ndarray:
    """場地平面 mm -> 手眼相機 2 影像像素（未旋轉）。"""
    return cv2.getPerspectiveTransform(ARM2_PLANE_MM, ARM2_IMG)


def zone_corners_img(zone: str) -> np.ndarray:
    """某區在環境相機影像中的角點 [tl,tr,br,bl]。"""
    x0 = {'c': 0.0, 'b': ZONE_W_MM, 'a': 2 * ZONE_W_MM}[zone]
    pts = np.float32([[x0, 0], [x0 + ZONE_W_MM, 0], [x0 + ZONE_W_MM, 300], [x0, 300]])
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), env_field_to_img()).reshape(-1, 2)


def a_plane_to_img() -> np.ndarray:
    """A 區平面 mm（原點 = A 區左上角）-> 環境相機影像像素。"""
    return env_field_to_img() @ np.array([[1, 0, A_X0_MM], [0, 1, 0], [0, 0, 1]], np.float64)


# --------------------------------------------------------------------------- #
# 繪圖
# --------------------------------------------------------------------------- #
def _poly(img: np.ndarray, H: np.ndarray, pts_mm, color) -> None:
    ip = cv2.perspectiveTransform(np.float32(pts_mm).reshape(-1, 1, 2), H).reshape(-1, 2)
    cv2.fillConvexPoly(img, np.round(ip).astype(np.int32), color, cv2.LINE_AA)


def _square(c: Pt, side: float):
    x, y = c
    h = side / 2.0
    return [[x - h, y - h], [x + h, y - h], [x + h, y + h], [x - h, y + h]]


def _circle(c: Pt, d: float, n: int = 40):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([c[0] + d / 2 * np.cos(t), c[1] + d / 2 * np.sin(t)], axis=1)


def _desk(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    w, h = IMG_SIZE
    img = np.empty((h, w, 3), np.uint8)
    img[:] = DESK
    shade = (np.linspace(-6, 6, w))[None, :, None] + rng.randn(h, 1, 1) * 1.5
    return np.clip(img.astype(np.float32) + shade, 0, 255).astype(np.uint8)


def _finish(img: np.ndarray, seed: int, sigma: float = 3.0) -> np.ndarray:
    rng = np.random.RandomState(seed + 1000)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    noisy = img.astype(np.float32) + rng.randn(*img.shape) * sigma
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _draw_field(img: np.ndarray, H: np.ndarray, caps_mm: Sequence[Pt], filled_slots: Sequence[int],
                arm_in_b: bool) -> None:
    W, Hh = FIELD_SIZE_MM
    _poly(img, H, [[0, 0], [W, 0], [W, Hh], [0, Hh]], PINK)
    _poly(img, H, TAB_MM, PINK)
    for c in SLOT_CENTERS_MM:
        _poly(img, H, _square(c, HOLE_MM), WHITE)
    for s in filled_slots:
        _poly(img, H, _circle(SLOT_CENTERS_MM[s - 1], CAP_D_MM), BLUE)
    for c in caps_mm:
        _poly(img, H, _circle(c, CAP_D_MM), BLUE)
    if arm_in_b:   # 手臂從上方伸進 B 區
        _poly(img, H, [[230, -40], [370, -40], [370, 200], [230, 200]], BLACK)


# --------------------------------------------------------------------------- #
# 場景
# --------------------------------------------------------------------------- #
def make_env_scene(a_caps: Sequence[Pt] = (), filled_slots: Sequence[int] = (), b_caps: Sequence[Pt] = (),
                   arm_in_b: bool = False, seed: int = 0) -> np.ndarray:
    """環境相機場景。a_caps / b_caps 為場地座標 mm；filled_slots 為 C 區已放瓶蓋的格位編號。"""
    img = _desk(seed)
    _draw_field(img, env_field_to_img(), list(a_caps) + list(b_caps), filled_slots, arm_in_b)
    return _finish(img, seed)


def make_arm2_scene(filled_slots: Sequence[int] = (), rotation_k: int = 0, seed: int = 0,
                    extra_caps_mm: Sequence[Pt] = (), cover_slot: Optional[int] = None) -> np.ndarray:
    """手眼相機 2 視角（俯看 C 區）。rotation_k：影像逆時針轉 k 個 90°（模擬相機裝的方向不同），
    節點的 view_rotation_k 設成同一個 k 就能還原成使用者視角。cover_slot：用黑色手臂遮住該孔（測沿用孔位）。"""
    img = _desk(seed)
    H = arm2_field_to_img()
    _draw_field(img, H, extra_caps_mm, filled_slots, False)
    if cover_slot is not None:   # 夾爪從右邊伸進來蓋住該孔（一路延伸到影像外，不會形成封閉的洞）
        x, y = SLOT_CENTERS_MM[cover_slot - 1]
        _poly(img, H, [[x - 20, y - 20], [400, y - 20], [400, y + 20], [x - 20, y + 20]], BLACK)
    img = _finish(img, seed)
    return np.ascontiguousarray(np.rot90(img, int(rotation_k) % 4))
