"""合成靜態場景（含透視、木紋桌面、噪點），供 pytest 與 hsv_tuner 離線練習使用。

直接執行  python tests/make_test_images.py  會把這些場景輸出成 PNG 到 tests/data/。
座標約定與正式系統一致：站台平面 mm，原點為站台左上角，x 向右、y 向下。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# 顏色（BGR）。木頭色刻意做成「橘褐色」，用來驗證橘色 HSV 不會把桌面誤判成物件。
WOOD = (86, 132, 176)
BLUE = (200, 90, 20)
GREEN = (60, 170, 40)
WHITE = (235, 235, 235)
ORANGE = (0, 128, 255)
BLACK = (25, 25, 25)

IMG_SIZE = (640, 480)  # (w, h)

# 環境相機視角下 A/B/C 站台的影像角點 [tl,tr,br,bl]（帶一點傾斜/透視）
ENV_ZONES: Dict[str, np.ndarray] = {
    'a': np.float32([[50, 250], [250, 256], [246, 448], [44, 442]]),
    'b': np.float32([[290, 40], [450, 36], [456, 196], [294, 202]]),
    'c': np.float32([[470, 250], [620, 244], [624, 430], [466, 436]]),
}
ZONE_SIZE_MM = (200.0, 200.0)

# 上面的舊場景（三塊不同顏色的站台）要用 color 定位模式；env_vision.yaml 預設已改成 field 模式
LEGACY_ENV_PARAMS = {
    'zone_locate_mode': 'color',
    'zone_color.a': 'blue', 'zone_color.b': 'green', 'zone_color.c': 'c_zone',
    'zone_size_mm.a': list(ZONE_SIZE_MM), 'zone_size_mm.b': list(ZONE_SIZE_MM), 'zone_size_mm.c': list(ZONE_SIZE_MM),
    'roi_erode_px': 4,
}

# 外框場地（field 模式）：60x30 cm 黑色外框、內部白色，A|B|C 各 20x30 cm，中間有分隔線
FIELD_CORNERS = np.float32([[40, 112], [606, 100], [612, 396], [32, 404]])   # 外框外緣 [tl,tr,br,bl]
FIELD_SIZE_MM = (600.0, 300.0)
FIELD_BORDER_MM = 12.0     # 外框線寬
FIELD_DIVIDER_MM = 8.0     # 分隔線寬
FIELD_A_OFFSET_MM = (0.0, 50.0)   # a_objects() 的座標（200x200 範圍）放進 A 區（200x300）時的位移

# A 區 8 個物件：尺寸 mm 與旋轉角（大的角度小，避免相鄰物件互相碰到）
A_SIZES = [20, 24, 28, 32, 36, 40, 44, 48]
A_ANGLES = {20: 40, 24: 25, 28: 35, 32: 10, 36: 20, 40: 5, 44: 12, 48: 0}
A_POSITIONS = [(34, 34), (100, 34), (166, 34), (34, 100), (166, 100), (34, 166), (100, 166), (166, 166)]
A_PERM = [4, 0, 6, 2, 7, 1, 5, 3]  # 尺寸 i 放在 A_POSITIONS[A_PERM[i]]，讓排序結果與空間位置無關

# arm2 視角下 C 站台（2 排 x 4 欄，每格 50x50 mm）
ARM2_C = np.float32([[110, 130], [510, 126], [516, 336], [106, 340]])
C_SIZE_MM = (200.0, 100.0)

ObjSpec = Tuple[float, float, float, float]  # cx_mm, cy_mm, side_mm, angle_deg


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
# 繪圖
# --------------------------------------------------------------------------- #
def plane_to_img_H(corners: np.ndarray, size_mm: Sequence[float]) -> np.ndarray:
    w, h = size_mm
    return cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32(corners))


def draw_square(img: np.ndarray, H: np.ndarray, spec: ObjSpec, color=ORANGE) -> None:
    cx, cy, side, ang = spec
    half = side / 2.0
    pts = np.float32([[-half, -half], [half, -half], [half, half], [-half, half]])
    a = np.deg2rad(ang)
    R = np.float32([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = pts @ R.T + np.float32([cx, cy])
    ip = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    cv2.fillConvexPoly(img, np.round(ip).astype(np.int32), color, cv2.LINE_AA)


def _desk(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    w, h = IMG_SIZE
    img = np.empty((h, w, 3), np.uint8)
    img[:] = WOOD
    grain = (np.sin(np.linspace(0, 60, h)) * 6)[:, None, None] + rng.randn(h, 1, 1) * 2
    return np.clip(img.astype(np.float32) + grain, 0, 255).astype(np.uint8)


def _finish(img: np.ndarray, seed: int, sigma: float = 3.0) -> np.ndarray:
    rng = np.random.RandomState(seed + 1000)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    noisy = img.astype(np.float32) + rng.randn(*img.shape) * sigma
    return np.clip(noisy, 0, 255).astype(np.uint8)


def a_objects(sizes: Optional[Sequence[float]] = None, skip: Sequence[int] = ()) -> List[ObjSpec]:
    """A 區 8 個物件的規格（平面 mm）。skip 為要略過的「尺寸索引」。"""
    sizes = list(A_SIZES if sizes is None else sizes)
    out = []
    for i, s in enumerate(sizes):
        if i in skip:
            continue
        x, y = A_POSITIONS[A_PERM[i]]
        out.append((x, y, s, A_ANGLES.get(int(round(s)), 15)))
    return out


# --------------------------------------------------------------------------- #
# 場景
# --------------------------------------------------------------------------- #
def make_env_scene(a_objs: Sequence[ObjSpec] = (), b_objs: Sequence[ObjSpec] = (),
                   c_objs: Sequence[ObjSpec] = (), arm_in_b: bool = False, seed: int = 0,
                   extra_draw=None) -> np.ndarray:
    """環境相機場景：木紋桌面 + 藍(A)/綠(B)/白(C)站台 + 物件 + 可選的手臂（黑色）。"""
    img = _desk(seed)
    colors = {'a': BLUE, 'b': GREEN, 'c': WHITE}
    for name, c in ENV_ZONES.items():
        cv2.fillConvexPoly(img, np.round(c).astype(np.int32), colors[name], cv2.LINE_AA)
    for name, objs in (('a', a_objs), ('b', b_objs), ('c', c_objs)):
        H = plane_to_img_H(ENV_ZONES[name], ZONE_SIZE_MM)
        for o in objs:
            draw_square(img, H, o)
    if arm_in_b:  # 手臂從左側伸進 B 區，遮住約 6 成綠色
        cv2.fillConvexPoly(img, np.int32([[270, 20], [392, 20], [392, 222], [270, 222]]), BLACK)
    if extra_draw is not None:
        extra_draw(img)
    return _finish(img, seed)


def make_field_scene(a_objs: Sequence[ObjSpec] = (), b_arm: bool = False, seed: int = 0) -> np.ndarray:
    """外框場地：木紋桌面 + 黑色外框（含兩條分隔線）+ 白色內部。a_objs 為 A 區平面座標（加上 FIELD_A_OFFSET_MM）。"""
    img = _desk(seed)
    H = plane_to_img_H(FIELD_CORNERS, FIELD_SIZE_MM)
    W, Hh = FIELD_SIZE_MM
    b, d = FIELD_BORDER_MM, FIELD_DIVIDER_MM

    def poly(pts_mm, color):
        ip = cv2.perspectiveTransform(np.float32(pts_mm).reshape(-1, 1, 2), H).reshape(-1, 2)
        cv2.fillConvexPoly(img, np.round(ip).astype(np.int32), color, cv2.LINE_AA)

    poly([[0, 0], [W, 0], [W, Hh], [0, Hh]], BLACK)
    poly([[b, b], [W - b, b], [W - b, Hh - b], [b, Hh - b]], WHITE)
    for x in (W / 3, 2 * W / 3):
        poly([[x - d / 2, 0], [x + d / 2, 0], [x + d / 2, Hh], [x - d / 2, Hh]], BLACK)
    ox, oy = FIELD_A_OFFSET_MM
    for cx, cy, side, ang in a_objs:
        draw_square(img, H, (cx + ox, cy + oy, side, ang))
    if b_arm:   # 手臂從上方伸進 B 區
        poly([[230, -40], [370, -40], [370, 200], [230, 200]], BLACK)
    return _finish(img, seed)


def c_slot_objects(slot_sides: Sequence[Optional[float]], rows: int = 2, cols: int = 4,
                   size_mm: Sequence[float] = C_SIZE_MM, jitter: float = 1.0, seed: int = 0) -> List[ObjSpec]:
    """依各格邊長產生放在格中心（帶少量偏移與旋轉）的物件規格；None = 該格空著。"""
    rng = np.random.RandomState(seed)
    cw, ch = size_mm[0] / cols, size_mm[1] / rows
    out = []
    for i, s in enumerate(slot_sides):
        if s is None:
            continue
        r, c = divmod(i, cols)
        out.append(((c + 0.5) * cw + rng.uniform(-jitter, jitter), (r + 0.5) * ch + rng.uniform(-jitter, jitter),
                    s, rng.uniform(-3, 3)))
    return out


def make_arm2_scene(slot_sides: Sequence[Optional[float]], rotation_k: int = 0, seed: int = 0,
                    extra_objs: Sequence[ObjSpec] = ()) -> np.ndarray:
    """手眼相機 2 視角的 C 區（白色站台）。rotation_k 模擬相機相對站台轉了 k*90°：
    平面原點落在影像第 k 個角點，與 geometry.rotate_corners(corners, k) 的定義一致。"""
    img = _desk(seed)
    cv2.fillConvexPoly(img, np.round(ARM2_C).astype(np.int32), WHITE, cv2.LINE_AA)
    plane_corners = np.roll(ARM2_C, -int(rotation_k), axis=0)
    H = plane_to_img_H(plane_corners, C_SIZE_MM)
    for o in list(c_slot_objects(slot_sides, seed=seed)) + list(extra_objs):
        draw_square(img, H, o)
    return _finish(img, seed)
