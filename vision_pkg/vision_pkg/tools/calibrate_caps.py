#!/usr/bin/env python3
"""calibrate_caps：用「瓶蓋 + 尺量」標定手眼相機 1 的 homography（像素 -> 手臂 1 基座座標，m）。

不需要用手臂 TCP 去碰點：在 A 區放幾顆瓶蓋、用尺量每顆瓶蓋中心到兩條板邊的距離，
程式自動抓瓶蓋中心像素，依場地幾何換算成基座座標後計算 homography，寫入 <calib_dir>/arm1_homography.yaml。

手臂 1 基座座標系（REP-103，單位 m；由 3D 模型 dual_arm_3d-main 推得，場地依 README 每塊 300x300）：
  原點  手臂 1 底座旋轉軸與場地頂面的交點
  x     朝場地內（離開手臂底座小板的方向）    y  朝左    z  朝上
  A 區  x = 0.020 ~ 0.320，y = -0.200 ~ 0（旋轉軸在板子前緣外 20 mm，對齊 A/B 分界）

量測定義（單位 mm，量到「瓶蓋中心」；量到瓶蓋邊緣的話 +10 mm）：
  front  瓶蓋中心到「前緣」的距離：前緣 = 手臂底座那一側的長邊（60 cm 那條）
  side   瓶蓋中心到「A 區外側邊」的距離：A 區外側邊 = A 區最外面那條短邊（30 cm 那條，場地盡頭）
  換算：x = front_edge_x + front，y = outer_edge_y + side

步驟
  1. 手臂 1 到 Home_A 並停穩。A 區放 6 顆以上瓶蓋，分散在畫面四角與中間，彼此不要相碰、不要碰到畫面邊緣。
  2. ros2 run vision_pkg calibrate_caps
     -> 抓 N 張影格取瓶蓋中心中位數，存 <calib_dir>/arm1_caps_numbered.png（含編號）與 arm1_caps_raw.png
  3. 依編號在終端機輸入每顆的 "front side"（mm）；或事先寫好檔案用 --measure（每行 "front side"，順序同編號）
  4. 看每點誤差（目標 < 3 mm），確認後存檔；執行中的 arm1_vision_node 會自動重新載入。

    ros2 run vision_pkg calibrate_caps --measure caps_mm.txt
    ros2 run vision_pkg calibrate_caps --image ~/.ros/vision_calib/arm1_caps_raw.png --measure caps_mm.txt   # 離線重算
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from vision_pkg import calib_io
from vision_pkg.tools.common import add_source_args, open_source, strip_ros_args

# 由 3D 模型推得的場地幾何（mm，手臂 1 基座座標）
FRONT_EDGE_X_MM = 20.0      # 板子前緣（手臂底座側長邊）在基座 x
OUTER_EDGE_Y_MM = -200.0    # A 區外側短邊在基座 y（A 區寬 200 mm，位於基座右側）


# --------------------------------------------------------------------------- #
# 純函式（可離線測試）
# --------------------------------------------------------------------------- #
def field_to_base(front_mm: float, side_mm: float, front_edge_x_mm: float = FRONT_EDGE_X_MM,
                  outer_edge_y_mm: float = OUTER_EDGE_Y_MM) -> Tuple[float, float]:
    """尺量距離（mm）-> 手臂 1 基座座標（m）。"""
    return (front_edge_x_mm + front_mm) / 1000.0, (outer_edge_y_mm + side_mm) / 1000.0


def order_caps(centers: np.ndarray) -> np.ndarray:
    """編號順序：由上到下分列（y 相差 < 30px 視為同列），列內由左到右。回傳排序索引。"""
    if len(centers) == 0:
        return np.zeros(0, int)
    idx = np.argsort(centers[:, 1])
    rows, cur = [], [idx[0]]
    for i in idx[1:]:
        if abs(centers[i, 1] - centers[cur[-1], 1]) < 30:
            cur.append(i)
        else:
            rows.append(cur)
            cur = [i]
    rows.append(cur)
    return np.array([i for r in rows for i in sorted(r, key=lambda j: centers[j, 0])])


def stable_centers(per_frame: Sequence[np.ndarray], match_px: float = 8.0) -> Tuple[np.ndarray, str]:
    """多張影格的瓶蓋中心 -> 每顆的中位數中心。

    以出現最多次的瓶蓋數當基準，取其中第一張為參考；其他影格中距離參考點 < match_px 的視為同一顆。
    回傳 (中心陣列 Nx2, 警告訊息)。
    """
    frames = [np.asarray(f, float).reshape(-1, 2) for f in per_frame if len(f)]
    if not frames:
        return np.zeros((0, 2)), '沒有任何影格偵測到瓶蓋'
    counts = [len(f) for f in frames]
    mode = max(set(counts), key=counts.count)
    ref = next(f for f in frames if len(f) == mode)
    out, warn = [], []
    for k, p in enumerate(ref):
        hits = []
        for f in frames:
            d = np.linalg.norm(f - p, axis=1)
            if d.min() < match_px:
                hits.append(f[d.argmin()])
        hits = np.array(hits)
        out.append(np.median(hits, axis=0))
        if len(hits) < 0.8 * len(frames):
            warn.append(f'瓶蓋 @({p[0]:.0f},{p[1]:.0f}) 只在 {len(hits)}/{len(frames)} 張影格出現')
        spread = float(np.max(np.linalg.norm(hits - out[-1], axis=1))) if len(hits) else 0.0
        if spread > 3.0:
            warn.append(f'瓶蓋 @({p[0]:.0f},{p[1]:.0f}) 中心晃動 {spread:.1f}px（手臂或瓶蓋還在動？）')
    if len(set(counts)) > 1:
        warn.append(f'各影格瓶蓋數不一致 {sorted(set(counts))}，以 {mode} 顆為準')
    return np.array(out), '；'.join(warn)


def read_measure(path: str) -> List[Tuple[float, float]]:
    out = []
    for ln in Path(path).read_text(encoding='utf-8').splitlines():
        ln = ln.split('#')[0].strip()
        if not ln:
            continue
        v = [float(t) for t in ln.replace(',', ' ').split()]
        if len(v) != 2:
            raise ValueError(f'每行必須是 "front side" 兩個數字（mm）: {ln!r}')
        out.append((v[0], v[1]))
    return out


def draw_numbered(img: np.ndarray, centers: np.ndarray, diam: Optional[Sequence[float]] = None) -> np.ndarray:
    view = img.copy()
    for n, (x, y) in enumerate(centers, start=1):
        r = int(diam[n - 1] / 2) if diam is not None else 12
        cv2.circle(view, (int(round(x)), int(round(y))), max(r, 6), (0, 255, 255), 2)
        cv2.drawMarker(view, (int(round(x)), int(round(y))), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
        cv2.putText(view, str(n), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(view, str(n), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return view


# --------------------------------------------------------------------------- #
def _detector(calib_dir: Path, max_area_override: Optional[float]):
    from vision_pkg.vision_core.cap_detect import CapParams, detect_caps
    from vision_pkg.vision_core.color_zone import MaskParams, hsv_mask
    from vision_pkg.vision_core.params_util import dataclass_from_flat, load_ros_params_yaml
    P = load_ros_params_yaml(calib_io.package_config_dir() / 'arm1_vision.yaml')
    cp = dataclass_from_flat(CapParams, P, 'cap')
    if max_area_override:
        cp = CapParams(**{**cp.__dict__, 'max_area_px': float(max_area_override)})
    mp = dataclass_from_flat(MaskParams, P, 'object_mask')
    color = P.get('object_color', 'blue')
    hsv = calib_io.load_hsv_ranges(calib_dir)[color]

    def detect(bgr):
        caps, rejected = detect_caps(hsv_mask(bgr, hsv, mp), cp)
        return caps, rejected
    return detect


def main():
    ap = argparse.ArgumentParser(description='瓶蓋 + 尺量 標定 arm1_homography', epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap, 'arm1')
    ap.add_argument('--measure', default='', help='量測檔：每行 "front side"（mm），順序同畫面編號')
    ap.add_argument('--frames', type=int, default=15, help='取幾張影格平均瓶蓋中心（預設 15）')
    ap.add_argument('--max-area-px', type=float, default=None,
                    help='覆寫 cap.max_area_px（鏡頭很近、瓶蓋在畫面中很大時用）')
    ap.add_argument('--front-edge-x-mm', type=float, default=FRONT_EDGE_X_MM, help='前緣在基座 x（mm）')
    ap.add_argument('--outer-edge-y-mm', type=float, default=OUTER_EDGE_Y_MM, help='A 區外側邊在基座 y（mm）')
    ap.add_argument('--dry-run', action='store_true', help='只計算，不存檔')
    ap.add_argument('--yes', action='store_true', help='不詢問直接存檔')
    args = ap.parse_args(strip_ros_args(sys.argv)[1:])

    calib_dir = calib_io.resolve_calib_dir(args.calib_dir)
    calib_dir.mkdir(parents=True, exist_ok=True)
    detect = _detector(calib_dir, args.max_area_px)

    src = open_source(args, 'calibrate_caps')
    per_frame, diams, last, rejected = [], [], None, []
    try:
        t0 = time.monotonic()
        while len(per_frame) < max(1, args.frames) and time.monotonic() - t0 < 15.0:
            f = src.read(0.1)
            if f is None:
                continue
            if args.image and per_frame:   # 靜態圖片只需一張
                break
            last = f
            caps, rejected = detect(f)
            per_frame.append(np.array([(c.cx, c.cy) for c in caps]).reshape(-1, 2))
            diams.append([c.diameter for c in caps])
    finally:
        src.close()
    if last is None:
        sys.exit('收不到影像（launch 有開嗎？ ros2 topic hz /arm1_cam/image_raw）')

    centers, warn = stable_centers(per_frame)
    if warn:
        print('警告:', warn)
    for r in rejected:
        print(f'  已排除的藍色區塊 @({r.cx:.0f},{r.cy:.0f}): {r.reject_reason}')
    order = order_caps(centers)
    centers = centers[order]
    d_med = float(np.median([d for ds in diams for d in ds])) if any(diams) else 0.0
    cv2.imwrite(str(calib_dir / 'arm1_caps_raw.png'), last)
    cv2.imwrite(str(calib_dir / 'arm1_caps_numbered.png'), draw_numbered(last, centers, [d_med] * len(centers)))
    print(f'\n偵測到 {len(centers)} 顆瓶蓋（{len(per_frame)} 張影格，直徑約 {d_med:.0f}px）')
    print(f'編號圖: {calib_dir / "arm1_caps_numbered.png"}')
    for n, (x, y) in enumerate(centers, start=1):
        print(f'  #{n}  pixel=({x:.1f}, {y:.1f})')
    if len(centers) < 4:
        sys.exit('至少需要 4 顆瓶蓋（建議 6 顆以上、分散畫面四角）')

    if args.measure:
        meas = read_measure(args.measure)
        if len(meas) != len(centers):
            sys.exit(f'量測檔有 {len(meas)} 行，但畫面上有 {len(centers)} 顆瓶蓋，數量要相同')
    else:
        print('\n依編號輸入瓶蓋中心到「前緣」與到「A 區外側邊」的距離（mm），例如: 85 120')
        meas = []
        for n in range(1, len(centers) + 1):
            while True:
                try:
                    v = [float(t) for t in input(f'  #{n} front side: ').replace(',', ' ').split()]
                    if len(v) == 2:
                        meas.append((v[0], v[1]))
                        break
                except ValueError:
                    pass
                except EOFError:
                    sys.exit('輸入中斷')
                print('  需要兩個數字')

    world = [field_to_base(f, s, args.front_edge_x_mm, args.outer_edge_y_mm) for f, s in meas]
    from vision_pkg.vision_core.geometry import compute_homography
    H, err = compute_homography([tuple(c) for c in centers], world)
    rms = float(np.sqrt(np.mean(err ** 2)))
    print('\n每點誤差:')
    for n, (c, (f, s), w, e) in enumerate(zip(centers, meas, world, err), start=1):
        flag = '' if e * 1000 < 3.0 else '  <-- 偏大，請重量這顆'
        print(f'  #{n} pixel=({c[0]:.1f},{c[1]:.1f})  front={f:.1f} side={s:.1f} mm  '
              f'-> base=({w[0]:.4f},{w[1]:.4f}) m  誤差 {e * 1000:.2f} mm{flag}')
    print(f'RMS {rms * 1000:.2f} mm，最大 {float(err.max()) * 1000:.2f} mm')
    if len(centers) == 4:
        print('（只有 4 點時誤差必為 0，無法檢查量測是否正確；建議 6 點以上）')

    if args.dry_run:
        return 0
    if not args.yes and not args.measure:
        if input('存檔？[y/N] ').strip().lower() not in ('y', 'yes'):
            print('未存檔')
            return 0
    out = calib_dir / 'arm1_homography.yaml'
    h, w = last.shape[:2]
    calib_io.save_homography(out, H, 'm', (w, h), [tuple(c) for c in centers], world, rms, args.undistort,
                             header='# arm1：影像像素 -> 手臂 1 基座座標（m）。由 calibrate_caps（瓶蓋+尺量）產生。\n'
                                    f'# 基座座標：原點=底座旋轉軸@場地頂面，x 朝場地內、y 朝左；'
                                    f'front_edge_x={args.front_edge_x_mm}mm outer_edge_y={args.outer_edge_y_mm}mm\n')
    print(f'已存檔 -> {out}（執行中的 arm1_vision_node 會自動重新載入）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
