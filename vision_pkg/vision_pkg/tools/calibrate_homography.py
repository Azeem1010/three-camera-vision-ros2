#!/usr/bin/env python3
"""calibrate_homography：在 Home 姿態點選畫面上「已知座標」的點，計算並儲存 homography。

三種目標（--target）：
  arm1       手眼相機 1：像素 -> 手臂 1 基座座標（單位 m）  -> arm1_homography.yaml
  a_station  手眼相機 1：像素 -> A 站平面座標（單位 mm）    -> a_station_homography.yaml
  arm2       手眼相機 2：像素 -> 手臂 2 基座座標（單位 m）  -> arm2_homography.yaml

    ros2 run vision_pkg calibrate_homography --target arm1
    ros2 run vision_pkg calibrate_homography --target a_station --corners          # 點 A 站四角自動帶入座標
    ros2 run vision_pkg calibrate_homography --target arm2 --points world_arm2.txt # 世界座標由檔案提供
    # 完全無視窗（批次）：像素與世界座標都由檔案提供
    ros2 run vision_pkg calibrate_homography --target arm1 --pixels px.txt --points world.txt --image home_a.png

操作（互動模式）
  SPACE 凍結/解除凍結畫面（先讓手臂到 Home 姿態再凍結）
  凍結後用滑鼠左鍵點一個「已知座標」的點；接著在終端機輸入該點的 x y（座標單位見上；用空白或逗號分隔）
  u 刪除上一點   r 全部重來   c 計算並顯示每點誤差   s 存檔（會先計算）   q / Esc 離開
至少 4 點，建議 8 點以上、分散整個工作區（不要共線）。

高度很重要：標定點請放在「要夾取/放置的那個高度」（物件頂面高度）。homography 只對應一個平面，
點不在該平面上，轉出來的座標會有視差誤差。--target a_station 也一樣，須與環境相機量到的物件座標同一高度。
座標檔格式（--points / --pixels）：每行一點 "x y"，# 開頭為註解，與點選順序對應。
輸入影像若要套用去畸變，加 --undistort（須與視覺節點的 undistort 設定一致）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from vision_pkg import calib_io
from vision_pkg.debug_draw import put_text
from vision_pkg.tools.common import (add_source_args, imread, open_source, require_display,
                                     strip_ros_args)
from vision_pkg.vision_core.geometry import compute_homography

TARGETS = {
    # target: (預設相機, 輸出檔名, 單位)
    'arm1': ('arm1', 'arm1_homography', 'm'),
    'a_station': ('arm1', 'a_station_homography', 'mm'),
    'arm2': ('arm2', 'arm2_homography', 'm'),
}


def read_pairs(path: str) -> List[Tuple[float, float]]:
    out = []
    for ln in Path(path).read_text(encoding='utf-8').splitlines():
        ln = ln.split('#')[0].strip()
        if not ln:
            continue
        v = [float(t) for t in ln.replace(',', ' ').split()]
        if len(v) != 2:
            raise ValueError(f'座標檔每行必須是兩個數字: {ln!r}')
        out.append((v[0], v[1]))
    return out


def solve(pixel_pts, world_pts, unit: str, ransac: Optional[float]):
    """回傳 (H, 每點誤差(單位同世界座標), rms)。"""
    H, err = compute_homography(pixel_pts, world_pts, ransac)
    return H, err, float(np.sqrt(np.mean(err ** 2)))


def fmt_err(e: float, unit: str) -> str:
    return f'{e * 1000:.2f} mm' if unit == 'm' else f'{e:.2f} mm'


def main():
    ap = argparse.ArgumentParser(description='Homography 標定', formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument('--target', choices=list(TARGETS), required=True)
    add_source_args(ap, 'arm1')
    ap.set_defaults(camera=None)   # 未指定時依 --target 決定預設相機
    ap.add_argument('--points', default='', help='世界座標檔（與點選順序對應）；未給則每點選完在終端機輸入')
    ap.add_argument('--pixels', default='', help='像素座標檔；有給就不開視窗（批次模式，需同時給 --points 或 --corners）')
    ap.add_argument('--corners', action='store_true', help='僅 a_station：依序點 A 站 左上、右上、右下、左下，座標自動帶入')
    ap.add_argument('--station-size-mm', type=float, nargs=2, default=None, metavar=('W', 'H'),
                    help='--corners 用的 A 站尺寸（預設讀 config/arm1_vision.yaml 的 a_station_size_mm）')
    ap.add_argument('--ransac', type=float, default=None, help='RANSAC 容許誤差（世界座標單位）；預設最小平方')
    ap.add_argument('--dry-run', action='store_true', help='只計算與顯示，不存檔')
    args = ap.parse_args(strip_ros_args(sys.argv)[1:])

    default_cam, fname, unit = TARGETS[args.target]
    if args.camera is None:
        args.camera = default_cam
    calib_dir = calib_io.resolve_calib_dir(args.calib_dir)
    out = calib_dir / f'{fname}.yaml'

    if args.corners and args.target != 'a_station':
        sys.exit('--corners 只適用於 --target a_station')
    corner_world: List[Tuple[float, float]] = []
    if args.corners:
        if args.station_size_mm:
            w, h = args.station_size_mm
        else:
            from vision_pkg.vision_core.params_util import load_ros_params_yaml
            w, h = load_ros_params_yaml(calib_io.package_config_dir() / 'arm1_vision.yaml')['a_station_size_mm']
        corner_world = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
        print(f'A 站尺寸 {w} x {h} mm；請依序點 左上、右上、右下、左下（x 向右、y 向下）')
    given_world = read_pairs(args.points) if args.points else []

    def finish(pixel_pts, world_pts, size_wh) -> int:
        H, err, rms = solve(pixel_pts, world_pts, unit, args.ransac)
        print('\n每點誤差:')
        for i, (p, w, e) in enumerate(zip(pixel_pts, world_pts, err), start=1):
            print(f'  #{i}  pixel=({p[0]:.1f},{p[1]:.1f})  world=({w[0]:.4f},{w[1]:.4f})  誤差 {fmt_err(e, unit)}')
        print(f'RMS 誤差 {fmt_err(rms, unit)}，最大 {fmt_err(float(err.max()), unit)}')
        print('H =\n', np.array2string(H, precision=6, suppress_small=True))
        if not args.dry_run:
            calib_io.save_homography(out, H, unit, size_wh, pixel_pts, world_pts, rms, args.undistort,
                                     header=f'# {args.target}：影像像素 -> 座標（單位 {unit}）。由 calibrate_homography 產生。\n')
            print(f'已存檔 -> {out}')
        return 0

    # ---------- 批次模式 ----------
    if args.pixels:
        px = read_pairs(args.pixels)
        world = corner_world or given_world
        if len(px) != len(world):
            sys.exit(f'像素點 {len(px)} 個，但世界座標 {len(world)} 個，數量要相同')
        size = (640, 480)
        if args.image:
            im = imread(args.image)
            size = (im.shape[1], im.shape[0])
        sys.exit(finish(px, world, size))

    # ---------- 互動模式 ----------
    require_display()
    src = open_source(args, 'calibrate_homography')
    win = f'calibrate_homography [{args.target}]'
    pts_px: List[Tuple[float, float]] = []
    pts_world: List[Tuple[float, float]] = []
    state = {'frozen': None, 'click': None, 'H': None, 'err': None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and state['frozen'] is not None:
            state['click'] = (float(x), float(y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print(__doc__)
    size = (640, 480)
    try:
        while True:
            live = src.read()
            base = state['frozen'] if state['frozen'] is not None else live
            if base is None:
                if (cv2.waitKey(30) & 0xFF) in (27, ord('q')):
                    break
                continue
            size = (base.shape[1], base.shape[0])
            view = base.copy()

            if state['click'] is not None:
                cx, cy = state['click']
                state['click'] = None
                idx = len(pts_px)
                wpt = None
                if corner_world:
                    if idx < 4:
                        wpt = corner_world[idx]
                    else:
                        print('四個角已點完，按 c 計算 / s 存檔')
                elif given_world:
                    if idx < len(given_world):
                        wpt = given_world[idx]
                    else:
                        print('座標檔的點都用完了')
                else:
                    try:
                        raw = input(f'點 #{idx + 1} 像素=({cx:.0f},{cy:.0f})，輸入世界座標 x y（{unit}）: ')
                        v = [float(t) for t in raw.replace(',', ' ').split()]
                        if len(v) == 2:
                            wpt = (v[0], v[1])
                        else:
                            print('需要兩個數字，已略過這次點選')
                    except (ValueError, EOFError):
                        print('輸入無法解析，已略過這次點選')
                if wpt is not None:
                    pts_px.append((cx, cy))
                    pts_world.append(wpt)
                    state['H'] = None
                    print(f'  已加入 #{len(pts_px)}  pixel=({cx:.0f},{cy:.0f})  world={wpt}')

            for i, p in enumerate(pts_px):
                col = (0, 255, 255)
                if state['err'] is not None and i < len(state['err']):
                    e = state['err'][i]
                    col = (0, 255, 0) if (e * (1000 if unit == 'm' else 1)) < 3.0 else (0, 165, 255)
                cv2.drawMarker(view, (int(p[0]), int(p[1])), col, cv2.MARKER_CROSS, 14, 2)
                label = f'{i + 1}'
                if state['err'] is not None and i < len(state['err']):
                    label += f' {fmt_err(state["err"][i], unit)}'
                put_text(view, label, (p[0] + 8, p[1] - 8), col, 0.5, 1)
            status = 'FROZEN - click points' if state['frozen'] is not None else 'LIVE - press SPACE to freeze'
            put_text(view, f'{args.target} ({unit})  points={len(pts_px)}  {status}', (8, 20), (0, 255, 255), 0.55, 1)
            put_text(view, 'SPACE freeze | u undo | r reset | c compute | s save | q quit', (8, size[1] - 10),
                     (255, 255, 255), 0.45, 1)
            cv2.imshow(win, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord(' '):
                state['frozen'] = None if state['frozen'] is not None else (live.copy() if live is not None else None)
            elif key == ord('u') and pts_px:
                pts_px.pop(); pts_world.pop(); state['err'] = None
            elif key == ord('r'):
                pts_px.clear(); pts_world.clear(); state['err'] = None
            elif key in (ord('c'), ord('s')):
                if len(pts_px) < 4:
                    print(f'至少需要 4 點（目前 {len(pts_px)}）')
                    continue
                try:
                    H, err, rms = solve(pts_px, pts_world, unit, args.ransac)
                except ValueError as e:
                    print(f'計算失敗: {e}')
                    continue
                state['H'], state['err'] = H, err
                print(f'RMS 誤差 {fmt_err(rms, unit)}，最大 {fmt_err(float(err.max()), unit)}')
                if key == ord('s') and not args.dry_run:
                    calib_io.save_homography(out, H, unit, size, pts_px, pts_world, rms, args.undistort,
                                             header=f'# {args.target}：影像像素 -> 座標（單位 {unit}）。由 calibrate_homography 產生。\n')
                    print(f'已存檔 -> {out}（執行中的節點會自動重新載入）')
    finally:
        src.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
