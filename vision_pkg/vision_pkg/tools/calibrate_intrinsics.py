#!/usr/bin/env python3
"""calibrate_intrinsics：棋盤格相機內參校正（三顆鏡頭各存一份）。

    ros2 run vision_pkg calibrate_intrinsics --camera env  --cols 9 --rows 6 --square-mm 25
    ros2 run vision_pkg calibrate_intrinsics --camera arm1 ...
    ros2 run vision_pkg calibrate_intrinsics --camera arm2 ...
    # 離線：用已拍好的照片資料夾
    python3 -m vision_pkg.tools.calibrate_intrinsics --camera env --image-dir ./chess_env --cols 9 --rows 6 --square-mm 25

--cols/--rows 是棋盤格「內角點」數（不是方格數）。
按鍵：SPACE 擷取目前畫面（偵測到棋盤格時）   u 刪除上一張   c 計算並存檔   q / Esc 離開
      --auto 時偵測到且與上一張夠不一樣就自動擷取
輸出：<calib_dir>/<camera>_cam_intrinsics.yaml（camera_info 格式，usb_cam 的 camera_info_url 可直接讀）。
之後若要讓視覺節點使用去畸變：節點參數 undistort:=true（或 launch undistort:=true），
並且 calibrate_homography 也要加 --undistort，兩邊必須一致。

拍攝建議：20 張以上；棋盤格要涵蓋畫面四角與邊緣、多種傾斜角度與距離；棋盤格務必平整、固定；
RMS 重投影誤差一般應 < 0.5 px。
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

from vision_pkg import calib_io
from vision_pkg.debug_draw import put_text
from vision_pkg.tools.common import add_source_args, imread, open_source, require_display, strip_ros_args

FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def find_corners(bgr: np.ndarray, pattern):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, pattern, FIND_FLAGS)
    if ok:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
    return ok, corners


def calibrate(imgpoints: List[np.ndarray], pattern, square_mm: float, size_wh, fix_k3: bool = True):
    """回傳 (rms, K, dist, 每張影像的平均重投影誤差)。物體座標單位 mm（內參與單位無關）。

    fix_k3=True（預設）：把 k3 固定為 0。一般網路攝影機畸變不大，棋盤格又很難涵蓋整個畫面邊緣，
    放開 k3 常會過度擬合（k2/k3 出現很大的數值）；需要時用 --with-k3 放開。
    """
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * float(square_mm)
    objpoints = [objp] * len(imgpoints)
    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, tuple(size_wh), None, None, flags=flags)
    per_view = []
    for op, ip, rv, tv in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - ip.reshape(-1, 2)) ** 2, axis=1)))))
    return float(rms), K, dist, per_view


def save_result(path: Path, camera_name: str, size_wh, K, dist, rms: float) -> None:
    calib_io.save_camera_info_yaml(path, camera_name, size_wh, K, dist, rms)
    print(f'\n已存內參 -> {path}')
    print(f'  RMS 重投影誤差 {rms:.3f} px（< 0.5 較理想）')
    print(f'  fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}  dist={np.round(dist.ravel(), 4).tolist()}')
    print('  啟用去畸變：視覺節點 undistort:=true；calibrate_homography 加 --undistort（兩邊要一致）。')
    print(f'  usb_cam 讀取：camera_info_url: file://{Path(path).resolve()}（launch 偵測到此檔會自動帶入）')


def main():
    ap = argparse.ArgumentParser(description='棋盤格內參校正', formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    add_source_args(ap, 'env')
    ap.add_argument('--cols', type=int, required=True, help='棋盤格橫向內角點數')
    ap.add_argument('--rows', type=int, required=True, help='棋盤格縱向內角點數')
    ap.add_argument('--square-mm', type=float, required=True, help='方格邊長 (mm)')
    ap.add_argument('--num-frames', type=int, default=20, help='目標張數（達到後提示可校正）')
    ap.add_argument('--min-frames', type=int, default=10, help='至少要幾張才允許校正')
    ap.add_argument('--auto', action='store_true', help='自動擷取')
    ap.add_argument('--with-k3', action='store_true', help='放開 k3 畸變係數（預設固定為 0，較不易過度擬合）')
    ap.add_argument('--image-dir', default='', help='離線：資料夾內所有 png/jpg 當作標定影像（不開視窗）')
    ap.add_argument('--out', default='', help='輸出檔（預設 <calib_dir>/<camera>_cam_intrinsics.yaml）')
    args = ap.parse_args(strip_ros_args(sys.argv)[1:])

    pattern = (args.cols, args.rows)
    camera_name = calib_io.CAMERA_NAMES[args.camera]
    calib_dir = calib_io.resolve_calib_dir(args.calib_dir)
    out = Path(args.out) if args.out else calib_io.intrinsics_path(camera_name, calib_dir)

    # ---------- 離線模式 ----------
    if args.image_dir:
        files = sorted(sum((glob.glob(str(Path(args.image_dir) / e)) for e in ('*.png', '*.jpg', '*.jpeg', '*.bmp')), []))
        pts, size = [], None
        for f in files:
            img = imread(f)
            size = (img.shape[1], img.shape[0])
            ok, c = find_corners(img, pattern)
            print(f'{Path(f).name}: {"OK" if ok else "找不到棋盤格"}')
            if ok:
                pts.append(c)
        if len(pts) < args.min_frames:
            sys.exit(f'可用影像只有 {len(pts)} 張，至少需要 {args.min_frames} 張')
        rms, K, dist, per = calibrate(pts, pattern, args.square_mm, size, not args.with_k3)
        print('各張誤差 (px):', [round(e, 2) for e in per])
        save_result(out, camera_name, size, K, dist, rms)
        return

    # ---------- 即時模式 ----------
    require_display()
    args.undistort = False   # 標內參時必須用原始影像
    src = open_source(args, 'calibrate_intrinsics')
    win = 'calibrate_intrinsics'
    pts: List[np.ndarray] = []
    size = None
    last_t, last_c = 0.0, None
    result = None
    print(__doc__)
    try:
        while True:
            frame = src.read()
            if frame is None:
                if (cv2.waitKey(30) & 0xFF) in (27, ord('q')):
                    break
                continue
            size = (frame.shape[1], frame.shape[0])
            ok, corners = find_corners(frame, pattern)
            view = frame.copy()
            if ok:
                cv2.drawChessboardCorners(view, pattern, corners, ok)
            now = time.time()
            if args.auto and ok and now - last_t > 1.0 and \
                    (last_c is None or float(np.mean(np.linalg.norm(corners - last_c, axis=2))) > 15.0):
                pts.append(corners)
                last_t, last_c = now, corners
                print(f'自動擷取 {len(pts)}')
            put_text(view, f'captured {len(pts)}/{args.num_frames}  board:{"FOUND" if ok else "not found"}'
                           + (f'  RMS={result:.3f}px' if result is not None else ''), (8, 20),
                     (0, 255, 0) if ok else (0, 0, 255), 0.55, 1)
            put_text(view, 'SPACE capture | u undo | c calibrate+save | q quit', (8, size[1] - 10), (255, 255, 255), 0.45, 1)
            cv2.imshow(win, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord(' ') and ok:
                pts.append(corners)
                last_t, last_c = now, corners
                print(f'擷取 {len(pts)}')
            elif key == ord('u') and pts:
                pts.pop()
                print(f'刪除，剩 {len(pts)}')
            elif key == ord('c'):
                if len(pts) < args.min_frames:
                    print(f'至少需要 {args.min_frames} 張（目前 {len(pts)}）')
                    continue
                rms, K, dist, per = calibrate(pts, pattern, args.square_mm, size, not args.with_k3)
                print('各張誤差 (px):', [round(e, 2) for e in per], '（誤差特別大的視角建議重拍；u 只能刪最後一張）')
                save_result(out, camera_name, size, K, dist, rms)
                result = rms
    finally:
        src.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
