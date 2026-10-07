#!/usr/bin/env python3
"""hsv_tuner：訂閱影像 topic（或讀靜態圖片），用滑桿即時調整 HSV，並分色存檔。

    ros2 run vision_pkg hsv_tuner --camera env --color blue
    ros2 run vision_pkg hsv_tuner --topic /arm1_cam/image_raw --color orange
    python3 -m vision_pkg.tools.hsv_tuner --image tests/data/env_empty.png --color green   # 離線

按鍵
  1 blue(A 區)   2 green(B 區)   3 c_zone(C 區顏色)   4 orange(物件)   5 black(手臂)
  s 存目前這個顏色     a 存所有「有改過」的顏色     p 印出目前數值
  m 切換顯示（遮罩 / 原圖套遮罩）     滑鼠左鍵 點畫面印出該點 HSV     q / Esc 離開
存檔位置：<calib_dir>/hsv.yaml（逐色覆寫套件預設 config/hsv.yaml）。
正在執行的節點每 hsv_reload_period_s 秒會自動重新載入，不必重啟。

H 下限 > H 上限 代表色相環繞（例如紅色），與節點的行為一致。
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict

import cv2
import numpy as np

from vision_pkg import calib_io
from vision_pkg.debug_draw import put_text
from vision_pkg.tools.common import add_source_args, open_source, require_display, strip_ros_args
from vision_pkg.vision_core.color_zone import HsvRange, in_range_hsv, to_hsv

KEYS = {ord('1'): 'blue', ord('2'): 'green', ord('3'): 'c_zone', ord('4'): 'orange', ord('5'): 'black'}
BARS = [('H low', 179), ('H high', 179), ('S low', 255), ('S high', 255), ('V low', 255), ('V high', 255)]
WIN = 'hsv_tuner'


def _nop(_v):
    pass


def main():
    ap = argparse.ArgumentParser(description='HSV 即時調整工具', formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    add_source_args(ap, 'env')
    ap.add_argument('--color', choices=['blue', 'green', 'c_zone', 'orange', 'black'], default='orange',
                    help='一開始要調哪個顏色')
    ap.add_argument('--blur', type=int, default=5, help='預覽用高斯模糊核（0 = 不模糊；請與 config 的 *_mask.blur_ksize 一致）')
    ap.add_argument('--scale', type=float, default=0.6, help='視窗縮放比例')
    args = ap.parse_args(strip_ros_args(sys.argv)[1:])
    require_display()

    calib_dir = calib_io.resolve_calib_dir(args.calib_dir)
    edited: Dict[str, HsvRange] = dict(calib_io.load_hsv_ranges(calib_dir))
    saved: Dict[str, HsvRange] = dict(edited)   # 最近一次載入/存檔的值，用來顯示 *unsaved*
    current = args.color
    src = open_source(args, 'hsv_tuner')

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    init = list(edited[current].lower[:1]) + list(edited[current].upper[:1]) + \
        [edited[current].lower[1], edited[current].upper[1], edited[current].lower[2], edited[current].upper[2]]
    for (name, mx), v in zip(BARS, init):
        cv2.createTrackbar(name, WIN, int(v), mx, _nop)

    state = {'view': 0, 'click': None, 'frame_w': 640}

    def read_bars() -> HsvRange:
        v = [cv2.getTrackbarPos(n, WIN) for n, _ in BARS]
        return HsvRange((v[0], v[2], v[4]), (v[1], v[3], v[5]))

    def set_bars(r: HsvRange) -> None:
        for (n, _), val in zip(BARS, [r.lower[0], r.upper[0], r.lower[1], r.upper[1], r.lower[2], r.upper[2]]):
            cv2.setTrackbarPos(n, WIN, int(val))

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state['click'] = (x, y)

    cv2.setMouseCallback(WIN, on_mouse)
    print(__doc__)
    try:
        while True:
            frame = src.read()
            if frame is None:
                blank = np.zeros((120, 480, 3), np.uint8)
                put_text(blank, 'waiting for image...', (10, 60), (255, 255, 255), 0.7, 1)
                cv2.imshow(WIN, blank)
                if (cv2.waitKey(30) & 0xFF) in (27, ord('q')):
                    break
                continue
            rng = read_bars()
            edited[current] = rng
            hsv = to_hsv(frame, args.blur)
            mask = in_range_hsv(hsv, rng)
            masked = cv2.bitwise_and(frame, frame, mask=mask)
            right = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if state['view'] == 0 else masked
            left = frame.copy()
            cov = 100.0 * cv2.countNonZero(mask) / mask.size
            put_text(left, f'[{current}] H {rng.lower[0]}-{rng.upper[0]}  S {rng.lower[1]}-{rng.upper[1]}  '
                           f'V {rng.lower[2]}-{rng.upper[2]}  cover {cov:.1f}%'
                           + ('  (hue wrap)' if rng.lower[0] > rng.upper[0] else '')
                           + ('  *unsaved*' if rng != saved[current] else ''), (8, 18), (0, 255, 255), 0.5, 1)
            put_text(left, 'keys: 1 blue 2 green 3 c_zone 4 orange 5 black | s save | a save all | m view | q quit',
                     (8, frame.shape[0] - 8), (255, 255, 255), 0.45, 1)
            if state['click'] is not None:
                cx, cy = int(state['click'][0] / args.scale), int(state['click'][1] / args.scale)
                if 0 <= cx < frame.shape[1] and 0 <= cy < frame.shape[0]:
                    h, s, v = [int(t) for t in hsv[cy, cx]]
                    print(f'pixel ({cx},{cy})  BGR={frame[cy, cx].tolist()}  HSV=({h},{s},{v})')
                    cv2.circle(left, (cx, cy), 5, (0, 0, 255), 1, cv2.LINE_AA)
                    put_text(left, f'HSV=({h},{s},{v})', (cx + 8, cy - 8), (0, 0, 255), 0.5, 1)
                state['click'] = None
            both = np.hstack([left, right])
            cv2.imshow(WIN, cv2.resize(both, None, fx=args.scale, fy=args.scale))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key in KEYS and KEYS[key] != current:
                current = KEYS[key]
                set_bars(edited[current])
                print(f'切換到 {current}')
            elif key == ord('m'):
                state['view'] ^= 1
            elif key == ord('p'):
                print(f'{current}: {edited[current].to_dict()}')
            elif key == ord('s'):
                p = calib_io.save_hsv_range(calib_dir, current, edited[current])
                saved[current] = edited[current]
                print(f'已存 {current} -> {p}  {edited[current].to_dict()}')
            elif key == ord('a'):
                changed = [n for n, r in edited.items() if r != saved[n]]
                for name in changed:
                    p = calib_io.save_hsv_range(calib_dir, name, edited[name])
                    saved[name] = edited[name]
                print(f'已存有修改的顏色 {changed}' if changed else '沒有需要存的修改')
    finally:
        src.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
