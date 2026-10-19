#!/usr/bin/env python3
"""手眼相機 2 節點 arm2_vision_node（手臂 2 在 Home_C 待機觀測 C 區）。

職責
  * 訂閱 /arm2_cam/image_raw，持續偵測 C 站台四角（HSV），透視校正後切成 2x4 格
  * 每格以橘色像素比例判斷占用（門檻參數化，預設 0.3），並量測各占用格內物件的邊長
  * 以 arm2_homography 把各格中心（影像像素）轉成手臂 2 基座座標（m）

Services
  /vision/arm2/get_grid_state   occupied[8]、各格放置座標、all_full
  /vision/arm2/verify_place     指定格位是否已放置、已占用格位的邊長是否隨編號遞增、排錯的格位

Topic（service 保留不變）
  /vision/arm2/grid_state       vision_interfaces/GridState：每處理一張新影格發布一次，內容 = get_grid_state
                                + 各格比例/邊長 + verify_place 的排序檢查；影像中斷或 C 區定位失敗時 valid=false

格位編號：上排（靠近 B 區）1–4，下排 5–8，各排由左至右。校正後影像的「上排」要對應到
靠近 B 區那一排，由參數 warp_rotation_k（0..3）決定，請看 debug 視窗確認。
此節點不呼叫其他 service；仍用 MultiThreadedExecutor，讓 service 等待新結果時不擋住影像接收。
"""
from __future__ import annotations

import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_interfaces.msg import GridState
from vision_interfaces.srv import GetGridState, VerifyPlace

from vision_pkg import calib_io
from vision_pkg.debug_draw import draw_corner_ids, draw_poly, put_text
from vision_pkg.node_common import (FlatParams, FrameBuffer, LatestValue, image_msg_to_bgr,
                                    image_problem, read_ros_params, spin_node, stamp)
from vision_pkg.vision_core.color_zone import (MaskParams, ZoneFindParams, find_zone, hsv_mask,
                                               warp_to_plane)
from vision_pkg.vision_core.geometry import apply_homography, rotate_corners
from vision_pkg.vision_core.grid import GridAnalysis, GridParams, analyze_grid, cell_centers
from vision_pkg.vision_core.size_ranker import SizeRanker, make_ranker
from vision_pkg.vision_core.square_detect import SquareParams

CAM_NAME = calib_io.CAMERA_NAMES['arm2']
N_SLOTS = 8   # srv 的 occupied[8] / place_*[8] 為固定長度


@dataclass
class GridResult:
    valid: bool
    message: str = ''
    occupied: List[bool] = field(default_factory=list)
    ratios: List[float] = field(default_factory=list)
    sides_mm: List[Optional[float]] = field(default_factory=list)   # 已占用且可量測者才有值
    approx_slots: List[int] = field(default_factory=list)
    unmeasurable_slots: List[int] = field(default_factory=list)
    base_xy: Optional[np.ndarray] = None    # (8,2) 各格中心的手臂 2 基座座標 m；homography 未標定時為 None


class Arm2VisionNode(Node):
    def __init__(self):
        super().__init__('arm2_vision_node', automatically_declare_parameters_from_overrides=True)
        self.P = FlatParams(read_ros_params(self))
        self._last_flat = dict(self.P.flat)
        self._last_param_check = 0.0
        self.cfg = self._build_cfg(self.P)
        self.ranker: SizeRanker = self._make_ranker(self.cfg)
        self.show_debug = self.P.b('show_debug')
        self.calib_dir = calib_io.resolve_calib_dir(self.P.s('calib_dir'))
        self.get_logger().info(f'校正檔目錄: {self.calib_dir}')

        self.frames = FrameBuffer(maxlen=4)
        self.results = LatestValue()
        self._last_seq = 0
        self._corner_hist: deque = deque(maxlen=max(1, self.cfg.corner_median))
        self._last_good_t = 0.0
        self._debug_img: Optional[np.ndarray] = None

        period = self.cfg.hsv_reload_period_s
        self.hsv = calib_io.ReloadableConfig(lambda: calib_io.load_hsv_ranges(self.calib_dir),
                                             calib_io.hsv_paths(self.calib_dir), period)
        pkg = calib_io.package_config_dir()
        self.h_arm = calib_io.ReloadableConfig(
            lambda: calib_io.load_homography('arm2_homography', self.calib_dir),
            [self.calib_dir / 'arm2_homography.yaml', pkg / 'arm2_homography.yaml'], period)

        self._undist = None
        if self.P.b('undistort'):
            path = self.P.s('intrinsics_file') or str(calib_io.intrinsics_path(CAM_NAME, self.calib_dir))
            try:
                self._undist = calib_io.Undistorter(path)
                self.get_logger().info(f'已啟用去畸變: {path}')
            except Exception as e:
                self.get_logger().error(f'undistort=true 但讀不到內參 {path}: {e}；改用原始影像（與標定不一致時結果會有誤）')

        g_img = MutuallyExclusiveCallbackGroup()
        g_proc = MutuallyExclusiveCallbackGroup()
        g_srv = MutuallyExclusiveCallbackGroup()
        self.create_subscription(Image, self.P.s('image_topic'), self._on_image, qos_profile_sensor_data,
                                 callback_group=g_img)
        self.create_timer(1.0 / self.cfg.process_hz, self._on_tick, callback_group=g_proc)
        self.create_service(GetGridState, '/vision/arm2/get_grid_state', self._srv_grid_state, callback_group=g_srv)
        self.create_service(VerifyPlace, '/vision/arm2/verify_place', self._srv_verify, callback_group=g_srv)
        self.pub_grid = self.create_publisher(GridState, '/vision/arm2/grid_state', 10)
        self.get_logger().info(f'arm2_vision_node 就緒，訂閱 {self.P.s("image_topic")}')

    # ------------------------------------------------------------------ #
    def _build_cfg(self, P: FlatParams) -> SimpleNamespace:
        c = SimpleNamespace()
        c.process_hz = max(1.0, P.f('process_hz'))
        c.image_timeout_s = P.f('image_timeout_s')
        c.service_timeout_s = P.f('service_timeout_s')
        c.hsv_reload_period_s = P.f('hsv_reload_period_s')
        c.zone_key = P.s('c_zone_color')
        c.size = tuple(P.fl('c_station_size_mm'))
        if len(c.size) != 2 or min(c.size) <= 0:
            raise ValueError('c_station_size_mm 必須是兩個正數 [寬, 高]')
        c.ppm = P.f('warp_px_per_mm')
        c.rot_k = P.i('warp_rotation_k') % 4
        c.corner_median = max(1, P.i('corner_median_frames'))
        c.corner_hold_s = P.f('corner_hold_s')
        c.zone_mp = P.group(MaskParams, 'zone_mask')
        c.zone_find = P.group(ZoneFindParams, 'zone_find')
        c.orange_mp = P.group(MaskParams, 'orange_mask')
        c.grid = P.group(GridParams, 'grid')
        if c.grid.rows * c.grid.cols != N_SLOTS:
            raise ValueError(f'grid.rows * grid.cols 必須等於 {N_SLOTS}（srv 的 occupied[8]），目前 {c.grid.rows}x{c.grid.cols}')
        c.square = P.group(SquareParams, 'square')
        c.ranker_type = P.s('ranker.type')
        c.margin_px = P.f('ranker.confidence_margin_px')
        c.table = P.fl('ranker.table_mm')
        c.table_tol = P.f('ranker.table_tolerance_mm')
        c.station_height_mm = P.f('place.station_height_mm')
        c.object_height_mm = P.f('place.object_height_mm')
        return c

    @staticmethod
    def _make_ranker(c: SimpleNamespace) -> SizeRanker:
        return make_ranker(c.ranker_type, c.margin_px / c.ppm, c.table, c.table_tol)

    def _refresh_cfg(self, now: float) -> None:
        if now - self._last_param_check < 1.0:
            return
        self._last_param_check = now
        flat = read_ros_params(self)
        if flat == self._last_flat:
            return
        self._last_flat = flat
        try:
            cfg = self._build_cfg(FlatParams(flat))
            ranker = self._make_ranker(cfg)
        except Exception as e:
            self.get_logger().error(f'參數更新無效，沿用舊設定: {e}')
            return
        self.cfg, self.ranker = cfg, ranker
        self._corner_hist = deque(self._corner_hist, maxlen=max(1, cfg.corner_median))
        self.get_logger().info('參數已更新')

    # ------------------------------------------------------------------ #
    def _on_image(self, msg: Image) -> None:
        try:
            bgr = image_msg_to_bgr(msg)
            if self._undist is not None:
                bgr = self._undist.apply(bgr)
            self.frames.push(bgr)
        except Exception as e:
            self.get_logger().warning(f'影像轉換失敗: {e}', throttle_duration_sec=5.0)

    def _on_tick(self) -> None:
        try:
            self._tick()
        except Exception:
            self.get_logger().error('處理迴圈例外（已忽略，節點繼續運作）:\n' + traceback.format_exc(),
                                    throttle_duration_sec=5.0)

    def _tick(self) -> None:
        now = time.monotonic()
        self._refresh_cfg(now)
        if self.hsv.poll():
            self.get_logger().info('hsv.yaml 已重新載入')
        if self.h_arm.poll():
            self.get_logger().info(f'arm2_homography 已重新載入（valid={self.h_arm.value["valid"]}）')
        latest = self.frames.latest()
        problem = image_problem(latest, now, self.cfg.image_timeout_s)
        if problem:
            self._publish(GridResult(False, problem))   # 不更新 self.results，讓舊結果自然過期
            return
        seq, t_arr, frame = latest
        if seq == self._last_seq:
            return
        self._last_seq = seq

        c = self.cfg
        hsv = self.hsv.value
        if hsv is None or 'orange' not in hsv or c.zone_key not in hsv:
            self._set_result(GridResult(False, f'HSV 設定不可用（需要 orange 與 {c.zone_key}）: {self.hsv.error}'))
            return
        # 物件遮罩併入站台遮罩：放滿物件、貼近站台邊緣時才不會把站台外框咬掉一角
        objects = hsv_mask(frame, hsv['orange'], c.orange_mp)
        zr, _ = find_zone(frame, hsv[c.zone_key], c.zone_mp, c.zone_find, extra_mask=objects)
        if zr.found:
            self._corner_hist.append(zr.corners)
            self._last_good_t = now
        if not self._corner_hist or now - self._last_good_t > c.corner_hold_s:
            self._corner_hist.clear()
            self._set_result(GridResult(False, f'C 區定位失敗: {zr.reason or "尚無偵測結果"}'))
            self._render(frame, None, None, None, None)
            return

        corners = np.median(np.stack(list(self._corner_hist)), axis=0).astype(np.float32)
        rot = rotate_corners(corners, c.rot_k)
        warped, M = warp_to_plane(frame, rot, c.size[0], c.size[1], c.ppm)
        ga = analyze_grid(warped, hsv['orange'], c.orange_mp, c.square, c.grid)

        centers_px = apply_homography(np.linalg.inv(M), cell_centers(ga.rects))   # 各格中心在原影像的像素
        ha = self.h_arm.value
        base_xy = apply_homography(ha['H'], centers_px) if (ha is not None and ha['valid']) else None
        res = GridResult(True, 'ok', list(ga.occupied), list(ga.ratios),
                         [None if s is None else s / c.ppm for s in ga.sides_px],
                         list(ga.approx_slots), list(ga.unmeasurable_slots), base_xy)
        self._set_result(res)
        self._render(frame, rot, warped, ga, res)

    # ------------------------------------------------------------------ #
    def _set_result(self, res: GridResult) -> None:
        self.results.set(res)
        self._publish(res)

    def _publish(self, res: GridResult) -> None:
        m = GridState()
        m.valid, m.message = bool(res.valid), res.message
        if res.valid:
            m.occupied = [bool(o) for o in res.occupied]
            m.fill_ratio = [float(r) for r in res.ratios]
            m.side_mm = [float('nan') if s is None else float(s) for s in res.sides_mm]
            m.all_full = all(res.occupied)
            m.unmeasurable_slots = [int(s) for s in res.unmeasurable_slots]
            m.approx_slots = [int(s) for s in res.approx_slots]
            notes = ['占用比例: ' + ' '.join(f'{i + 1}:{r:.2f}' for i, r in enumerate(res.ratios))]
            if res.base_xy is not None:
                z = (self.cfg.station_height_mm + self.cfg.object_height_mm) / 1000.0
                m.place_valid = True
                m.place_x = [float(v) for v in res.base_xy[:, 0]]
                m.place_y = [float(v) for v in res.base_xy[:, 1]]
                m.place_z = [z] * N_SLOTS
            else:
                notes.append('arm2_homography 未標定，place_* 無效')
            if res.unmeasurable_slots:
                notes.append(f'格位 {res.unmeasurable_slots} 已占用但無法量測邊長，未做排序檢查')
            else:
                chk = self.ranker.check_order(res.sides_mm)
                m.order_checked, m.order_ok = True, bool(chk.order_ok)
                m.wrong_slots = [int(s) for s in chk.wrong_slots]
                if chk.message:
                    notes.append(chk.message)
            m.message = '；'.join(notes)
        self.pub_grid.publish(stamp(self, m, 'arm2_base'))   # place_* 是手臂 2 基座座標

    # ------------------------------------------------------------------ #
    def _get_result(self, resp):
        """取一個夠新且有效的結果；失敗時直接把 success=False 與原因填進 resp，並回傳 None。"""
        res, why = self.results.wait_fresh(self.cfg.image_timeout_s, self.cfg.service_timeout_s)
        if res is None:
            resp.success, resp.message = False, why
            return None
        if not res.valid:
            resp.success, resp.message = False, res.message
            return None
        return res

    def _fill_grid(self, out, res: GridResult) -> None:
        """把有效結果填進 GetGridState 回應。"""
        out.occupied = [bool(o) for o in res.occupied]
        out.all_full = all(res.occupied)
        if res.base_xy is None:
            out.success = False
            out.message = ('arm2_homography 尚未標定（valid=false 或讀取失敗），無法提供放置座標；'
                           '請執行 ros2 run vision_pkg calibrate_homography --target arm2')
            return
        z = (self.cfg.station_height_mm + self.cfg.object_height_mm) / 1000.0
        out.place_x = [float(v) for v in res.base_xy[:, 0]]
        out.place_y = [float(v) for v in res.base_xy[:, 1]]
        out.place_z = [z] * N_SLOTS
        out.success = True
        out.message = '占用比例: ' + ' '.join(f'{i + 1}:{r:.2f}' for i, r in enumerate(res.ratios))

    @staticmethod
    def _grid_defaults(out) -> None:
        out.occupied = [False] * N_SLOTS
        out.place_x = [0.0] * N_SLOTS
        out.place_y = [0.0] * N_SLOTS
        out.place_z = [0.0] * N_SLOTS
        out.all_full = False

    def _srv_grid_state(self, req, resp):
        self._grid_defaults(resp)
        try:
            res = self._get_result(resp)
            if res is not None:
                self._fill_grid(resp, res)
        except Exception as e:
            self.get_logger().error('get_grid_state 例外:\n' + traceback.format_exc())
            resp.success, resp.message = False, f'內部錯誤: {e!r}'
        return resp

    def _srv_verify(self, req, resp):
        resp.placed, resp.order_ok, resp.wrong_slots = False, False, []
        try:
            if not 1 <= req.slot <= N_SLOTS:
                resp.success, resp.message = False, f'格位編號必須在 1..{N_SLOTS}（收到 {req.slot}）'
                return resp
            res = self._get_result(resp)
            if res is None:
                return resp
            resp.placed = bool(res.occupied[req.slot - 1])
            notes = [f'格位 {req.slot} 占用比例 {res.ratios[req.slot - 1]:.2f}'
                     f'（{"已放置" if resp.placed else "未放置"}）']
            if res.unmeasurable_slots:
                resp.success = False
                resp.message = ('；'.join(notes) + f'；格位 {res.unmeasurable_slots} 已占用但無法量測邊長'
                                '（物件相連 / 被截斷 / 找不到區塊），無法驗證排序，不猜測')
                return resp
            chk = self.ranker.check_order(res.sides_mm)
            resp.order_ok = bool(chk.order_ok)
            resp.wrong_slots = [int(s) for s in chk.wrong_slots]
            if chk.message:
                notes.append(chk.message)
            if res.approx_slots:
                notes.append(f'格位 {res.approx_slots} 形狀不是標準正方形，邊長為近似值')
            resp.success = True
            resp.message = '；'.join(notes)
        except Exception as e:
            self.get_logger().error('verify_place 例外:\n' + traceback.format_exc())
            resp.success, resp.message = False, f'內部錯誤: {e!r}'
        return resp

    # ------------------------------------------------------------------ #
    def get_debug_image(self) -> Optional[np.ndarray]:
        return self._debug_img

    def _render(self, frame, rot_corners, warped, ga: Optional[GridAnalysis], res: Optional[GridResult]) -> None:
        if not self.show_debug:
            return
        img = frame.copy()
        if rot_corners is None:
            put_text(img, 'C ZONE NOT FOUND', (8, 18), (0, 0, 255), 0.6, 2)
            self._debug_img = img
            return
        draw_poly(img, rot_corners, (0, 255, 255), 2)
        draw_corner_ids(img, rot_corners)   # 0 = 校正後影像左上角；上排應是靠近 B 區那一排
        put_text(img, f'corner 0 = origin (warp_rotation_k={self.cfg.rot_k})', (8, 18), (0, 255, 255), 0.5, 1)
        w = warped.copy()
        for b in ga.blobs:   # 物件的最小外接矩形：綠 = 合格正方形，橘 = 形狀不合（量測僅供參考）
            draw_poly(w, b.box, (0, 255, 0) if b.is_square else (0, 165, 255), 2)
        for slot, (x0, y0, x1, y1) in enumerate(ga.rects, start=1):
            occ = ga.occupied[slot - 1]
            col = (0, 0, 255) if occ else (200, 200, 200)
            cv2.rectangle(w, (x0, y0), (x1 - 1, y1 - 1), col, 2)
            put_text(w, str(slot), (x0 + 6, y0 + 24), col, 0.8, 2)
            put_text(w, f'{ga.ratios[slot - 1] * 100:.0f}%', (x0 + 6, y1 - 8), col, 0.5, 1)
            side = res.sides_mm[slot - 1]
            if side is not None:
                put_text(w, f'{side:.1f}mm' + ('~' if slot in res.approx_slots else ''), (x0 + 6, y0 + 46), col, 0.5, 1)
            elif slot in res.unmeasurable_slots:
                put_text(w, 'N/A', (x0 + 6, y0 + 46), (0, 0, 255), 0.5, 1)
        w = cv2.resize(w, (img.shape[1], max(1, int(w.shape[0] * img.shape[1] / w.shape[1]))))
        put_text(w, 'TOP ROW = near B (slots 1-4)', (8, 16), (0, 255, 255), 0.5, 1)
        self._debug_img = np.vstack([img, w])


def main(args=None):
    rclpy.init(args=args)
    node = None
    code = 0
    try:
        node = Arm2VisionNode()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        spin_node(rclpy, node, executor, node.show_debug, 'arm2_vision')
    except (KeyError, ValueError) as e:
        print(f'[arm2_vision_node] 啟動失敗：{e}', file=sys.stderr)
        code = 2
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        try:
            if node is not None:
                node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
