#!/usr/bin/env python3
"""手眼相機 2 節點 arm2_vision_node（手臂 2 在 Home_C 待機觀測 C 區）。

職責
  * 訂閱 /arm2_cam/image_raw，持續在粉紅場地上找出 C 區 8 個方孔，依固定編號排序
  * 每個孔以藍色（瓶蓋）像素比例判斷是否已放置（門檻參數化）
  * 以 arm2_homography 把各孔中心（影像像素）轉成手臂 2 基座座標（m）

Services
  /vision/arm2/get_grid_state   occupied[8]、各格放置座標、all_full
  /vision/arm2/verify_place     指定格位是否已放置；order_ok = 已放置的格位是否為 1..n 連續（依序放置）；
                                wrong_slots = 跳號放置的格位

Topic
  /vision/arm2/grid_state       vision_interfaces/GridState：每處理一張新影格發布一次，內容 = get_grid_state
                                + 各孔比例 + verify_place 的順序檢查；影像中斷或孔位定位失敗時 valid=false

格位編號（固定）：使用者站在場地長邊前、B 區在右手邊，最下排左 1、右 2，往上 3、4，… 最上排 7、8。
手眼相機影像的方向由參數 view_rotation_k（0..3）對應，請看 debug 視窗確認 1 號在左下。
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
from vision_pkg.debug_draw import draw_poly, put_text, put_text_centered
from vision_pkg.node_common import (FlatParams, FrameBuffer, LatestValue, image_msg_to_bgr,
                                    image_problem, read_ros_params, spin_node, stamp)
from vision_pkg.vision_core.color_zone import MaskParams, to_hsv, mask_from_hsv
from vision_pkg.vision_core.geometry import apply_homography
from vision_pkg.vision_core.holes import (N_SLOTS, HoleLayout, HoleParams, check_fill_order, locate_holes,
                                          slot_ratios)

CAM_NAME = calib_io.CAMERA_NAMES['arm2']


@dataclass
class GridResult:
    valid: bool
    message: str = ''
    occupied: List[bool] = field(default_factory=list)
    ratios: List[float] = field(default_factory=list)
    holes: Optional[np.ndarray] = None      # (8,4,2) 各孔角點（影像像素），依格位 1..8
    base_xy: Optional[np.ndarray] = None    # (8,2) 各孔中心的手臂 2 基座座標 m；homography 未標定時為 None
    held: bool = False                      # 本張影像沒找到 8 個孔，沿用最近的孔位


class Arm2VisionNode(Node):
    def __init__(self):
        super().__init__('arm2_vision_node', automatically_declare_parameters_from_overrides=True)
        self.P = FlatParams(read_ros_params(self))
        self._last_flat = dict(self.P.flat)
        self._last_param_check = 0.0
        self.cfg = self._build_cfg(self.P)
        self.show_debug = self.P.b('show_debug')
        self.calib_dir = calib_io.resolve_calib_dir(self.P.s('calib_dir'))
        self.get_logger().info(f'校正檔目錄: {self.calib_dir}')

        self.frames = FrameBuffer(maxlen=4)
        self.results = LatestValue()
        self._last_seq = 0
        self._hole_hist: deque = deque(maxlen=max(1, self.cfg.hole_median))
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
        c.field_color = P.s('field_color')
        c.field_mp = P.group(MaskParams, 'field_mask')
        c.rot_k = P.i('view_rotation_k') % 4
        c.hole_median = max(1, P.i('hole_median_frames'))
        c.hole_hold_s = P.f('hole_hold_s')
        c.hole = P.group(HoleParams, 'hole')
        if not 0.0 <= c.hole.margin_ratio < 0.5:
            raise ValueError(f'hole.margin_ratio 必須在 [0, 0.5)（目前 {c.hole.margin_ratio}）')
        c.obj_color = P.s('object_color')
        c.obj_mp = P.group(MaskParams, 'object_mask')
        c.station_height_mm = P.f('place.station_height_mm')
        c.object_height_mm = P.f('place.object_height_mm')
        return c

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
        except Exception as e:
            self.get_logger().error(f'參數更新無效，沿用舊設定: {e}')
            return
        self.cfg = cfg
        self._hole_hist = deque(maxlen=max(1, cfg.hole_median))   # 參數（例如 view_rotation_k）改了，舊孔位作廢
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
        if hsv is None or c.obj_color not in hsv or c.field_color not in hsv:
            self._set_result(GridResult(False, f'HSV 設定不可用（需要 {c.field_color} 與 {c.obj_color}）: {self.hsv.error}'))
            return
        field_mask = mask_from_hsv(to_hsv(frame, c.field_mp.blur_ksize), hsv[c.field_color], c.field_mp)
        layout = locate_holes(field_mask, c.hole, c.rot_k)
        if layout.ok:
            self._hole_hist.append(np.stack([h.corners for h in layout.holes]))
            self._last_good_t = now
        if not self._hole_hist or now - self._last_good_t > c.hole_hold_s:
            self._hole_hist.clear()
            self._set_result(GridResult(False, f'C 區方孔定位失敗: {layout.message}'))
            self._render(frame, layout, None)
            return

        # 孔中心取最近幾次的中位數；孔的形狀用最近一次的（角點順序在不同影格可能輪換，不能逐點取中位數）
        last = self._hole_hist[-1]
        med_ctr = np.median(np.stack([h.mean(axis=1) for h in self._hole_hist]), axis=0)
        holes = (last - last.mean(axis=1, keepdims=True) + med_ctr[:, None, :]).astype(np.float32)   # (8,4,2)
        obj_mask = mask_from_hsv(to_hsv(frame, c.obj_mp.blur_ksize), hsv[c.obj_color], c.obj_mp)
        ratios = slot_ratios(obj_mask, list(holes), c.hole.margin_ratio)
        occupied = [r >= c.hole.occupied_threshold for r in ratios]
        centers = holes.mean(axis=1)
        ha = self.h_arm.value
        base_xy = apply_homography(ha['H'], centers) if (ha is not None and ha['valid']) else None
        res = GridResult(True, 'ok', occupied, ratios, holes, base_xy, held=not layout.ok)
        self._set_result(res)
        self._render(frame, layout, res)

    # ------------------------------------------------------------------ #
    def _set_result(self, res: GridResult) -> None:
        self.results.set(res)
        self._publish(res)

    @staticmethod
    def _ratio_text(res: GridResult) -> str:
        t = '占用比例: ' + ' '.join(f'{i + 1}:{r:.2f}' for i, r in enumerate(res.ratios))
        return t + ('（本張影像孔位被遮住，沿用最近的孔位）' if res.held else '')

    def _publish(self, res: GridResult) -> None:
        m = GridState()
        m.valid, m.message = bool(res.valid), res.message
        if res.valid:
            m.occupied = [bool(o) for o in res.occupied]
            m.fill_ratio = [float(r) for r in res.ratios]
            m.side_mm = [float('nan')] * N_SLOTS           # 瓶蓋大小相同，不量尺寸
            m.all_full = all(res.occupied)
            notes = [self._ratio_text(res)]
            if res.base_xy is not None:
                z = (self.cfg.station_height_mm + self.cfg.object_height_mm) / 1000.0
                m.place_valid = True
                m.place_x = [float(v) for v in res.base_xy[:, 0]]
                m.place_y = [float(v) for v in res.base_xy[:, 1]]
                m.place_z = [z] * N_SLOTS
            else:
                notes.append('arm2_homography 未標定，place_* 無效')
            chk = check_fill_order(res.occupied)
            m.order_checked, m.order_ok = True, bool(chk.order_ok)
            m.wrong_slots = [int(s) for s in chk.wrong_slots]
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
        out.message = self._ratio_text(res)

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
            chk = check_fill_order(res.occupied)
            resp.order_ok = bool(chk.order_ok)
            resp.wrong_slots = [int(s) for s in chk.wrong_slots]
            resp.success = True
            resp.message = (f'格位 {req.slot} 占用比例 {res.ratios[req.slot - 1]:.2f}'
                            f'（{"已放置" if resp.placed else "未放置"}）；{chk.message}')
        except Exception as e:
            self.get_logger().error('verify_place 例外:\n' + traceback.format_exc())
            resp.success, resp.message = False, f'內部錯誤: {e!r}'
        return resp

    # ------------------------------------------------------------------ #
    def get_debug_image(self) -> Optional[np.ndarray]:
        return self._debug_img

    def _render(self, frame, layout: HoleLayout, res: Optional[GridResult]) -> None:
        if not self.show_debug:
            return
        img = frame.copy()
        for h in layout.candidates:     # 不合格的候選（紅色細框）
            if not h.ok:
                draw_poly(img, h.corners, (0, 0, 255), 1)
        if res is None:
            put_text(img, 'C HOLES NOT FOUND: ' + layout.message[:60], (8, 18), (0, 0, 255), 0.5, 1)
            self._debug_img = img
            return
        for slot, q in enumerate(res.holes, start=1):
            occ = res.occupied[slot - 1]
            col = (0, 0, 255) if occ else (0, 255, 0)
            draw_poly(img, q, col, 2)
            ctr = q.mean(axis=0)
            half = max(8.0, 0.5 * float(np.linalg.norm(q[0] - q[3])))     # 孔高的一半（影像像素）
            put_text(img, str(slot), (q[0][0] + 3, q[0][1] + 14), col, 0.45, 1)
            # 放置座標（手臂 2 基座 m）；未標定時顯示孔中心像素座標。x、y 分兩行寫在孔中央，相鄰的孔才不會疊在一起
            if res.base_xy is not None:
                x, y = res.base_xy[slot - 1]
                lines = (f'{x:.3f}', f'{y:.3f}')
            else:
                lines = (f'{ctr[0]:.0f}', f'{ctr[1]:.0f}')
            put_text_centered(img, lines[0], (ctr[0], ctr[1] - 6), col, 0.35, 1)
            put_text_centered(img, lines[1], (ctr[0], ctr[1] + 8), col, 0.35, 1)
            put_text_centered(img, f'{res.ratios[slot - 1] * 100:.0f}%', (ctr[0], ctr[1] + half + 9), col, 0.35, 1)
        put_text(img, f'slot 1 must be bottom-left in user view (view_rotation_k={self.cfg.rot_k})'
                      + ('  [held]' if res.held else ''), (8, 18), (0, 255, 255), 0.5, 1)
        put_text(img, 'in each hole: x / y = ' + ('base m' if res.base_xy is not None else 'pixel (arm2_homography not calibrated)'),
                 (8, 36), (0, 255, 255), 0.5, 1)
        self._debug_img = img


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
