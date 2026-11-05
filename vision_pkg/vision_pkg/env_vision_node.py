#!/usr/bin/env python3
"""環境相機節點 env_vision_node。

場地：60x30 cm 粉紅色泡棉板，由左至右 C | B | A 各 20x30 cm；物件：藍色圓形瓶蓋（直徑 20 mm，大小相同）。

職責
  * 訂閱 /env_cam/image_raw（只訂閱 topic，絕不自己開 /dev/video）
  * 開機（場地淨空）時找出粉紅色場地、均分成三區，角點存入 zones.yaml；失敗則讀上次存檔並警告
  * 持續判斷：各區有無瓶蓋、B 區是否有手臂、C 區瓶蓋數；存成「最新狀態」，service 被呼叫時直接回傳

Services（皆有逾時處理，失敗時 success=false 並附明確 message）
  /vision/env/recalibrate_zones   重新定位三區
  /vision/env/get_zone_state      a_has_object / b_has_object / b_arm_clear / c_object_count
  /vision/env/capture_ranking     此場地不排序：只清點 A 區瓶蓋數（count），保留介面相容
  /vision/env/get_rank            此場地不排序：固定回 success=true、rank=0（保留介面相容）
  /vision/env/clear_taken         此場地不排序：無作用，固定回 success=true

Topics
  /vision/env/zone_state   vision_interfaces/ZoneState        每處理一張新影格發布一次；影像中斷時 valid=false
  /vision/env/ranking      vision_interfaces/RankingSnapshot  transient_local；固定 valid=false（不排序）

執行緒模型：MultiThreadedExecutor；影像訂閱、處理計時器、service 伺服端各放在不同的
MutuallyExclusiveCallbackGroup，service 等新影像時不會擋住影像接收。
"""
from __future__ import annotations

import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from vision_interfaces.msg import RankingSnapshot
from vision_interfaces.msg import ZoneState as ZoneStateMsg
from vision_interfaces.srv import CaptureRanking, GetRank, GetZoneState, RecalibrateZones

from vision_pkg import calib_io
from vision_pkg.debug_draw import draw_poly, put_text
from vision_pkg.node_common import (Debouncer, FlatParams, FrameBuffer, LatestValue, image_msg_to_bgr,
                                    image_problem, read_ros_params, spin_node, stamp)
from vision_pkg.vision_core.color_zone import (FieldParams, MaskParams, OccupancyParams, ZoneFindParams, ZoneRoi,
                                               count_blobs, hsv_mask, judge_arm_present, judge_object_present,
                                               locate_field_zones, make_roi, measure_zones)
from vision_pkg.vision_core.geometry import polygon_area

ZONES = ('a', 'b', 'c')
LEFT_TO_RIGHT = ('c', 'b', 'a')     # 目前場地在影像中由左至右的順序（field.reverse=false）
ZONE_BGR = {'a': (255, 128, 0), 'b': (0, 200, 0), 'c': (0, 220, 255)}
CAM_NAME = calib_io.CAMERA_NAMES['env']
NO_RANKING = '此場地瓶蓋大小相同，不排序；放置順序由 C 區格位編號 1..8 決定'


@dataclass
class ZoneInfo:
    corners: np.ndarray     # (4,2) tl,tr,br,bl（影像像素）
    baseline: float         # 淨空時場地色佔 ROI 的比例
    source: str             # 'detected' | 'loaded'


@dataclass
class ZoneState:
    """某一次處理的結果（valid=False 時其餘欄位無意義，只看 message）。"""
    valid: bool
    message: str = ''
    a_has_object: bool = False
    b_has_object: bool = False
    b_arm_clear: bool = False
    c_object_count: int = 0
    detail: str = ''


class EnvVisionNode(Node):
    def __init__(self):
        super().__init__('env_vision_node', automatically_declare_parameters_from_overrides=True)
        self._lock = threading.RLock()

        # ---- 參數（全部來自 config/env_vision.yaml；缺少會在這裡一次報錯） ----
        self.P = FlatParams(read_ros_params(self))
        self._last_flat = dict(self.P.flat)
        self._last_param_check = 0.0
        self.cfg = self._build_cfg(self.P)
        self.show_debug = self.P.b('show_debug')
        self.calib_dir = calib_io.resolve_calib_dir(self.P.s('calib_dir'))
        self.get_logger().info(f'校正檔目錄: {self.calib_dir}')

        # ---- 狀態 ----
        self.frames = FrameBuffer(maxlen=max(8, self.cfg.boot_frames + 2, self.cfg.avg_frames + 2))
        self.state = LatestValue()
        self._zones: Dict[str, ZoneInfo] = {}
        self._rois: Dict[str, ZoneRoi] = {}
        self._rois_key = None
        self._boot_done = False
        self._t_start = time.monotonic()
        self._last_seq = 0
        self._debug_img: Optional[np.ndarray] = None
        self._c_counts: deque = deque(maxlen=max(1, self.cfg.count_median))
        self._make_debouncers()

        self.hsv = calib_io.ReloadableConfig(lambda: calib_io.load_hsv_ranges(self.calib_dir),
                                             calib_io.hsv_paths(self.calib_dir), self.cfg.hsv_reload_period_s)
        if self.hsv.value is None:
            self.get_logger().error(f'HSV 設定載入失敗: {self.hsv.error}')

        self._undist = None
        if self.P.b('undistort'):
            path = self.P.s('intrinsics_file') or str(calib_io.intrinsics_path(CAM_NAME, self.calib_dir))
            try:
                self._undist = calib_io.Undistorter(path)
                self.get_logger().info(f'已啟用去畸變: {path}')
            except Exception as e:
                self.get_logger().error(f'undistort=true 但讀不到內參 {path}: {e}；改用原始影像（與標定不一致時結果會有誤）')

        # ---- ROS 介面 ----
        g_img = MutuallyExclusiveCallbackGroup()
        g_proc = MutuallyExclusiveCallbackGroup()
        g_srv = MutuallyExclusiveCallbackGroup()
        self.create_subscription(Image, self.P.s('image_topic'), self._on_image, qos_profile_sensor_data,
                                 callback_group=g_img)
        self.create_timer(1.0 / self.cfg.process_hz, self._on_tick, callback_group=g_proc)
        self.create_service(RecalibrateZones, '/vision/env/recalibrate_zones', self._srv_recalibrate,
                            callback_group=g_srv)
        self.create_service(GetZoneState, '/vision/env/get_zone_state', self._srv_zone_state, callback_group=g_srv)
        self.create_service(CaptureRanking, '/vision/env/capture_ranking', self._srv_capture, callback_group=g_srv)
        self.create_service(GetRank, '/vision/env/get_rank', self._srv_get_rank, callback_group=g_srv)
        self.create_service(Trigger, '/vision/env/clear_taken', self._srv_clear_taken, callback_group=g_srv)
        self.pub_state = self.create_publisher(ZoneStateMsg, '/vision/env/zone_state', 10)
        self.pub_ranking = self.create_publisher(
            RankingSnapshot, '/vision/env/ranking',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._publish_ranking()
        self.get_logger().info(f'env_vision_node 就緒，訂閱 {self.P.s("image_topic")}，等待影像與場地定位…')

    # ------------------------------------------------------------------ #
    # 參數 / 設定
    # ------------------------------------------------------------------ #
    def _build_cfg(self, P: FlatParams) -> SimpleNamespace:
        c = SimpleNamespace()
        c.process_hz = max(1.0, P.f('process_hz'))
        c.image_timeout_s = P.f('image_timeout_s')
        c.service_timeout_s = P.f('service_timeout_s')
        c.hsv_reload_period_s = P.f('hsv_reload_period_s')
        c.boot_frames = max(1, P.i('boot_locate_frames'))
        c.boot_min_det = max(1, P.i('boot_min_detections'))
        c.boot_wait_s = P.f('boot_wait_s')
        c.min_baseline = P.f('min_baseline_ratio')
        c.erode = P.i('roi_erode_px')
        c.field = P.group(FieldParams, 'field')
        c.field_size = tuple(P.fl('field.size_mm'))
        if len(c.field_size) != 2 or min(c.field_size) <= 0:
            raise ValueError('field.size_mm 必須是兩個正數 [寬, 高]')
        c.zone_size = {z: (c.field_size[0] / len(ZONES), c.field_size[1]) for z in ZONES}   # 均分
        c.field_mp = P.group(MaskParams, 'field_mask')
        c.zone_mp = P.group(MaskParams, 'zone_mask')
        c.zone_find = P.group(ZoneFindParams, 'zone_find')
        c.obj_color = P.s('object_color')
        c.obj_mp = P.group(MaskParams, 'object_mask')
        c.black_mp = P.group(MaskParams, 'black_mask')
        c.occ = P.group(OccupancyParams, 'occupancy')
        c.obj_frames = P.i('debounce.object_frames')
        c.arm_present_frames = P.i('debounce.arm_present_frames')
        c.arm_clear_frames = P.i('debounce.arm_clear_frames')
        c.count_min_mm2 = P.f('object_count.min_blob_area_mm2')
        c.count_median = P.i('object_count.median_frames')
        c.avg_frames = max(1, P.i('capture.avg_frames'))
        c.expected = P.i('capture.expected_count')
        return c

    def _make_debouncers(self) -> None:
        c = self.cfg
        self._deb_a = Debouncer(c.obj_frames, c.obj_frames)
        self._deb_b = Debouncer(c.obj_frames, c.obj_frames)
        # 手臂：偵測到立刻採信；離開需連續確認。初始視為「在場」（確認 clear 前不放行，安全優先）
        self._deb_arm = Debouncer(c.arm_present_frames, c.arm_clear_frames, initial=True)

    def _refresh_cfg(self, now: float) -> None:
        """約每秒檢查一次參數是否被 ros2 param set 改過；改壞了就保留舊設定。"""
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
        self._make_debouncers()
        self._rois_key = None
        self.get_logger().info('參數已更新')

    def _colors(self) -> Tuple[Optional[dict], str]:
        """場地色 / 物件色 / 黑色 的 HSV 範圍；hsv.yaml 缺顏色時回傳原因。"""
        hsv = self.hsv.value
        if hsv is None:
            return None, f'HSV 設定載入失敗: {self.hsv.error}'
        try:
            return {'field': hsv[self.cfg.field.color], 'obj': hsv[self.cfg.obj_color], 'black': hsv['black']}, ''
        except (KeyError, ValueError) as e:
            return None, f'hsv.yaml 缺少顏色 {e}'

    # ------------------------------------------------------------------ #
    # 影像接收（只存最新影格，處理交給計時器）
    # ------------------------------------------------------------------ #
    def _on_image(self, msg: Image) -> None:
        try:
            bgr = image_msg_to_bgr(msg)
            if self._undist is not None:
                bgr = self._undist.apply(bgr)
            self.frames.push(bgr)
        except Exception as e:
            self.get_logger().warning(f'影像轉換失敗: {e}', throttle_duration_sec=5.0)

    # ------------------------------------------------------------------ #
    # 場地定位
    # ------------------------------------------------------------------ #
    def _locate_zones(self, frames: List[np.ndarray], source: str) -> Tuple[bool, str]:
        """用多張影格定位場地並均分三區；失敗的區沿用現有/上次存檔值並警告。回傳 (三區皆成功, 說明)。"""
        colors, why = self._colors()
        if colors is None:
            return False, why
        c = self.cfg
        res = locate_field_zones(frames, colors['field'], LEFT_TO_RIGHT, c.field_mp, c.zone_mp, c.zone_find,
                                 c.field, c.field_size, c.boot_min_det, c.erode, c.min_baseline)
        saved = None
        msgs, all_ok, any_new = [], True, False
        with self._lock:
            for z in ZONES:
                r = res[z]
                if r.ok:
                    self._zones[z] = ZoneInfo(r.corners, r.baseline, 'detected')
                    any_new = True
                    msgs.append(f'{z.upper()} 區定位成功（{r.message}）')
                    continue
                all_ok = False
                if z in self._zones:
                    self.get_logger().warning(f'[{source}] {z.upper()} 區重新定位失敗，沿用目前的 ROI: {r.message}')
                    msgs.append(f'{z.upper()} 區失敗（沿用舊 ROI）: {r.message}')
                    continue
                if saved is None:
                    try:
                        saved = calib_io.load_zones(self.calib_dir) or {}
                    except Exception as e:
                        self.get_logger().error(f'讀取 zones.yaml 失敗: {e}')
                        saved = {}
                if z in saved:
                    self._zones[z] = ZoneInfo(saved[z]['corners'], saved[z]['baseline'], 'loaded')
                    self.get_logger().warning(f'[{source}] {z.upper()} 區偵測失敗（{r.message}），改用上次存檔')
                    msgs.append(f'{z.upper()} 區失敗，改用上次存檔: {r.message}')
                else:
                    self.get_logger().error(f'[{source}] {z.upper()} 區偵測失敗且沒有上次存檔: {r.message}')
                    msgs.append(f'{z.upper()} 區失敗且無存檔: {r.message}')
            if any_new:
                try:
                    h, w = frames[0].shape[:2]
                    calib_io.save_zones(self.calib_dir, (w, h),
                                        {z: {'corners': i.corners, 'baseline': i.baseline}
                                         for z, i in self._zones.items()})
                except Exception as e:
                    self.get_logger().warning(f'zones.yaml 寫入失敗（本次仍可使用）: {e}')
            self._rois_key = None
        self._make_debouncers()
        return all_ok, '；'.join(msgs)

    def _boot_step(self, now: float) -> None:
        """開機定位：影像夠多就定位；等不到影像就讀上次存檔。"""
        n = min(self.cfg.boot_frames, self.frames.maxlen)
        got = self.frames.latest_n(n)
        if len(got) >= n:
            ok, msg = self._locate_zones([f for _, _, f in got], 'boot')
            (self.get_logger().info if ok else self.get_logger().warning)(f'開機定位: {msg}')
            self._boot_done = True
        elif now - self._t_start > self.cfg.boot_wait_s:
            try:
                saved = calib_io.load_zones(self.calib_dir) or {}
            except Exception as e:
                self.get_logger().error(f'讀取 zones.yaml 失敗: {e}')
                saved = {}
            with self._lock:
                for z, d in saved.items():
                    if z in ZONES:
                        self._zones[z] = ZoneInfo(d['corners'], d['baseline'], 'loaded')
            self.get_logger().warning(f'等不到足夠影像（{self.cfg.boot_wait_s:.0f}s），改用上次存檔的場地（{len(saved)} 區）')
            self._boot_done = True

    def _ensure_rois(self, shape) -> Optional[Dict[str, ZoneRoi]]:
        with self._lock:
            zones = dict(self._zones)
        if not all(z in zones for z in ZONES):
            return None
        key = (shape[:2], self.cfg.erode, tuple(zones[z].corners.tobytes() for z in ZONES))
        if key != self._rois_key:
            self._rois = {z: make_roi(shape, zones[z].corners, self.cfg.erode) for z in ZONES}
            self._rois_key = key
        return self._rois

    def _count_in(self, obj_mask: np.ndarray, zone: str, rois: Dict[str, ZoneRoi], corners: np.ndarray) -> int:
        """某區內的瓶蓋數（藍色塊數）。面積門檻由 mm² 換算成像素²（以該區四邊形的平均比例）。"""
        w_mm, h_mm = self.cfg.zone_size[zone]
        mm2_per_px = (w_mm * h_mm) / max(1.0, polygon_area(corners))
        return count_blobs(cv2.bitwise_and(obj_mask, rois[zone].mask), self.cfg.count_min_mm2 / mm2_per_px)

    # ------------------------------------------------------------------ #
    # 持續處理
    # ------------------------------------------------------------------ #
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
        if not self._boot_done:
            self._boot_step(now)
            if not self._boot_done:
                return
        latest = self.frames.latest()
        problem = image_problem(latest, now, self.cfg.image_timeout_s)
        if problem:
            # 不更新 self.state（讓舊結果自然過期），但 topic 要明確告知訂閱者目前沒有可信結果
            self._publish_state(ZoneState(False, problem))
            return
        seq, t_arr, frame = latest
        if seq == self._last_seq:
            return  # 沒有新影格
        self._last_seq = seq

        colors, why = self._colors()
        rois = self._ensure_rois(frame.shape)
        if colors is None:
            self._set_state(ZoneState(False, why))
            return
        if rois is None:
            self._set_state(ZoneState(False, '場地尚未定位（請在場地淨空時呼叫 /vision/env/recalibrate_zones）'))
            self._render(frame, None, None, None)
            return

        c = self.cfg
        with self._lock:
            zones = dict(self._zones)
        metrics, obj_mask = measure_zones(frame, rois, colors['field'], colors['obj'], colors['black'],
                                          c.zone_mp, c.obj_mp, c.black_mp)
        a_raw, a_info = judge_object_present(metrics['a'], zones['a'].baseline, c.occ)
        b_raw, b_info = judge_object_present(metrics['b'], zones['b'].baseline, c.occ)
        arm_raw, arm_info = judge_arm_present(metrics['b'], zones['b'].baseline, c.occ)
        self._c_counts.append(self._count_in(obj_mask, 'c', rois, zones['c'].corners))

        prev_arm = self._deb_arm.state
        a_obj = self._deb_a.update(a_raw)
        b_obj = self._deb_b.update(b_raw)
        arm = self._deb_arm.update(arm_raw)
        if prev_arm is not None and prev_arm != arm:
            self.get_logger().info('B 區偵測到手臂進入' if arm else 'B 區手臂已離開（連續確認 clear）')
        c_cnt = int(np.median(list(self._c_counts)))
        detail = f'A[{a_info}] B[{b_info}; {arm_info}] C[n={c_cnt}]'
        st = ZoneState(True, 'ok', bool(a_obj), bool(b_obj), not bool(arm), c_cnt, detail)
        self._set_state(st)
        self._render(frame, st, obj_mask, rois)

    # ------------------------------------------------------------------ #
    # Topics
    # ------------------------------------------------------------------ #
    def _set_state(self, st: ZoneState) -> None:
        self.state.set(st)
        self._publish_state(st)

    def _publish_state(self, st: ZoneState) -> None:
        m = ZoneStateMsg()
        m.valid = bool(st.valid)
        m.message = st.detail if st.valid else st.message
        m.a_has_object, m.b_has_object = bool(st.a_has_object), bool(st.b_has_object)
        m.b_arm_clear = bool(st.valid and st.b_arm_clear)   # 無效時一律保守回報「手臂還在」
        m.c_object_count = int(st.c_object_count)
        self.pub_state.publish(stamp(self, m, CAM_NAME))

    def _publish_ranking(self) -> None:
        """此場地不排序：ranking topic 只發布一次 valid=false 與原因（transient_local，晚訂閱也收得到）。"""
        m = RankingSnapshot()
        m.valid, m.message = False, NO_RANKING
        self.pub_ranking.publish(stamp(self, m, CAM_NAME))

    # ------------------------------------------------------------------ #
    # Services
    # ------------------------------------------------------------------ #
    def _srv_recalibrate(self, req, resp):
        try:
            self.hsv.poll(force=True)
            n = min(self.cfg.boot_frames, self.frames.maxlen)
            frames = self.frames.wait_fresh(n, self.cfg.service_timeout_s)
            if len(frames) < n:
                resp.success = False
                resp.message = f'逾時：{self.cfg.service_timeout_s:.1f}s 內只收到 {len(frames)}/{n} 張新影像（影像中斷？）'
                return resp
            resp.success, resp.message = self._locate_zones(frames, 'recalibrate')
        except Exception as e:
            self.get_logger().error('recalibrate_zones 例外:\n' + traceback.format_exc())
            resp.success, resp.message = False, f'內部錯誤: {e!r}'
        return resp

    def _srv_zone_state(self, req, resp):
        resp.a_has_object = resp.b_has_object = False
        resp.b_arm_clear = False          # 無法判斷時保守回報「手臂還在」
        resp.c_object_count = 0
        try:
            st, why = self.state.wait_fresh(self.cfg.image_timeout_s, self.cfg.service_timeout_s)
            if st is None:
                resp.success, resp.message = False, why
            elif not st.valid:
                resp.success, resp.message = False, st.message
            else:
                resp.a_has_object, resp.b_has_object = st.a_has_object, st.b_has_object
                resp.b_arm_clear, resp.c_object_count = st.b_arm_clear, st.c_object_count
                resp.success, resp.message = True, st.detail
        except Exception as e:
            self.get_logger().error('get_zone_state 例外:\n' + traceback.format_exc())
            resp.success, resp.message = False, f'內部錯誤: {e!r}'
        return resp

    def _srv_capture(self, req, resp):
        """不排序：清點 A 區瓶蓋數。影像/場地有問題才回 success=false；數量不符只寫在 message。"""
        resp.count, resp.low_confidence = 0, False
        try:
            colors, why = self._colors()
            if colors is None:
                resp.success, resp.message = False, why
                return resp
            c = self.cfg
            n = min(c.avg_frames, self.frames.maxlen)
            frames = self.frames.wait_fresh(n, c.service_timeout_s)
            if len(frames) < n:
                resp.success = False
                resp.message = f'逾時：{c.service_timeout_s:.1f}s 內只收到 {len(frames)}/{n} 張新影像（影像中斷？）'
                return resp
            img = frames[-1] if n == 1 else np.mean(np.stack(frames), axis=0, dtype=np.float32).astype(np.uint8)
            rois = self._ensure_rois(img.shape)
            if rois is None:
                resp.success, resp.message = False, '場地尚未定位（請先呼叫 recalibrate_zones）'
                return resp
            with self._lock:
                corners = self._zones['a'].corners
            count = self._count_in(hsv_mask(img, colors['obj'], c.obj_mp), 'a', rois, corners)
            resp.success, resp.count = True, int(count)
            resp.message = f'A 區 {count} 個瓶蓋；{NO_RANKING}'
            if count != c.expected:
                resp.message += f'（注意：預期 {c.expected} 個）'
        except Exception as e:
            self.get_logger().error('capture_ranking 例外:\n' + traceback.format_exc())
            resp.success, resp.message = False, f'內部錯誤: {e!r}'
        return resp

    def _srv_get_rank(self, req, resp):
        resp.success, resp.message, resp.rank, resp.distance_mm = True, NO_RANKING + '；rank 固定為 0', 0, 0.0
        return resp

    def _srv_clear_taken(self, req, resp):
        resp.success, resp.message = True, NO_RANKING + '；沒有「已取走」標記需要清除'
        return resp

    # ------------------------------------------------------------------ #
    # debug 視窗
    # ------------------------------------------------------------------ #
    def get_debug_image(self) -> Optional[np.ndarray]:
        return self._debug_img

    def _render(self, frame: np.ndarray, st: Optional[ZoneState], obj_mask: Optional[np.ndarray],
                rois: Optional[Dict[str, ZoneRoi]]) -> None:
        if not self.show_debug:
            return
        img = frame.copy()
        if obj_mask is not None and rois:   # 瓶蓋輪廓（只畫場地範圍內的）
            inside = np.zeros_like(obj_mask)
            for roi in rois.values():
                inside = cv2.bitwise_or(inside, roi.mask)
            cnts, _ = cv2.findContours(cv2.bitwise_and(obj_mask, inside), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, [c for c in cnts if cv2.contourArea(c) >= 20], -1, (0, 255, 255), 2, cv2.LINE_AA)
        with self._lock:
            zones = dict(self._zones)
        labels = {z: z.upper() for z in ZONES}
        if st is not None and st.valid:
            labels['a'] = 'A: OBJECT' if st.a_has_object else 'A: empty'
            labels['b'] = 'B: ' + ('OBJECT ' if st.b_has_object else '') + ('clear' if st.b_arm_clear else 'ARM')
            labels['c'] = f'C: n={st.c_object_count}'
        for z, info in zones.items():
            draw_poly(img, info.corners, ZONE_BGR[z], 2)
            put_text(img, labels[z] + (' (saved)' if info.source == 'loaded' else ''),
                     (info.corners[0][0] + 4, info.corners[0][1] + 18), ZONE_BGR[z], 0.55, 1)
        status = 'zones OK' if len(zones) == 3 else 'ZONES MISSING'
        put_text(img, f'{status}  frame_age={self.frames.age() or 0:.2f}s  '
                      f'{"" if st is None or st.valid else "STATE INVALID"}', (8, 18), (255, 255, 255), 0.5, 1)
        self._debug_img = img


def main(args=None):
    rclpy.init(args=args)
    node = None
    code = 0
    try:
        node = EnvVisionNode()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        spin_node(rclpy, node, executor, node.show_debug, 'env_vision')
    except (KeyError, ValueError) as e:   # 缺少設定
        print(f'[env_vision_node] 啟動失敗：{e}', file=sys.stderr)
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
