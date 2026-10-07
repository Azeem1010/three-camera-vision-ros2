#!/usr/bin/env python3
"""環境相機節點 env_vision_node。

職責
  * 訂閱 /env_cam/image_raw（只訂閱 topic，絕不自己開 /dev/video）
  * 開機（桌面淨空）時以 HSV 找出 A/B/C 站台，角點存入 zones.yaml；失敗則讀上次存檔並警告
  * 持續判斷：各區有無物件、B 區是否有手臂；結果存成「最新狀態」，service 被呼叫時直接回傳
  * 批次開始時拍快照（8 個物件由小到大排序）並提供 A 站平面座標 -> rank 的配對

Services（皆有逾時處理，失敗時 success=false 並附明確 message）
  /vision/env/recalibrate_zones   重新定位三區
  /vision/env/get_zone_state      a_has_object / b_has_object / b_arm_clear / c_object_count
  /vision/env/capture_ranking     拍快照並排序
  /vision/env/get_rank            輸入 A 站平面座標 -> rank（已取走者標記，避免重複配對）
  /vision/env/clear_taken         （額外）清除「已取走」標記，供重試/復原使用

執行緒模型：MultiThreadedExecutor；影像訂閱、處理計時器、service 伺服端各放在不同的
MutuallyExclusiveCallbackGroup，service 等新影像時不會擋住影像接收。
"""
from __future__ import annotations

import math
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from vision_interfaces.srv import CaptureRanking, GetRank, GetZoneState, RecalibrateZones

from vision_pkg import calib_io
from vision_pkg.debug_draw import draw_poly, put_text
from vision_pkg.node_common import (Debouncer, FlatParams, FrameBuffer, LatestValue,
                                    image_msg_to_bgr, read_ros_params, spin_node)
from vision_pkg.vision_core.color_zone import (HsvRange, MaskParams, OccupancyParams, ZoneFindParams,
                                               ZoneRoi, count_blobs, judge_arm_present, judge_object_present,
                                               locate_zones, make_roi, measure_zones)
from vision_pkg.vision_core.geometry import polygon_area
from vision_pkg.vision_core.size_ranker import SizeRanker, make_ranker
from vision_pkg.vision_core.snapshot import ObjectInfo, capture_snapshot, match_object
from vision_pkg.vision_core.square_detect import SquareParams

ZONES = ('a', 'b', 'c')
ZONE_BGR = {'a': (255, 128, 0), 'b': (0, 200, 0), 'c': (220, 220, 220)}
CAM_NAME = calib_io.CAMERA_NAMES['env']


@dataclass
class ZoneInfo:
    corners: np.ndarray     # (4,2) tl,tr,br,bl（影像像素）
    baseline: float         # 淨空時站台色佔 ROI 的比例
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


@dataclass
class Snapshot:
    objects: List[ObjectInfo]
    taken: Set[int] = field(default_factory=set)


class EnvVisionNode(Node):
    def __init__(self):
        super().__init__('env_vision_node', automatically_declare_parameters_from_overrides=True)
        self._lock = threading.RLock()

        # ---- 參數（全部來自 config/env_vision.yaml；缺少會在這裡一次報錯） ----
        self.P = FlatParams(read_ros_params(self))
        self._last_flat = dict(self.P.flat)
        self._last_param_check = 0.0
        self.cfg = self._build_cfg(self.P)
        self.ranker: SizeRanker = self._make_ranker(self.cfg)
        self.show_debug = self.P.b('show_debug')
        self.calib_dir = calib_io.resolve_calib_dir(self.P.s('calib_dir'))
        self.get_logger().info(f'校正檔目錄: {self.calib_dir}')

        # ---- 狀態 ----
        self.frames = FrameBuffer(maxlen=max(8, self.cfg.boot_frames + 2, self.cfg.avg_frames + 2))
        self.state = LatestValue()
        self._zones: Dict[str, ZoneInfo] = {}
        self._rois: Dict[str, ZoneRoi] = {}
        self._rois_key = None
        self._snap: Optional[Snapshot] = None
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
        self.get_logger().info(f'env_vision_node 就緒，訂閱 {self.P.s("image_topic")}，等待影像與站台定位…')

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
        c.zone_color = {z: P.s(f'zone_color.{z}') for z in ZONES}
        c.zone_size = {z: tuple(P.fl(f'zone_size_mm.{z}')) for z in ZONES}
        for z, s in c.zone_size.items():
            if len(s) != 2 or min(s) <= 0:
                raise ValueError(f'zone_size_mm.{z} 必須是兩個正數 [寬, 高]')
        c.zone_mp = P.group(MaskParams, 'zone_mask')
        c.zone_find = P.group(ZoneFindParams, 'zone_find')
        c.orange_mp = P.group(MaskParams, 'orange_mask')
        c.black_mp = P.group(MaskParams, 'black_mask')
        c.occ = P.group(OccupancyParams, 'occupancy')
        c.obj_frames = P.i('debounce.object_frames')
        c.arm_present_frames = P.i('debounce.arm_present_frames')
        c.arm_clear_frames = P.i('debounce.arm_clear_frames')
        c.count_min_mm2 = P.f('c_count.min_blob_area_mm2')
        c.count_median = P.i('c_count.median_frames')
        c.ppm = P.f('snapshot.warp_px_per_mm')
        c.avg_frames = max(1, P.i('snapshot.avg_frames'))
        c.expected = P.i('snapshot.expected_count')
        c.square = P.group(SquareParams, 'square')
        c.ranker_type = P.s('ranker.type')
        c.margin_px = P.f('ranker.confidence_margin_px')
        c.table = P.fl('ranker.table_mm')
        c.table_tol = P.f('ranker.table_tolerance_mm')
        c.match_mm = P.f('match_threshold_mm')
        return c

    @staticmethod
    def _make_ranker(c: SimpleNamespace) -> SizeRanker:
        # 排序信心門檻的單位是「A 區校正後像素」，換成 mm 後交給 ranker（ranker 一律用 mm）
        return make_ranker(c.ranker_type, c.margin_px / c.ppm, c.table, c.table_tol)

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
            ranker = self._make_ranker(cfg)
        except Exception as e:
            self.get_logger().error(f'參數更新無效，沿用舊設定: {e}')
            return
        self.cfg, self.ranker = cfg, ranker
        self._make_debouncers()
        self._rois_key = None
        self.get_logger().info('參數已更新')

    def _ranges(self) -> Tuple[Optional[Dict[str, HsvRange]], str]:
        """各站台的 HSV 範圍；hsv.yaml 缺顏色時回傳原因。"""
        hsv = self.hsv.value
        if hsv is None:
            return None, f'HSV 設定載入失敗: {self.hsv.error}'
        try:
            return {z: hsv[self.cfg.zone_color[z]] for z in ZONES}, ''
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
    # 站台定位
    # ------------------------------------------------------------------ #
    def _locate_zones(self, frames: List[np.ndarray], source: str) -> Tuple[bool, str]:
        """用多張影格定位三個站台；失敗的站台沿用現有/上次存檔值並警告。回傳 (三區皆成功, 說明)。"""
        ranges, why = self._ranges()
        if ranges is None:
            return False, why
        c = self.cfg
        res = locate_zones(frames, ranges, c.zone_mp, c.zone_find, c.boot_min_det, c.erode, c.min_baseline)
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
                self._snap = None  # 站台平面改了，舊快照作廢
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
            self.get_logger().warning(f'等不到足夠影像（{self.cfg.boot_wait_s:.0f}s），改用上次存檔的站台（{len(saved)} 區）')
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
        if latest is None:
            return
        seq, t_arr, frame = latest
        if seq == self._last_seq or now - t_arr > self.cfg.image_timeout_s:
            return  # 沒有新影格 / 影像已中斷：不產生新結果，讓舊結果自然過期
        self._last_seq = seq

        ranges, why = self._ranges()
        rois = self._ensure_rois(frame.shape)
        if ranges is None:
            self.state.set(ZoneState(False, why))
            return
        if rois is None:
            self.state.set(ZoneState(False, '站台尚未定位（請在桌面淨空時呼叫 /vision/env/recalibrate_zones）'))
            self._render(frame, None, None, None)
            return

        c = self.cfg
        hsv = self.hsv.value
        with self._lock:
            zones = dict(self._zones)
        metrics, orange_mask = measure_zones(frame, rois, ranges, hsv['orange'], hsv['black'],
                                             c.zone_mp, c.orange_mp, c.black_mp)
        a_raw, a_info = judge_object_present(metrics['a'], zones['a'].baseline, c.occ)
        b_raw, b_info = judge_object_present(metrics['b'], zones['b'].baseline, c.occ)
        arm_raw, arm_info = judge_arm_present(metrics['b'], zones['b'].baseline, c.occ)

        # C 區橘色塊數：面積門檻由 mm² 換算成像素²（以站台四邊形平均比例）
        w_mm, h_mm = c.zone_size['c']
        mm2_per_px = (w_mm * h_mm) / max(1.0, polygon_area(zones['c'].corners))
        in_c = cv2.bitwise_and(orange_mask, rois['c'].mask)
        self._c_counts.append(count_blobs(in_c, c.count_min_mm2 / mm2_per_px))

        prev_arm = self._deb_arm.state
        a_obj = self._deb_a.update(a_raw)
        b_obj = self._deb_b.update(b_raw)
        arm = self._deb_arm.update(arm_raw)
        if prev_arm is not None and prev_arm != arm:
            self.get_logger().info('B 區偵測到手臂進入' if arm else 'B 區手臂已離開（連續確認 clear）')
        c_cnt = int(np.median(list(self._c_counts)))
        detail = f'A[{a_info}] B[{b_info}; {arm_info}] C[n={c_cnt}]'
        st = ZoneState(True, 'ok', bool(a_obj), bool(b_obj), not bool(arm), c_cnt, detail)
        self.state.set(st)
        self._render(frame, st, orange_mask, rois)

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
        resp.count, resp.low_confidence = 0, False
        try:
            with self._lock:
                self._snap = None   # 先作廢舊快照：這次若失敗，不能讓後面的 get_rank 配到上一批的資料
                zone = self._zones.get('a')
            if zone is None:
                resp.success, resp.message = False, 'A 站台尚未定位（請先呼叫 recalibrate_zones）'
                return resp
            hsv = self.hsv.value
            if hsv is None:
                resp.success, resp.message = False, f'HSV 設定載入失敗: {self.hsv.error}'
                return resp
            c = self.cfg
            n = min(c.avg_frames, self.frames.maxlen)
            frames = self.frames.wait_fresh(n, c.service_timeout_s)
            if len(frames) < n:
                resp.success = False
                resp.message = f'逾時：{c.service_timeout_s:.1f}s 內只收到 {len(frames)}/{n} 張新影像（影像中斷？）'
                return resp
            img = frames[-1] if n == 1 else np.mean(np.stack(frames), axis=0, dtype=np.float32).astype(np.uint8)
            res = capture_snapshot(img, zone.corners, c.zone_size['a'], c.ppm, hsv['orange'], c.orange_mp,
                                   c.square, self.ranker, c.expected)
            resp.success, resp.message = res.success, res.message
            resp.count, resp.low_confidence = int(res.count), bool(res.low_confidence)
            if res.success:
                with self._lock:
                    self._snap = Snapshot(res.objects)
                self.get_logger().info(f'快照完成: {res.count} 個物件，low_confidence={res.low_confidence}')
            else:
                self.get_logger().warning(f'快照失敗: {res.message}')
        except Exception as e:
            self.get_logger().error('capture_ranking 例外:\n' + traceback.format_exc())
            resp.success, resp.message = False, f'內部錯誤: {e!r}'
        return resp

    def _srv_get_rank(self, req, resp):
        resp.rank, resp.distance_mm = 0, 0.0
        try:
            if not (math.isfinite(req.x_mm) and math.isfinite(req.y_mm)):
                resp.success, resp.message = False, '座標必須是有限數值'
                return resp
            with self._lock:
                snap = self._snap
                objs = list(snap.objects) if snap else []
                taken = set(snap.taken) if snap else set()
            if snap is None:
                resp.success, resp.message = False, '尚未拍攝快照（請先 capture_ranking；或上次拍攝失敗）'
                return resp
            m = match_object(objs, taken, req.x_mm, req.y_mm, self.cfg.match_mm)
            resp.success, resp.message = m.success, m.message
            resp.rank, resp.distance_mm = int(m.rank), float(m.distance_mm)
            if m.success and not req.peek_only:
                with self._lock:
                    snap.taken.add(m.index)
        except Exception as e:
            self.get_logger().error('get_rank 例外:\n' + traceback.format_exc())
            resp.success, resp.message = False, f'內部錯誤: {e!r}'
        return resp

    def _srv_clear_taken(self, req, resp):
        with self._lock:
            n = len(self._snap.taken) if self._snap else 0
            if self._snap:
                self._snap.taken.clear()
        resp.success, resp.message = True, f'已清除 {n} 個「已取走」標記'
        return resp

    # ------------------------------------------------------------------ #
    # debug 視窗
    # ------------------------------------------------------------------ #
    def get_debug_image(self) -> Optional[np.ndarray]:
        return self._debug_img

    def _render(self, frame: np.ndarray, st: Optional[ZoneState], orange_mask: Optional[np.ndarray],
                rois: Optional[Dict[str, ZoneRoi]]) -> None:
        if not self.show_debug:
            return
        img = frame.copy()
        if orange_mask is not None and rois:   # 橘色物件輪廓（只畫站台範圍內的）
            inside = np.zeros_like(orange_mask)
            for roi in rois.values():
                inside = cv2.bitwise_or(inside, roi.mask)
            cnts, _ = cv2.findContours(cv2.bitwise_and(orange_mask, inside), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, [c for c in cnts if cv2.contourArea(c) >= 20], -1, (0, 128, 255), 1, cv2.LINE_AA)
        with self._lock:
            zones = dict(self._zones)
            snap = self._snap
            objs = list(snap.objects) if snap else []
            taken = set(snap.taken) if snap else set()
        labels = {z: z.upper() for z in ZONES}
        if st is not None and st.valid:
            labels['a'] = 'A: OBJECT' if st.a_has_object else 'A: empty'
            labels['b'] = 'B: ' + ('OBJECT ' if st.b_has_object else '') + ('clear' if st.b_arm_clear else 'ARM')
            labels['c'] = f'C: n={st.c_object_count}'
        for z, info in zones.items():
            draw_poly(img, info.corners, ZONE_BGR[z], 2)
            put_text(img, labels[z] + (' (saved)' if info.source == 'loaded' else ''),
                     (info.corners[0][0], info.corners[0][1] - 6), ZONE_BGR[z], 0.55, 1)
        for i, o in enumerate(objs):
            col = (128, 128, 128) if i in taken else (0, 255, 255)
            cv2.circle(img, (int(o.px), int(o.py)), 4, col, -1, cv2.LINE_AA)
            put_text(img, f'r{o.rank}' + ('x' if i in taken else ''), (o.px + 5, o.py - 5), col, 0.5, 1)
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
