#!/usr/bin/env python3
"""手眼相機 1 節點 arm1_vision_node（手臂 1 在 Home_A 待機觀測 A 區）。

瓶蓋大小都一樣、不排序：手臂 1 夾「任一個」都可以，放置順序由 C 區格位編號決定。

職責
  * 訂閱 /arm1_cam/image_raw，持續找藍色圓形瓶蓋，保存最新結果
  * /vision/arm1/get_pick_target：從看得到的瓶蓋中挑一個回傳手臂 1 基座座標（m），z = 場地高度 + 物件高度
      - 只考慮可達範圍內的瓶蓋
      - 優先挑「周圍沒有其他瓶蓋擋住」者（最近鄰距離 >= pick.min_clearance_m），再挑離基座最近者
      - 回傳的 rank = 請求的 target_rank（原樣帶回，協調器可拿來當「這次要放第幾格」）
      - peek_only 保留但沒有作用（不再需要標記已取走）
  * /vision/arm1/pick_target（topic，vision_interfaces/PickTarget）：以 process_hz 持續發布同樣的選擇結果
  * /vision/arm1/detections（topic，vision_interfaces/PickDetections）：每處理一張新影格發布一次，
    含各瓶蓋的像素 / A 區平面 / 手臂 1 基座座標

此節點不呼叫其他 service。執行緒模型：MultiThreadedExecutor；影像 / 處理計時器 / service / topic 發布
各自放在不同的 MutuallyExclusiveCallbackGroup。
"""
from __future__ import annotations

import math
import sys
import time
import traceback
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_interfaces.msg import PickDetections, PickTarget
from vision_interfaces.srv import GetPickTarget

from vision_pkg import calib_io
from vision_pkg.debug_draw import draw_poly, put_text, put_text_centered
from vision_pkg.node_common import (FlatParams, FrameBuffer, LatestValue, image_msg_to_bgr, image_problem,
                                    read_ros_params, spin_node, stamp)
from vision_pkg.vision_core.cap_detect import Cap, CapParams, describe_rejected, detect_caps, nearest_neighbor_dist
from vision_pkg.vision_core.color_zone import MaskParams, hsv_mask
from vision_pkg.vision_core.geometry import apply_homography, polygon_mask

CAM_NAME = calib_io.CAMERA_NAMES['arm1']


@dataclass
class Detection:
    caps: List[Cap]
    rejected: List[Cap]
    roi_poly: Optional[np.ndarray]    # A 區在影像中的範圍（需 a_station_homography 有效）


@dataclass
class Target:
    index: int              # det.caps 的索引
    x: float                # 手臂 1 基座座標 m
    y: float
    z: float
    message: str


class Arm1VisionNode(Node):
    def __init__(self):
        super().__init__('arm1_vision_node', automatically_declare_parameters_from_overrides=True)
        self.P = FlatParams(read_ros_params(self))
        self._last_flat = dict(self.P.flat)
        self._last_param_check = 0.0
        self.cfg = self._build_cfg(self.P)
        self.show_debug = self.P.b('show_debug')
        self.calib_dir = calib_io.resolve_calib_dir(self.P.s('calib_dir'))
        self.get_logger().info(f'校正檔目錄: {self.calib_dir}')

        self.frames = FrameBuffer(maxlen=4)
        self.detections = LatestValue()
        self._last_seq = 0
        self._roi_key = None
        self._roi_mask: Optional[np.ndarray] = None
        self._roi_poly: Optional[np.ndarray] = None
        self._last_target: Tuple[float, Optional[Tuple[float, float]]] = (0.0, None)   # (時間, 目標像素)
        self._debug_img: Optional[np.ndarray] = None

        period = self.cfg.hsv_reload_period_s
        pkg = calib_io.package_config_dir()
        self.hsv = calib_io.ReloadableConfig(lambda: calib_io.load_hsv_ranges(self.calib_dir),
                                             calib_io.hsv_paths(self.calib_dir), period)
        self.h_arm = self._homography_watch('arm1_homography', pkg, period)
        self.h_plane = self._homography_watch('a_station_homography', pkg, period)

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
        g_pub = MutuallyExclusiveCallbackGroup()
        self.create_subscription(Image, self.P.s('image_topic'), self._on_image, qos_profile_sensor_data,
                                 callback_group=g_img)
        self.create_timer(1.0 / self.cfg.process_hz, self._on_tick, callback_group=g_proc)
        self.create_service(GetPickTarget, '/vision/arm1/get_pick_target', self._srv_pick, callback_group=g_srv)
        self.pub_target = self.create_publisher(PickTarget, '/vision/arm1/pick_target', 10)
        self.create_timer(1.0 / self.cfg.process_hz, self._on_publish, callback_group=g_pub)
        self.pub_det = self.create_publisher(PickDetections, '/vision/arm1/detections', 10)
        self.get_logger().info(f'arm1_vision_node 就緒，訂閱 {self.P.s("image_topic")}')

    # ------------------------------------------------------------------ #
    def _homography_watch(self, name: str, pkg, period: float) -> calib_io.ReloadableConfig:
        return calib_io.ReloadableConfig(lambda: calib_io.load_homography(name, self.calib_dir),
                                         [self.calib_dir / f'{name}.yaml', pkg / f'{name}.yaml'], period)

    def _build_cfg(self, P: FlatParams) -> SimpleNamespace:
        c = SimpleNamespace()
        c.process_hz = max(1.0, P.f('process_hz'))
        c.image_timeout_s = P.f('image_timeout_s')
        c.service_timeout_s = P.f('service_timeout_s')
        c.hsv_reload_period_s = P.f('hsv_reload_period_s')
        c.station = tuple(P.fl('a_station_size_mm'))
        if len(c.station) != 2 or min(c.station) <= 0:
            raise ValueError('a_station_size_mm 必須是兩個正數 [寬, 高]')
        c.roi_margin_mm = P.f('roi_margin_mm')
        c.station_height_mm = P.f('station_height_mm')
        c.object_height_mm = P.f('object_height_mm')
        c.reach_min = P.f('reach.min_m')
        c.reach_max = P.f('reach.max_m')
        c.min_clearance = P.f('pick.min_clearance_m')
        c.obj_color = P.s('object_color')
        c.obj_mp = P.group(MaskParams, 'object_mask')
        c.cap = P.group(CapParams, 'cap')
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
            self.cfg = self._build_cfg(FlatParams(flat))
            self._roi_key = None
            self.get_logger().info('參數已更新')
        except Exception as e:
            self.get_logger().error(f'參數更新無效，沿用舊設定: {e}')

    # ------------------------------------------------------------------ #
    def _on_image(self, msg: Image) -> None:
        try:
            bgr = image_msg_to_bgr(msg)
            if self._undist is not None:
                bgr = self._undist.apply(bgr)
            self.frames.push(bgr)
        except Exception as e:
            self.get_logger().warning(f'影像轉換失敗: {e}', throttle_duration_sec=5.0)

    def _roi_for(self, shape) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """A 區在手眼相機影像中的多邊形與遮罩（由 a_station_homography 反推）；未標定則 (None, None)。"""
        hp = self.h_plane.value
        if hp is None or not hp['valid']:
            return None, None
        key = (shape[:2], hp['H'].tobytes(), self.cfg.station, self.cfg.roi_margin_mm)
        if key != self._roi_key:
            m = self.cfg.roi_margin_mm
            w, h = self.cfg.station
            try:
                poly = apply_homography(np.linalg.inv(hp['H']),
                                        [[-m, -m], [w + m, -m], [w + m, h + m], [-m, h + m]])
                self._roi_mask = polygon_mask(shape, poly, 0)
                self._roi_poly = poly
            except Exception as e:
                self.get_logger().warning(f'無法由 a_station_homography 推算 A 區範圍: {e}', throttle_duration_sec=10.0)
                self._roi_mask, self._roi_poly = None, None
            self._roi_key = key
        return self._roi_poly, self._roi_mask

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
        for name, w in (('arm1_homography', self.h_arm), ('a_station_homography', self.h_plane)):
            if w.poll():
                self.get_logger().info(f'{name} 已重新載入（valid={w.value["valid"]}）')
        latest = self.frames.latest()
        problem = image_problem(latest, now, self.cfg.image_timeout_s)
        if problem:
            self._publish_invalid(problem)
            return
        seq, t_arr, frame = latest
        if seq == self._last_seq:
            return
        self._last_seq = seq
        hsv = self.hsv.value
        if hsv is None or self.cfg.obj_color not in hsv:
            why = f'HSV 設定不可用（需要 {self.cfg.obj_color}）: {self.hsv.error}'
            self.get_logger().error(why, throttle_duration_sec=10.0)
            self._publish_invalid(why)
            return
        poly, roi = self._roi_for(frame.shape)
        mask = hsv_mask(frame, hsv[self.cfg.obj_color], self.cfg.obj_mp)
        if roi is not None:
            mask = cv2.bitwise_and(mask, roi)
        caps, rejected = detect_caps(mask, self.cfg.cap)
        self.detections.set(Detection(caps, rejected, poly))
        self._publish(caps, rejected)
        self._render(frame, caps, rejected, poly)

    # ------------------------------------------------------------------ #
    def _publish_invalid(self, message: str) -> None:
        m = PickDetections()
        m.valid, m.message = False, message
        self.pub_det.publish(stamp(self, m, CAM_NAME))

    def _publish(self, caps: List[Cap], rejected: List[Cap]) -> None:
        c = self.cfg
        m = PickDetections()
        m.valid = True
        m.count = len(caps)
        m.pixel_x = [float(s.cx) for s in caps]
        m.pixel_y = [float(s.cy) for s in caps]
        m.side_px = [float(s.diameter) for s in caps]
        notes = [f'{len(caps)} 個瓶蓋']
        if rejected:
            notes.append(f'{len(rejected)} 個藍色區塊被排除（{describe_rejected(rejected)}）')
        px = np.array([[s.cx, s.cy] for s in caps], dtype=np.float64).reshape(-1, 2)
        hp, ha = self.h_plane.value, self.h_arm.value
        m.plane_valid = bool(hp is not None and hp['valid'])
        if m.plane_valid and caps:
            plane = apply_homography(hp['H'], px)
            m.plane_x_mm = [float(v) for v in plane[:, 0]]
            m.plane_y_mm = [float(v) for v in plane[:, 1]]
        elif not m.plane_valid:
            notes.append('a_station_homography 未標定（偵測整張影像）')
        m.base_valid = bool(ha is not None and ha['valid'])
        if m.base_valid and caps:
            base = apply_homography(ha['H'], px)
            m.base_x = [float(v) for v in base[:, 0]]
            m.base_y = [float(v) for v in base[:, 1]]
        elif not m.base_valid:
            notes.append('arm1_homography 未標定')
        m.base_z = (c.station_height_mm + c.object_height_mm) / 1000.0
        m.message = '；'.join(notes)
        self.pub_det.publish(stamp(self, m, CAM_NAME))

    # ------------------------------------------------------------------ #
    def _fail(self, resp, msg: str):
        resp.success, resp.message = False, msg
        self.get_logger().warning(f'get_pick_target 失敗: {msg}')
        return resp

    def _srv_pick(self, req, resp):
        resp.rank = 0
        resp.x = resp.y = resp.z = 0.0
        try:
            if req.target_rank < 0:
                return self._fail(resp, f'target_rank 不可為負（{req.target_rank}）')
            det, why = self.detections.wait_fresh(self.cfg.image_timeout_s, self.cfg.service_timeout_s)
            if det is None:
                return self._fail(resp, f'沒有可用的影像結果: {why}')
            t, err = self._find_target(det)
            if t is None:
                return self._fail(resp, err)
            resp.success = True
            resp.rank = int(req.target_rank)
            resp.x, resp.y, resp.z = t.x, t.y, t.z
            resp.message = t.message
            return resp
        except Exception as e:
            self.get_logger().error('get_pick_target 例外:\n' + traceback.format_exc())
            return self._fail(resp, f'內部錯誤: {e!r}')

    def _on_publish(self) -> None:
        """以處理頻率發布目前會選中的目標。失敗時照樣發布 success=false 與原因。"""
        try:
            msg = PickTarget()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'arm1_base'
            det, why = self.detections.wait_fresh(self.cfg.image_timeout_s, 0.0)
            if det is None:
                msg.success, msg.message = False, f'沒有可用的影像結果: {why}'
            else:
                t, err = self._find_target(det)
                if t is None:
                    msg.success, msg.message = False, err
                else:
                    msg.success, msg.message = True, t.message
                    msg.x, msg.y, msg.z = t.x, t.y, t.z
            self.pub_target.publish(msg)
        except Exception:
            self.get_logger().error('pick_target 發布例外:\n' + traceback.format_exc(), throttle_duration_sec=5.0)

    def _find_target(self, det: Detection) -> Tuple[Optional[Target], str]:
        """從看得到的瓶蓋挑一個（可達、周圍淨空優先、離基座近優先）。回傳 (目標, '') 或 (None, 原因)。"""
        c = self.cfg
        ha = self.h_arm.value
        if ha is None or not ha['valid']:
            return None, ('arm1_homography 尚未標定（valid=false 或讀取失敗）；'
                          '請執行 ros2 run vision_pkg calibrate_homography --target arm1')
        if not det.caps:
            extra = f'；另有 {len(det.rejected)} 個藍色區塊被排除（{describe_rejected(det.rejected)}）' if det.rejected else ''
            return None, f'A 區沒有偵測到瓶蓋{extra}'

        # 被排除的區塊（例如兩顆相連）也算「障礙物」，計算淨空距離時一起考慮
        all_px = np.array([[b.cx, b.cy] for b in det.caps + det.rejected], dtype=np.float64)
        all_base = apply_homography(ha['H'], all_px)
        n = len(det.caps)
        clearance = nearest_neighbor_dist(all_base)[:n]
        radius = np.hypot(all_base[:n, 0], all_base[:n, 1])
        reachable = [i for i in range(n) if c.reach_min <= radius[i] <= c.reach_max]
        if not reachable:
            i = int(np.argmin(radius))
            return None, (f'{n} 個瓶蓋都超出可達範圍 [{c.reach_min:.2f}, {c.reach_max:.2f}] m'
                          f'（最近者 ({all_base[i][0]:.3f}, {all_base[i][1]:.3f}) m 半徑 {radius[i]:.3f} m）')
        clear = [i for i in reachable if clearance[i] >= c.min_clearance]
        pool = clear or reachable
        i = min(pool, key=lambda k: radius[k])
        x, y = float(all_base[i][0]), float(all_base[i][1])
        z = (c.station_height_mm + c.object_height_mm) / 1000.0
        self._last_target = (time.monotonic(), (det.caps[i].cx, det.caps[i].cy))
        note = '' if clear else f'；注意：沒有周圍淨空 >= {c.min_clearance:.3f} m 的瓶蓋，選最近者'
        msg = (f'base({x:.4f}, {y:.4f}, {z:.4f}) m，半徑 {radius[i]:.3f} m，最近鄰 {clearance[i]:.3f} m；'
               f'看得到 {n} 個、可達 {len(reachable)} 個{note}')
        return Target(i, x, y, z, msg), ''

    # ------------------------------------------------------------------ #
    def get_debug_image(self) -> Optional[np.ndarray]:
        return self._debug_img

    def _render(self, frame, caps: List[Cap], rejected: List[Cap], poly) -> None:
        if not self.show_debug:
            return
        img = frame.copy()
        if poly is not None:
            draw_poly(img, poly, (255, 128, 0), 2)
            put_text(img, 'A ROI', (poly[0][0], poly[0][1] - 6), (255, 128, 0), 0.5, 1)
        else:
            put_text(img, 'a_station_homography not calibrated: ROI = whole image', (8, 36), (0, 0, 255), 0.5, 1)
        t_ov, tgt = self._last_target
        fresh = tgt is not None and time.monotonic() - t_ov < 2.0
        coords = self._coord_labels(caps)
        for s, coord in zip(caps, coords):
            is_target = fresh and math.hypot(s.cx - tgt[0], s.cy - tgt[1]) < 1.0
            col = (0, 0, 255) if is_target else (0, 255, 0)
            cv2.circle(img, (int(round(s.cx)), int(round(s.cy))), int(round(s.radius)), col, 3 if is_target else 2,
                       cv2.LINE_AA)
            if is_target:
                put_text_centered(img, 'PICK', (s.cx, s.cy - s.radius - 10), col, 0.45, 1)
            for k, line in enumerate(coord):   # x、y 分兩行寫在瓶蓋正下方，相鄰的瓶蓋才不會疊在一起
                put_text_centered(img, line, (s.cx, s.cy + s.radius + 9 + 13 * k), col, 0.35, 1)
        for s in rejected:
            cv2.circle(img, (int(round(s.cx)), int(round(s.cy))), int(round(s.radius)), (0, 0, 255), 1, cv2.LINE_AA)
            put_text(img, 'x ' + s.reject_reason.split('；')[0], (s.cx - 20, s.cy), (0, 0, 255), 0.4, 1)
        unit = 'x / y = base m' if self._arm_h_valid() else 'x / y = pixel (arm1_homography not calibrated)'
        put_text(img, f'caps={len(caps)} rejected={len(rejected)}  coords: {unit}', (8, 18), (255, 255, 255), 0.5, 1)
        self._debug_img = img

    def _arm_h_valid(self) -> bool:
        ha = self.h_arm.value
        return ha is not None and bool(ha['valid'])

    def _coord_labels(self, caps: List[Cap]) -> List[Tuple[str, str]]:
        """debug 用：每顆瓶蓋的手臂 1 基座座標（m）；未標定時改顯示像素座標。"""
        if not caps:
            return []
        px = np.array([[s.cx, s.cy] for s in caps], dtype=np.float64)
        if self._arm_h_valid():
            base = apply_homography(self.h_arm.value['H'], px)
            return [(f'{x:.3f}', f'{y:.3f}') for x, y in base]
        return [(f'{x:.0f}', f'{y:.0f}') for x, y in px]


def main(args=None):
    rclpy.init(args=args)
    node = None
    code = 0
    try:
        node = Arm1VisionNode()
        executor = MultiThreadedExecutor(num_threads=5)   # 4 個 callback group + 餘裕
        executor.add_node(node)
        spin_node(rclpy, node, executor, node.show_debug, 'arm1_vision')
    except (KeyError, ValueError) as e:
        print(f'[arm1_vision_node] 啟動失敗：{e}', file=sys.stderr)
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
