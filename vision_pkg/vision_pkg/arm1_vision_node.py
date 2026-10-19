#!/usr/bin/env python3
"""手眼相機 1 節點 arm1_vision_node（手臂 1 在 Home_A 待機觀測 A 區）。

職責
  * 訂閱 /arm1_cam/image_raw，持續在 A 區範圍內找橘色正方形，保存最新結果
  * /vision/arm1/get_pick_target：
      - 以 a_station_homography 把像素中心轉成 A 站平面座標（mm），逐一呼叫
        /vision/env/get_rank 取得 rank
      - 取協調器指定的 target_rank（0 = 預設取 rank 最小者）
      - 以 arm1_homography 轉成手臂 1 基座座標（m），z = 站台高度 + 物件高度
      - 檢查可達範圍；peek_only=false 時通知環境相機把該物件標記為「已取走」
  * /vision/arm1/pick_target（topic，vision_interfaces/PickTarget）：以 process_hz 持續發布
    rank 最小的目標，算法同 get_pick_target(target_rank=0, peek_only=true)，不會標記已取走
  * Topic /vision/arm1/detections（vision_interfaces/PickDetections）：每處理一張新影格發布一次，
    含各正方形的像素 / A 站平面 / 手臂 1 基座座標（不含 rank，rank 仍需 get_pick_target）

執行緒模型（避免 service 內呼叫 service 死鎖）：
  MultiThreadedExecutor；影像 / 處理計時器 / service 伺服端 / env 服務 client 各自放在
  不同的 MutuallyExclusiveCallbackGroup；client 用 call_async + threading.Event 等待（含逾時），
  回呼中絕不 spin。
"""
from __future__ import annotations

import math
import sys
import time
import traceback
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_interfaces.msg import PickTarget
from vision_interfaces.msg import PickDetections
from vision_interfaces.srv import GetPickTarget, GetRank

from vision_pkg import calib_io
from vision_pkg.debug_draw import draw_poly, put_text
from vision_pkg.node_common import (FlatParams, FrameBuffer, LatestValue, call_service, image_msg_to_bgr,
                                    image_problem, read_ros_params, spin_node, stamp)
from vision_pkg.vision_core.color_zone import MaskParams, hsv_mask
from vision_pkg.vision_core.geometry import apply_homography, polygon_mask
from vision_pkg.vision_core.square_detect import Blob, SquareParams, describe_rejected, detect_squares

CAM_NAME = calib_io.CAMERA_NAMES['arm1']


@dataclass
class Detection:
    squares: List[Blob]
    rejected: List[Blob]
    roi_poly: Optional[np.ndarray]    # A 區在影像中的範圍（需 a_station_homography 有效）


@dataclass
class Target:
    rank: int
    x: float                # 手臂 1 基座座標 m
    y: float
    z: float
    plane_x: float          # A 站平面座標 mm（標記已取走時要用）
    plane_y: float
    visible: List[int]      # 畫面中配對得到的所有 rank
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
        self._overlay: Tuple[float, List[Tuple[float, float, int]], Optional[int]] = (0.0, [], None)
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

        # 五個 callback group：影像 / 處理計時器 / service 伺服端 / topic 發布計時器 / env 服務 client
        # （發布計時器也要呼叫 env 服務，獨立一組才不會擋住影像處理或 service）
        g_img = MutuallyExclusiveCallbackGroup()
        g_proc = MutuallyExclusiveCallbackGroup()
        g_srv = MutuallyExclusiveCallbackGroup()
        g_pub = MutuallyExclusiveCallbackGroup()
        g_cli = MutuallyExclusiveCallbackGroup()
        self.create_subscription(Image, self.P.s('image_topic'), self._on_image, qos_profile_sensor_data,
                                 callback_group=g_img)
        self.create_timer(1.0 / self.cfg.process_hz, self._on_tick, callback_group=g_proc)
        self.cli_rank = self.create_client(GetRank, '/vision/env/get_rank', callback_group=g_cli)
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
        c.env_call_timeout_s = P.f('env_call_timeout_s')
        c.hsv_reload_period_s = P.f('hsv_reload_period_s')
        c.station = tuple(P.fl('a_station_size_mm'))
        if len(c.station) != 2 or min(c.station) <= 0:
            raise ValueError('a_station_size_mm 必須是兩個正數 [寬, 高]')
        c.roi_margin_mm = P.f('roi_margin_mm')
        c.station_height_mm = P.f('station_height_mm')
        c.object_height_mm = P.f('object_height_mm')
        c.reach_min = P.f('reach.min_m')
        c.reach_max = P.f('reach.max_m')
        c.orange_mp = P.group(MaskParams, 'orange_mask')
        c.square = P.group(SquareParams, 'square')
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
        if hsv is None or 'orange' not in hsv:
            self.get_logger().error(f'HSV 設定不可用: {self.hsv.error}', throttle_duration_sec=10.0)
            self._publish_invalid(f'HSV 設定不可用: {self.hsv.error}')
            return
        poly, roi = self._roi_for(frame.shape)
        mask = hsv_mask(frame, hsv['orange'], self.cfg.orange_mp)
        if roi is not None:
            mask = cv2.bitwise_and(mask, roi)
        squares, rejected = detect_squares(mask, self.cfg.square)
        self.detections.set(Detection(squares, rejected, poly))
        self._publish(squares, rejected)
        self._render(frame, squares, rejected, poly)

    # ------------------------------------------------------------------ #
    def _publish_invalid(self, message: str) -> None:
        m = PickDetections()
        m.valid, m.message = False, message
        self.pub_det.publish(stamp(self, m, CAM_NAME))

    def _publish(self, squares: List[Blob], rejected: List[Blob]) -> None:
        c = self.cfg
        m = PickDetections()
        m.valid = True
        m.count = len(squares)
        m.pixel_x = [float(s.cx) for s in squares]
        m.pixel_y = [float(s.cy) for s in squares]
        m.side_px = [float(s.side) for s in squares]
        notes = [f'{len(squares)} 個正方形']
        if rejected:
            notes.append(f'{len(rejected)} 個橘色區塊被排除（{describe_rejected(rejected)}）')
        px = np.array([[s.cx, s.cy] for s in squares], dtype=np.float64).reshape(-1, 2)
        hp, ha = self.h_plane.value, self.h_arm.value
        m.plane_valid = bool(hp is not None and hp['valid'])
        if m.plane_valid and squares:
            plane = apply_homography(hp['H'], px)
            m.plane_x_mm = [float(v) for v in plane[:, 0]]
            m.plane_y_mm = [float(v) for v in plane[:, 1]]
        elif not m.plane_valid:
            notes.append('a_station_homography 未標定')
        m.base_valid = bool(ha is not None and ha['valid'])
        if m.base_valid and squares:
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
            return self._pick(req, resp)
        except Exception as e:
            self.get_logger().error('get_pick_target 例外:\n' + traceback.format_exc())
            return self._fail(resp, f'內部錯誤: {e!r}')

    def _pick(self, req, resp):
        c = self.cfg
        deadline = time.monotonic() + c.service_timeout_s
        if req.target_rank < 0:
            return self._fail(resp, f'target_rank 不可為負（{req.target_rank}）；0 = 預設取 rank 最小者')
        det, why = self.detections.wait_fresh(c.image_timeout_s, c.service_timeout_s)
        if det is None:
            return self._fail(resp, f'沒有可用的影像結果: {why}')
        t, err = self._find_target(det, int(req.target_rank), deadline)
        if t is None:
            return self._fail(resp, err)

        if not req.peek_only:   # 確認要取：通知環境相機標記為「已取走」
            remain = deadline - time.monotonic()
            if remain <= 0:
                return self._fail(resp, '逾時：來不及標記已取走')
            rq = GetRank.Request()
            rq.x_mm, rq.y_mm, rq.peek_only = t.plane_x, t.plane_y, False
            r, err = call_service(self.cli_rank, rq, min(c.env_call_timeout_s, remain))
            if r is None:
                return self._fail(resp, f'標記已取走失敗: {err}')
            if not r.success or r.rank != t.rank:
                return self._fail(resp, f'標記已取走失敗: {r.message}')

        resp.success = True
        resp.rank, resp.x, resp.y, resp.z = t.rank, t.x, t.y, t.z
        resp.message = t.message + ('（peek_only，未標記）' if req.peek_only else '')
        return resp

    def _on_publish(self) -> None:
        """以處理頻率發布 rank 最小的目標（只查詢，不標記已取走）。失敗時照樣發布 success=false 與原因。"""
        try:
            msg = PickTarget()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'arm1_base'
            det, why = self.detections.wait_fresh(self.cfg.image_timeout_s, 0.0)
            if det is None:
                msg.success, msg.message = False, f'沒有可用的影像結果: {why}'
            else:
                t, err = self._find_target(det, 0, time.monotonic() + self.cfg.service_timeout_s)
                if t is None:
                    msg.success, msg.message = False, err
                else:
                    msg.success, msg.message = True, t.message
                    msg.rank, msg.x, msg.y, msg.z = t.rank, t.x, t.y, t.z
                    msg.visible_ranks = list(t.visible)
            self.pub_target.publish(msg)
        except Exception:
            self.get_logger().error('pick_target 發布例外:\n' + traceback.format_exc(), throttle_duration_sec=5.0)

    def _find_target(self, det: Detection, target_rank: int, deadline: float) -> Tuple[Optional[Target], str]:
        """向環境相機查出畫面中各物件的 rank，挑出目標並換算成基座座標（不標記已取走）。

        target_rank=0 取 rank 最小者。回傳 (目標, '') 或 (None, 失敗原因)。
        """
        c = self.cfg
        ha, hp = self.h_arm.value, self.h_plane.value
        if ha is None or not ha['valid']:
            return None, ('arm1_homography 尚未標定（valid=false 或讀取失敗）；'
                          '請執行 ros2 run vision_pkg calibrate_homography --target arm1')
        if hp is None or not hp['valid']:
            return None, ('a_station_homography 尚未標定（valid=false 或讀取失敗）；'
                          '請執行 ros2 run vision_pkg calibrate_homography --target a_station')
        if not det.squares:
            extra = f'；另有 {len(det.rejected)} 個橘色區塊被排除（{describe_rejected(det.rejected)}）' if det.rejected else ''
            return None, f'A 區沒有偵測到橘色正方形{extra}'

        px = np.array([[s.cx, s.cy] for s in det.squares], dtype=np.float64)
        plane = apply_homography(hp['H'], px)     # A 站平面座標 mm
        base = apply_homography(ha['H'], px)      # 手臂 1 基座座標 m

        # 逐一向環境相機查 rank（peek_only：先不標記）
        by_rank: Dict[int, Tuple[int, float]] = {}
        unmatched: List[str] = []
        for i in range(len(det.squares)):
            remain = deadline - time.monotonic()
            if remain <= 0:
                return None, f'逾時：{c.service_timeout_s:.1f}s 內只查到 {i}/{len(det.squares)} 個物件的 rank'
            rq = GetRank.Request()
            rq.x_mm, rq.y_mm, rq.peek_only = float(plane[i][0]), float(plane[i][1]), True
            r, err = call_service(self.cli_rank, rq, min(c.env_call_timeout_s, remain))
            if r is None:
                return None, f'無法向環境相機查詢 rank: {err}'
            if not r.success:
                unmatched.append(f'#{i + 1}@({plane[i][0]:.0f},{plane[i][1]:.0f})mm: {r.message}')
                continue
            if r.rank not in by_rank or r.distance_mm < by_rank[r.rank][1]:
                by_rank[r.rank] = (i, r.distance_mm)

        ranks_px = [(float(px[i][0]), float(px[i][1]), rk) for rk, (i, _) in by_rank.items()]
        self._overlay = (time.monotonic(), ranks_px, None)
        if not by_rank:
            return None, f'{len(det.squares)} 個物件全部配對失敗: ' + '；'.join(unmatched[:2])
        if target_rank > 0:
            if target_rank not in by_rank:
                return None, f'指定的 rank {target_rank} 不在可見物件中（可見 rank: {sorted(by_rank)}）'
            rank = target_rank
        else:
            if unmatched:   # 有看得到卻配不到的物件，無法確定誰才是最小 -> 不猜測
                return None, f'有 {len(unmatched)} 個物件無法配對 rank，無法確定最小者: ' + '；'.join(unmatched[:2])
            rank = min(by_rank)

        i = by_rank[rank][0]
        x, y = float(base[i][0]), float(base[i][1])
        radius = math.hypot(x, y)
        if not (c.reach_min <= radius <= c.reach_max):
            return None, (f'rank {rank} 目標 ({x:.3f}, {y:.3f}) m 半徑 {radius:.3f} m '
                          f'超出可達範圍 [{c.reach_min:.2f}, {c.reach_max:.2f}] m')
        z = (c.station_height_mm + c.object_height_mm) / 1000.0
        self._overlay = (time.monotonic(), ranks_px, rank)
        msg = f'rank {rank} -> base({x:.4f}, {y:.4f}, {z:.4f}) m，半徑 {radius:.3f} m；可見 rank {sorted(by_rank)}'
        return Target(int(rank), x, y, z, float(plane[i][0]), float(plane[i][1]), sorted(by_rank), msg), ''

    # ------------------------------------------------------------------ #
    def get_debug_image(self) -> Optional[np.ndarray]:
        return self._debug_img

    def _render(self, frame, squares: List[Blob], rejected: List[Blob], poly) -> None:
        if not self.show_debug:
            return
        img = frame.copy()
        if poly is not None:
            draw_poly(img, poly, (255, 128, 0), 2)
            put_text(img, 'A ROI', (poly[0][0], poly[0][1] - 6), (255, 128, 0), 0.5, 1)
        else:
            put_text(img, 'a_station_homography not calibrated: ROI = whole image', (8, 36), (0, 0, 255), 0.5, 1)
        t_ov, ranks, target = self._overlay
        fresh = time.monotonic() - t_ov < 8.0
        for s in squares:
            rk = next((r for (cx, cy, r) in ranks if abs(cx - s.cx) < 1 and abs(cy - s.cy) < 1), None) if fresh else None
            is_target = fresh and target is not None and rk == target
            col = (0, 0, 255) if is_target else (0, 255, 0)
            draw_poly(img, s.box, col, 3 if is_target else 2)
            put_text(img, f's={s.side:.0f}px' + (f' r{rk}' if rk else ''), (s.cx - 20, s.cy + 4), col, 0.5, 1)
        for s in rejected:
            draw_poly(img, s.box, (0, 0, 255), 1)
            put_text(img, 'x ' + s.reject_reason.split('；')[0], (s.cx - 20, s.cy), (0, 0, 255), 0.4, 1)
        put_text(img, f'squares={len(squares)} rejected={len(rejected)}', (8, 18), (255, 255, 255), 0.5, 1)
        self._debug_img = img


def main(args=None):
    rclpy.init(args=args)
    node = None
    code = 0
    try:
        node = Arm1VisionNode()
        executor = MultiThreadedExecutor(num_threads=7)   # 5 個 callback group + 餘裕，避免 service 等 client 回應時缺執行緒
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
