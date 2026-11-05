"""三個視覺節點的端到端測試（用 tests/fake_ros.py 取代 rclpy，不需要安裝 ROS）。

做法：背景執行緒以 30 FPS 把合成影像餵進節點的影像回呼、以節點實際註冊的週期呼叫處理計時器，
然後直接呼叫 service 回呼，驗證回應內容、逾時與失敗處理。
fake_ros 會模擬 rclpy 訊息欄位的嚴格型別檢查，能抓到 float 欄位塞 int、numpy.bool_ 等真實 ROS 才會報的錯。
"""
import math
import threading
import time
from pathlib import Path

import numpy as np
import synthetic as S

import fake_ros

fake_ros.install()   # 必須在 import 節點模組之前

from fake_ros import SRV, FakeNode, Image, Trigger  # noqa: E402

from vision_pkg import arm1_vision_node as arm1_mod  # noqa: E402
from vision_pkg import arm2_vision_node as arm2_mod  # noqa: E402
from vision_pkg import calib_io  # noqa: E402
from vision_pkg import env_vision_node as env_mod  # noqa: E402
from vision_pkg.vision_core.geometry import apply_homography  # noqa: E402
from vision_pkg.vision_core.params_util import load_ros_params_yaml  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / 'config'


# --------------------------------------------------------------------------- #
# 測試輔助
# --------------------------------------------------------------------------- #
def to_msg(bgr: np.ndarray) -> Image:
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])   # usb_cam 的 mjpeg2rgb 輸出是 rgb8
    h, w = rgb.shape[:2]
    return Image(h, w, 'rgb8', w * 3, rgb.tobytes())


def wait_until(fn, timeout=6.0, step=0.05):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if fn():
            return True
        time.sleep(step)
    return False


class Rig:
    """模擬 executor：影像回呼（30 FPS）與處理計時器（節點註冊的週期）各一條執行緒。"""

    def __init__(self, node, scene, fps=30.0):
        self.node, self.fps = node, fps
        self.msg = to_msg(scene)
        self.paused = False
        self._stop = threading.Event()
        topic, cb = node.subs[0]
        period, tick = node.timers[0]
        self._threads = [threading.Thread(target=self._feed, args=(cb,), daemon=True),
                         threading.Thread(target=self._tick, args=(tick, period), daemon=True)]

    def _feed(self, cb):
        while not self._stop.is_set():
            if not self.paused:
                cb(self.msg)
            time.sleep(1.0 / self.fps)

    def _tick(self, tick, period):
        while not self._stop.is_set():
            tick()
            time.sleep(period)

    def set_scene(self, scene):
        self.msg = to_msg(scene)

    def start(self):
        for t in self._threads:
            t.start()
        return self

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)


def make_node(cls, name, yaml_name, tmp_path, **over):
    flat = load_ros_params_yaml(CONFIG / yaml_name)
    flat.update({'show_debug': False, 'calib_dir': str(tmp_path), 'service_timeout_s': 1.0,
                 'image_timeout_s': 0.5})
    flat.update(over)
    FakeNode.overrides[name] = flat
    return cls()


def call(node, svc, req=None, **fields):
    """呼叫節點上註冊的 service 回呼，回傳 response。"""
    cb = node.services[svc]
    kind = {'recalibrate_zones': 'RecalibrateZones', 'get_zone_state': 'GetZoneState',
            'capture_ranking': 'CaptureRanking', 'get_rank': 'GetRank', 'get_pick_target': 'GetPickTarget',
            'get_grid_state': 'GetGridState', 'verify_place': 'VerifyPlace'}.get(svc.rsplit('/', 1)[-1])   # 其餘（clear_taken）為 Trigger
    srv = Trigger if kind is None else SRV[kind]
    r = req if req is not None else srv.Request(**fields)
    return cb(r, srv.Response())


def save_h(tmp_path, name, H, unit):
    px = np.float64([[0, 0], [100, 0], [100, 100], [0, 100]])
    calib_io.save_homography(Path(tmp_path) / f'{name}.yaml', H, unit, (640, 480), px,
                             apply_homography(H, px), 0.0, False)


def _env(tmp_path, **over):
    return make_node(env_mod.EnvVisionNode, 'env_vision_node', 'env_vision.yaml', tmp_path, **over)


def _zs(node):
    return call(node, '/vision/env/get_zone_state')


# --------------------------------------------------------------------------- #
# 環境相機節點
# --------------------------------------------------------------------------- #
def test_env_boot_locates_zones_saves_file_and_reports_state(tmp_path, env_scenes):
    node = _env(tmp_path)
    rig = Rig(node, env_scenes['empty']).start()
    try:
        assert wait_until(lambda: len(node._zones) == 3 and node._boot_done)
        assert all(z.source == 'detected' for z in node._zones.values())
        assert (tmp_path / 'zones.yaml').exists()
        for z in 'abc':
            assert np.abs(node._zones[z].corners - S.zone_corners_img(z)).max() < 3.0
        # 手臂預設視為「在場」，要連續幾次確認 clear 才放行（安全優先）
        assert wait_until(lambda: _zs(node).b_arm_clear, 4)
        r = _zs(node)
        assert r.success, r.message
        assert (r.a_has_object, r.b_has_object, r.c_object_count) == (False, False, 0)
        assert any(' 區定位成功' in t for _, t in node.get_logger().records)
    finally:
        rig.stop()


def test_env_caps_and_arm_are_reported(tmp_path, env_scenes):
    node = _env(tmp_path)
    rig = Rig(node, env_scenes['empty']).start()
    try:
        assert wait_until(lambda: _zs(node).b_arm_clear, 6)
        rig.set_scene(env_scenes['c3'])
        assert wait_until(lambda: _zs(node).a_has_object and _zs(node).c_object_count == 3, 4)
        assert not _zs(node).b_has_object

        rig.set_scene(S.make_env_scene(b_caps=[(300, 150)]))       # 瓶蓋在 B 區
        assert wait_until(lambda: _zs(node).b_has_object and not _zs(node).a_has_object, 4)
        assert _zs(node).b_arm_clear

        rig.set_scene(env_scenes['b_arm'])   # 手臂進入 B 區：偵測到後立刻回報（不經 clear 去彈跳）
        assert wait_until(lambda: not _zs(node).b_arm_clear, 3)
        assert wait_until(lambda: not _zs(node).b_has_object, 3)   # 手臂不是物件（物件旗標有去彈跳，晚幾格才清掉）
        assert not _zs(node).b_arm_clear
        rig.set_scene(env_scenes['empty'])   # 離開：需連續確認才回報 clear
        assert wait_until(lambda: _zs(node).b_arm_clear, 6)
    finally:
        rig.stop()


def test_env_ranking_services_are_kept_but_do_not_rank(tmp_path, env_scenes):
    node = _env(tmp_path)
    pub = node.pubs['/vision/env/ranking']
    assert pub.qos.durability == 'transient_local'
    m = pub.last()
    assert not m.valid and '不排序' in m.message and list(m.rank) == []
    rig = Rig(node, env_scenes['a8']).start()
    try:
        assert wait_until(lambda: len(node._zones) == 3 and node._boot_done)
        r = call(node, '/vision/env/capture_ranking')
        assert r.success and r.count == 8 and not r.low_confidence and '不排序' in r.message, r.message
        rig.set_scene(env_scenes['c3'])
        time.sleep(0.2)
        r = call(node, '/vision/env/capture_ranking')
        assert r.success and r.count == 5 and '預期 8' in r.message      # 數量不符只提醒、不失敗
        g = call(node, '/vision/env/get_rank', x_mm=10.0, y_mm=10.0)
        assert g.success and g.rank == 0 and g.distance_mm == 0.0
        assert call(node, '/vision/env/clear_taken').success
    finally:
        rig.stop()


def test_env_services_time_out_cleanly_when_images_stop(tmp_path, env_scenes):
    node = _env(tmp_path, service_timeout_s=0.4)
    rig = Rig(node, env_scenes['empty']).start()
    try:
        assert wait_until(lambda: _zs(node).b_arm_clear, 6)
        rig.paused = True                       # 影像中斷
        time.sleep(0.7)
        t0 = time.time()
        r = _zs(node)
        assert not r.success and '過期' in r.message and not r.b_arm_clear
        assert time.time() - t0 < 1.5           # 守住逾時，不會卡死
        c = call(node, '/vision/env/capture_ranking')
        assert not c.success and '逾時' in c.message
        rc = call(node, '/vision/env/recalibrate_zones')
        assert not rc.success and '逾時' in rc.message
    finally:
        rig.stop()


def test_env_recalibrate_and_fallback_to_saved_zones(tmp_path, env_scenes):
    node = _env(tmp_path)
    rig = Rig(node, env_scenes['empty']).start()
    try:
        assert wait_until(lambda: len(node._zones) == 3 and node._boot_done)
        r = call(node, '/vision/env/recalibrate_zones')
        assert r.success, r.message
        assert 'A 區定位成功' in r.message and 'C 區定位成功' in r.message
        # 場地被移走（只剩桌面）：重新定位失敗，沿用舊 ROI 並說明原因
        blank = env_scenes['empty'].copy()
        blank[:] = S.DESK
        rig.set_scene(blank)
        time.sleep(0.3)
        r = call(node, '/vision/env/recalibrate_zones')
        assert not r.success and '沿用舊 ROI' in r.message
        assert len(node._zones) == 3
    finally:
        rig.stop()

    # 新節點：沒有影像可用 -> 讀上次存檔
    node2 = _env(tmp_path, boot_wait_s=0.2)
    time.sleep(0.3)
    node2._on_tick()
    assert node2._boot_done and len(node2._zones) == 3
    assert all(z.source == 'loaded' for z in node2._zones.values())
    assert any('上次存檔' in t for _, t in node2.get_logger().records)
    r = _zs(node2)    # 沒有影像：回 false 並說明，不崩潰
    assert not r.success and r.message


def test_env_without_any_saved_zones_reports_clear_error(tmp_path):
    node = _env(tmp_path, boot_wait_s=0.1, service_timeout_s=0.3)
    time.sleep(0.2)
    node._on_tick()
    assert node._boot_done and not node._zones
    for svc in ('/vision/env/get_zone_state', '/vision/env/capture_ranking'):
        r = call(node, svc)
        assert not r.success and r.message


def test_env_live_parameter_update_and_bad_value_is_rejected(tmp_path):
    node = _env(tmp_path)
    assert node.cfg.occ.object_ratio_thresh == 0.002
    node.set_param('occupancy.object_ratio_thresh', 0.01)
    node._refresh_cfg(time.monotonic() + 10)
    assert node.cfg.occ.object_ratio_thresh == 0.01
    node.set_param('field.min_aspect', 'abc')
    node._refresh_cfg(time.monotonic() + 20)
    assert node.cfg.occ.object_ratio_thresh == 0.01
    assert any('參數更新無效' in t for _, t in node.get_logger().records)


def test_env_missing_parameter_is_reported_with_its_name(tmp_path):
    flat = load_ros_params_yaml(CONFIG / 'env_vision.yaml')
    del flat['object_color']
    FakeNode.overrides['env_vision_node'] = flat
    try:
        env_mod.EnvVisionNode()
        assert False, '應該要丟 KeyError'
    except KeyError as e:
        assert 'object_color' in str(e)


def test_env_zone_state_topic_matches_service_and_reports_image_loss(tmp_path, env_scenes):
    node = _env(tmp_path)
    pub = node.pubs['/vision/env/zone_state']
    rig = Rig(node, env_scenes['empty']).start()
    try:
        assert wait_until(lambda: pub.last() is not None and pub.last().valid and pub.last().b_arm_clear, 6)
        rig.set_scene(env_scenes['c3'])
        assert wait_until(lambda: pub.last().a_has_object and pub.last().c_object_count == 3, 4)
        m = pub.last()
        assert m.header.frame_id == 'env_cam' and not m.b_has_object
        r = _zs(node)
        assert (r.a_has_object, r.b_has_object, r.b_arm_clear, r.c_object_count) == \
               (m.a_has_object, m.b_has_object, m.b_arm_clear, m.c_object_count)

        rig.paused = True                                   # 影像中斷：topic 改發 valid=false，且保守回報手臂在場
        assert wait_until(lambda: not pub.last().valid, 3)
        m = pub.last()
        assert '影像中斷' in m.message and not m.b_arm_clear
    finally:
        rig.stop()


# --------------------------------------------------------------------------- #
# 手眼相機 1 節點（直接拿環境相機的合成影像當手眼相機畫面）
# --------------------------------------------------------------------------- #
T1 = np.array([[0.001, 0, 0.10], [0, 0.001, 0.05], [0, 0, 1.0]])   # A 區平面 mm -> 手臂 1 基座 m


def _base_of(field_xy):
    """場地座標 mm -> 期望的手臂 1 基座座標 m。"""
    return 0.10 + (field_xy[0] - S.A_X0_MM) / 1000.0, 0.05 + field_xy[1] / 1000.0


def _arm1(tmp_path, scene, with_h=True, **over):
    if with_h:
        H_plane = np.linalg.inv(S.a_plane_to_img())     # 像素 -> A 區平面 mm
        save_h(tmp_path, 'a_station_homography', H_plane, 'mm')
        save_h(tmp_path, 'arm1_homography', T1 @ H_plane, 'm')
    node = make_node(arm1_mod.Arm1VisionNode, 'arm1_vision_node', 'arm1_vision.yaml', tmp_path,
                     **{'cap.min_area_px': 100.0, **over})   # 合成圖裡的瓶蓋比真實手眼相機看到的小
    return node, Rig(node, scene).start()


def _pick(arm1, **fields):
    return call(arm1, '/vision/arm1/get_pick_target', **fields)


def _publish(node, topic):
    """手動觸發節點的發布計時器（Rig 只跑處理計時器 timers[0]），回傳剛發布的訊息。"""
    node.timers[1][1]()
    return node.pubs[topic].msgs[-1]


def test_arm1_picks_nearest_reachable_cap_and_echoes_rank(tmp_path, env_scenes):
    arm1, rig = _arm1(tmp_path, env_scenes['a8'])
    try:
        assert wait_until(lambda: _pick(arm1).success, 4)
        r = _pick(arm1, target_rank=3)
        ex, ey = _base_of(S.A_CAPS_MM[0])                  # (440,60)：離基座最近
        assert r.success and r.rank == 3, r.message
        assert abs(r.x - ex) < 0.004 and abs(r.y - ey) < 0.004 and abs(r.z - 0.01) < 1e-9
        assert '看得到 8 個' in r.message
        assert _pick(arm1, target_rank=0).rank == 0
        assert not _pick(arm1, target_rank=-1).success
    finally:
        rig.stop()


def test_arm1_prefers_a_cap_with_clearance(tmp_path):
    crowded = [(430, 30), (460, 30)]          # 兩顆中心相距 30 mm
    lone = (500, 150)
    arm1, rig = _arm1(tmp_path, S.make_env_scene(a_caps=crowded + [lone]), **{'pick.min_clearance_m': 0.035})
    try:
        assert wait_until(lambda: _pick(arm1).success, 4)
        r = _pick(arm1)
        ex, ey = _base_of(lone)
        assert abs(r.x - ex) < 0.004 and abs(r.y - ey) < 0.004, r.message
        arm1.set_param('pick.min_clearance_m', 0.0)       # 不要求淨空 -> 選最近的
        arm1._refresh_cfg(time.monotonic() + 10)
        r = _pick(arm1)
        ex, ey = _base_of(crowded[0])
        assert abs(r.x - ex) < 0.004 and abs(r.y - ey) < 0.004, r.message
    finally:
        rig.stop()


def test_arm1_reports_reach_and_empty_failures(tmp_path, env_scenes):
    arm1, rig = _arm1(tmp_path, env_scenes['a8'])
    try:
        assert wait_until(lambda: _pick(arm1).success, 4)
        arm1.set_param('reach.max_m', 0.1)
        arm1._refresh_cfg(time.monotonic() + 10)
        r = _pick(arm1)
        assert not r.success and '超出可達範圍' in r.message
        arm1.set_param('reach.max_m', 0.35)
        arm1._refresh_cfg(time.monotonic() + 20)
        rig.set_scene(env_scenes['empty'])                 # A 區沒東西
        assert wait_until(lambda: '沒有偵測到瓶蓋' in _pick(arm1).message, 3)
    finally:
        rig.stop()


def test_arm1_topics(tmp_path, env_scenes):
    arm1, rig = _arm1(tmp_path, env_scenes['a8'])
    try:
        assert wait_until(lambda: _publish(arm1, '/vision/arm1/pick_target').success)
        m = _publish(arm1, '/vision/arm1/pick_target')
        ex, ey = _base_of(S.A_CAPS_MM[0])
        assert m.header.frame_id == 'arm1_base' and m.rank == 0
        assert abs(m.x - ex) < 0.004 and abs(m.y - ey) < 0.004

        det = arm1.pubs['/vision/arm1/detections']
        assert wait_until(lambda: det.last() is not None and det.last().valid and det.last().count == 8)
        d = det.last()
        assert d.plane_valid and d.base_valid and len(d.plane_x_mm) == 8 and len(d.base_x) == 8
        got = sorted(zip(np.round(d.plane_x_mm), np.round(d.plane_y_mm)))
        exp = sorted((x - S.A_X0_MM, y) for x, y in S.A_CAPS_MM)
        assert all(math.hypot(g[0] - e[0], g[1] - e[1]) < 4 for g, e in zip(got, exp))
        assert all(16 < s < 23 for s in d.side_px)        # 直徑約 19 px
    finally:
        rig.stop()


def test_arm1_without_homography_or_images_fails_gracefully(tmp_path, env_scenes):
    arm1, rig = _arm1(tmp_path, env_scenes['a8'], with_h=False, service_timeout_s=0.3)
    rig.paused = True
    try:
        r = _pick(arm1)                                           # 完全沒有影像
        assert not r.success and r.message
        m = _publish(arm1, '/vision/arm1/pick_target')
        assert not m.success and m.message
        rig.paused = False
        assert wait_until(lambda: arm1.detections.peek()[0] is not None)
        r = _pick(arm1)                                           # 有影像但 homography 還是範本（valid=false）
        assert not r.success and '尚未標定' in r.message
        d = arm1.pubs['/vision/arm1/detections'].last()
        assert d.valid and not d.plane_valid and not d.base_valid and d.count == 8   # ROI = 整張影像
    finally:
        rig.stop()


# --------------------------------------------------------------------------- #
# 手眼相機 2 節點
# --------------------------------------------------------------------------- #
T2 = np.array([[0.001, 0, 0.20], [0, 0.001, -0.10], [0, 0, 1.0]])   # 場地 mm -> 手臂 2 基座 m


def _arm2(tmp_path, scene, with_h=True, **over):
    if with_h:
        save_h(tmp_path, 'arm2_homography', T2 @ np.linalg.inv(S.arm2_field_to_img()), 'm')
    node = make_node(arm2_mod.Arm2VisionNode, 'arm2_vision_node', 'arm2_vision.yaml', tmp_path, **over)
    return node, Rig(node, scene).start()


def _valid(node):
    return node.results.peek()[0] is not None and node.results.peek()[0].valid


def test_arm2_grid_state_gives_occupancy_and_place_coordinates(tmp_path):
    node, rig = _arm2(tmp_path, S.make_arm2_scene([1, 2, 3]))
    try:
        assert wait_until(lambda: _valid(node))
        r = call(node, '/vision/arm2/get_grid_state')
        assert r.success, r.message
        assert list(r.occupied) == [True, True, True, False, False, False, False, False]
        assert not r.all_full
        for slot in (1, 2, 8):   # 格位中心 = 方孔中心 -> 基座 = 場地 mm / 1000 + (0.20, -0.10)
            x, y = S.SLOT_CENTERS_MM[slot - 1]
            assert abs(r.place_x[slot - 1] - (0.20 + x / 1000)) < 0.003
            assert abs(r.place_y[slot - 1] - (-0.10 + y / 1000)) < 0.003
        assert all(abs(z - 0.01) < 1e-9 for z in r.place_z)
    finally:
        rig.stop()


def test_arm2_verify_place_checks_fill_order(tmp_path):
    node, rig = _arm2(tmp_path, S.make_arm2_scene([1, 2]))
    try:
        assert wait_until(lambda: _valid(node))
        v = call(node, '/vision/arm2/verify_place', slot=2)
        assert v.success and v.placed and v.order_ok and list(v.wrong_slots) == [], v.message
        assert '下一格 3' in v.message
        v = call(node, '/vision/arm2/verify_place', slot=3)
        assert v.success and not v.placed and v.order_ok

        rig.set_scene(S.make_arm2_scene([1, 2, 4]))         # 跳過 3 號直接放 4 號
        assert wait_until(lambda: not call(node, '/vision/arm2/verify_place', slot=4).order_ok, 4)
        v = call(node, '/vision/arm2/verify_place', slot=4)
        assert v.success and v.placed and list(v.wrong_slots) == [4] and '跳號' in v.message

        bad = call(node, '/vision/arm2/verify_place', slot=9)
        assert not bad.success and '1..8' in bad.message
    finally:
        rig.stop()


def test_arm2_rotated_camera(tmp_path):
    filled = [1, 4, 6]
    node, rig = _arm2(tmp_path, S.make_arm2_scene(filled, rotation_k=1), with_h=False, view_rotation_k=1)
    try:
        assert wait_until(lambda: _valid(node))
        assert node.results.peek()[0].occupied == [s in filled for s in range(1, 9)]
        r = call(node, '/vision/arm2/get_grid_state')   # 沒有 homography：給占用但 success=false
        assert not r.success and '尚未標定' in r.message and list(r.occupied) == [s in filled for s in range(1, 9)]
        node.set_param('view_rotation_k', 0)              # 設錯方向：回報失敗而不是給錯的編號
        node._refresh_cfg(time.monotonic() + 10)
        assert wait_until(lambda: not _valid(node), 3)
        assert 'view_rotation_k' in node.results.peek()[0].message
    finally:
        rig.stop()


def test_arm2_holds_holes_briefly_when_covered_then_fails(tmp_path):
    node, rig = _arm2(tmp_path, S.make_arm2_scene([1]), hole_hold_s=0.5, service_timeout_s=0.4)
    try:
        assert wait_until(lambda: _valid(node))
        rig.set_scene(S.make_arm2_scene([1], cover_slot=5))       # 夾爪遮住孔 -> 短時間內沿用孔位
        assert wait_until(lambda: node.results.peek()[0].held, 2)
        r = call(node, '/vision/arm2/get_grid_state')
        assert r.success and '沿用' in r.message and r.occupied[0]
        assert wait_until(lambda: not call(node, '/vision/arm2/get_grid_state').success, 3)
        r = call(node, '/vision/arm2/get_grid_state')
        assert '方孔定位失敗' in r.message and list(r.occupied) == [False] * 8

        rig.set_scene(S.make_arm2_scene([1]))
        assert wait_until(lambda: call(node, '/vision/arm2/get_grid_state').success, 4)
        rig.paused = True                                         # 影像中斷
        time.sleep(0.7)
        r = call(node, '/vision/arm2/get_grid_state')
        assert not r.success and '過期' in r.message
    finally:
        rig.stop()


def test_arm2_topic_matches_service(tmp_path):
    node, rig = _arm2(tmp_path, S.make_arm2_scene([1, 3]))
    pub = node.pubs['/vision/arm2/grid_state']
    try:
        assert wait_until(lambda: pub.last() is not None and pub.last().valid)
        m = pub.last()
        r = call(node, '/vision/arm2/get_grid_state')
        assert list(m.occupied) == list(r.occupied) == [True, False, True] + [False] * 5
        assert m.header.frame_id == 'arm2_base' and m.place_valid
        assert np.allclose(m.place_x, r.place_x, atol=1e-3)
        assert m.order_checked and not m.order_ok and list(m.wrong_slots) == [3]
        assert all(math.isnan(s) for s in m.side_mm)
        assert m.fill_ratio[0] > 0.4 and m.fill_ratio[1] < 0.02
    finally:
        rig.stop()


def test_arm2_bad_config_is_rejected_at_startup(tmp_path):
    try:
        make_node(arm2_mod.Arm2VisionNode, 'arm2_vision_node', 'arm2_vision.yaml', tmp_path,
                  **{'hole.margin_ratio': 0.6})
        assert False, '應該要丟 ValueError'
    except ValueError as e:
        assert 'margin_ratio' in str(e)


# --------------------------------------------------------------------------- #
# debug 視窗的繪圖路徑（show_debug=true）
# --------------------------------------------------------------------------- #
def test_debug_images_are_produced_when_show_debug_is_true(tmp_path, env_scenes):
    node = _env(tmp_path, show_debug=True)
    rig = Rig(node, env_scenes['c3']).start()
    try:
        assert wait_until(lambda: node.get_debug_image() is not None and len(node._zones) == 3)
        time.sleep(0.3)
        img = node.get_debug_image()
        assert img.shape == (480, 640, 3) and img.dtype == np.uint8
        assert not np.array_equal(img, env_scenes['c3'])              # 確實畫了東西
    finally:
        rig.stop()

    arm1, rig = _arm1(tmp_path, env_scenes['a8'], show_debug=True)
    try:
        assert wait_until(lambda: _pick(arm1).success)
        time.sleep(0.2)
        assert arm1.get_debug_image().shape == (480, 640, 3)
    finally:
        rig.stop()

    arm2, rig = _arm2(tmp_path, S.make_arm2_scene([1]), show_debug=True)
    try:
        assert wait_until(lambda: arm2.get_debug_image() is not None and _valid(arm2))
        assert arm2.get_debug_image().shape == (480, 640, 3)
    finally:
        rig.stop()
