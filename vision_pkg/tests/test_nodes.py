"""三個視覺節點的端到端測試（用 tests/fake_ros.py 取代 rclpy，不需要安裝 ROS）。

做法：背景執行緒以 30 FPS 把合成影像餵進節點的影像回呼、以節點實際註冊的週期呼叫處理計時器，
然後直接呼叫 service 回呼，驗證回應內容、逾時與失敗處理。
fake_ros 會模擬 rclpy 訊息欄位的嚴格型別檢查，能抓到 float 欄位塞 int、numpy.bool_ 等真實 ROS 才會報的錯。
"""
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
from vision_pkg.vision_core.geometry import apply_homography, plane_homography  # noqa: E402
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


def make_node(cls, name, yaml_name, tmp_path, legacy=True, **over):
    flat = load_ros_params_yaml(CONFIG / yaml_name)
    if yaml_name == 'env_vision.yaml' and legacy:
        flat.update(S.LEGACY_ENV_PARAMS)   # 舊合成場景是三塊彩色站台；legacy=False 用 YAML 原樣（field 模式）
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


# --------------------------------------------------------------------------- #
# 環境相機節點
# --------------------------------------------------------------------------- #
def test_env_boot_locates_zones_saves_file_and_reports_state(tmp_path, env_scenes):
    node = make_node(env_mod.EnvVisionNode, 'env_vision_node', 'env_vision.yaml', tmp_path)
    rig = Rig(node, env_scenes['empty']).start()
    try:
        assert wait_until(lambda: len(node._zones) == 3 and node._boot_done)
        assert all(z.source == 'detected' for z in node._zones.values())
        assert (tmp_path / 'zones.yaml').exists()
        # 手臂預設視為「在場」，要連續幾次確認 clear 才放行（安全優先）
        assert wait_until(lambda: call(node, '/vision/env/get_zone_state').b_arm_clear, 4)
        r = call(node, '/vision/env/get_zone_state')
        assert r.success, r.message
        assert (r.a_has_object, r.b_has_object, r.c_object_count) == (False, False, 0)
        assert any(' 區定位成功' in t for _, t in node.get_logger().records)
    finally:
        rig.stop()


def test_env_objects_and_arm_are_reported(tmp_path, env_scenes):
    node = make_node(env_mod.EnvVisionNode, 'env_vision_node', 'env_vision.yaml', tmp_path)
    rig = Rig(node, env_scenes['empty']).start()
    try:
        assert wait_until(lambda: call(node, '/vision/env/get_zone_state').b_arm_clear, 6)
        rig.set_scene(S.make_env_scene(a_objs=S.a_objects(), c_objs=[(40, 40, 30, 0), (150, 60, 30, 20)]))
        assert wait_until(lambda: call(node, '/vision/env/get_zone_state').a_has_object, 4)
        r = call(node, '/vision/env/get_zone_state')
        assert r.success and r.a_has_object and not r.b_has_object and r.c_object_count == 2

        rig.set_scene(env_scenes['b_arm'])   # 手臂進入 B 區：偵測到後立刻回報（不經 clear 去彈跳）
        assert wait_until(lambda: not call(node, '/vision/env/get_zone_state').b_arm_clear, 3)
        assert not call(node, '/vision/env/get_zone_state').b_has_object
        rig.set_scene(env_scenes['empty'])   # 離開：需連續確認才回報 clear
        assert wait_until(lambda: call(node, '/vision/env/get_zone_state').b_arm_clear, 6)
    finally:
        rig.stop()


def test_env_ranking_get_rank_and_taken_marking(tmp_path, env_scenes):
    node = make_node(env_mod.EnvVisionNode, 'env_vision_node', 'env_vision.yaml', tmp_path)
    rig = Rig(node, env_scenes['empty']).start()
    try:
        assert wait_until(lambda: len(node._zones) == 3 and node._boot_done)
        rig.set_scene(env_scenes['a8'])
        r = call(node, '/vision/env/capture_ranking')
        assert r.success, r.message
        assert r.count == 8 and not r.low_confidence

        x, y = S.A_POSITIONS[S.A_PERM[2]]          # 第 3 小（28mm）物件的真實位置
        g = call(node, '/vision/env/get_rank', x_mm=float(x + 3), y_mm=float(y - 2), peek_only=True)
        assert g.success and g.rank == 3 and g.distance_mm < 5.0
        again = call(node, '/vision/env/get_rank', x_mm=float(x), y_mm=float(y), peek_only=True)
        assert again.success and again.rank == 3          # peek 不會標記

        g = call(node, '/vision/env/get_rank', x_mm=float(x), y_mm=float(y))   # 標記已取走
        assert g.success and g.rank == 3
        dup = call(node, '/vision/env/get_rank', x_mm=float(x), y_mm=float(y))
        assert not dup.success and '已被標記為取走' in dup.message and dup.rank == 0

        far = call(node, '/vision/env/get_rank', x_mm=100.0, y_mm=100.0)     # 中央沒有物件
        assert not far.success and '門檻' in far.message

        c = call(node, '/vision/env/clear_taken')
        assert c.success and '1' in c.message
        assert call(node, '/vision/env/get_rank', x_mm=float(x), y_mm=float(y), peek_only=True).success

        bad = call(node, '/vision/env/get_rank', x_mm=float('nan'), y_mm=1.0)
        assert not bad.success
    finally:
        rig.stop()


def test_env_capture_failure_does_not_guess_and_invalidates_old_snapshot(tmp_path, env_scenes):
    node = make_node(env_mod.EnvVisionNode, 'env_vision_node', 'env_vision.yaml', tmp_path)
    rig = Rig(node, env_scenes['a8']).start()
    try:
        assert wait_until(lambda: len(node._zones) == 3 and node._boot_done)
        assert call(node, '/vision/env/capture_ranking').success
        rig.set_scene(env_scenes['a7'])
        r = call(node, '/vision/env/capture_ranking')
        assert not r.success and r.count == 7 and '預期 8' in r.message
        x, y = S.A_POSITIONS[S.A_PERM[0]]
        g = call(node, '/vision/env/get_rank', x_mm=float(x), y_mm=float(y))
        assert not g.success and '尚未拍攝快照' in g.message
    finally:
        rig.stop()


def test_env_services_time_out_cleanly_when_images_stop(tmp_path, env_scenes):
    node = make_node(env_mod.EnvVisionNode, 'env_vision_node', 'env_vision.yaml', tmp_path,
                     **{'service_timeout_s': 0.4})
    rig = Rig(node, env_scenes['empty']).start()
    try:
        assert wait_until(lambda: call(node, '/vision/env/get_zone_state').b_arm_clear, 6)
        rig.paused = True                       # 影像中斷
        time.sleep(0.7)
        t0 = time.time()
        r = call(node, '/vision/env/get_zone_state')
        assert not r.success and '過期' in r.message and not r.b_arm_clear
        assert time.time() - t0 < 1.5           # 守住逾時，不會卡死
        c = call(node, '/vision/env/capture_ranking')
        assert not c.success and '逾時' in c.message
        rc = call(node, '/vision/env/recalibrate_zones')
        assert not rc.success and '逾時' in rc.message
    finally:
        rig.stop()


def test_env_recalibrate_and_fallback_to_saved_zones(tmp_path, env_scenes):
    node = make_node(env_mod.EnvVisionNode, 'env_vision_node', 'env_vision.yaml', tmp_path)
    rig = Rig(node, env_scenes['empty']).start()
    try:
        assert wait_until(lambda: len(node._zones) == 3 and node._boot_done)
        r = call(node, '/vision/env/recalibrate_zones')
        assert r.success, r.message
        assert 'A 區定位成功' in r.message and 'C 區定位成功' in r.message
        # 桌面被遮住（顏色全看不到）：重新定位失敗，沿用舊 ROI 並說明原因
        blank = S.make_env_scene()
        blank[:] = S.WOOD
        rig.set_scene(blank)
        time.sleep(0.3)
        r = call(node, '/vision/env/recalibrate_zones')
        assert not r.success and '沿用舊 ROI' in r.message
        assert len(node._zones) == 3
    finally:
        rig.stop()

    # 新節點：沒有影像可用 -> 讀上次存檔
    node2 = make_node(env_mod.EnvVisionNode, 'env_vision_node', 'env_vision.yaml', tmp_path,
                      **{'boot_wait_s': 0.2})
    time.sleep(0.3)
    node2._on_tick()
    assert node2._boot_done and len(node2._zones) == 3
    assert all(z.source == 'loaded' for z in node2._zones.values())
    assert any('上次存檔' in t for _, t in node2.get_logger().records)
    r = call(node2, '/vision/env/get_zone_state')    # 沒有影像：回 false 並說明，不崩潰
    assert not r.success and r.message


def test_env_without_any_saved_zones_reports_clear_error(tmp_path):
    node = make_node(env_mod.EnvVisionNode, 'env_vision_node', 'env_vision.yaml', tmp_path,
                     **{'boot_wait_s': 0.1, 'service_timeout_s': 0.3})
    time.sleep(0.2)
    node._on_tick()
    assert node._boot_done and not node._zones
    for svc in ('/vision/env/get_zone_state', '/vision/env/capture_ranking'):
        r = call(node, svc)
        assert not r.success and r.message


def test_env_live_parameter_update_and_bad_value_is_rejected(tmp_path, env_scenes):
    node = make_node(env_mod.EnvVisionNode, 'env_vision_node', 'env_vision.yaml', tmp_path)
    assert node.cfg.match_mm == 20.0
    node.set_param('match_threshold_mm', 5.0)
    node._refresh_cfg(time.monotonic() + 10)
    assert node.cfg.match_mm == 5.0
    node.set_param('square.max_aspect', 'abc')
    node._refresh_cfg(time.monotonic() + 20)
    assert node.cfg.match_mm == 5.0
    assert any('參數更新無效' in t for _, t in node.get_logger().records)


def test_env_missing_parameter_is_reported_with_its_name(tmp_path):
    flat = load_ros_params_yaml(CONFIG / 'env_vision.yaml')
    del flat['match_threshold_mm']
    FakeNode.overrides['env_vision_node'] = flat
    try:
        env_mod.EnvVisionNode()
        assert False, '應該要丟 KeyError'
    except KeyError as e:
        assert 'match_threshold_mm' in str(e)


def test_env_field_mode_end_to_end(tmp_path):
    """field 模式（YAML 預設）：黑色外框場地 -> 三區定位、占用/手臂判斷、A 區排序與座標配對。"""
    node = make_node(env_mod.EnvVisionNode, 'env_vision_node', 'env_vision.yaml', tmp_path, legacy=False)
    assert node.cfg.zone_size['a'] == (200.0, 300.0)
    rig = Rig(node, S.make_field_scene()).start()
    try:
        assert wait_until(lambda: len(node._zones) == 3 and node._boot_done)
        assert all(z.source == 'detected' for z in node._zones.values())
        assert wait_until(lambda: call(node, '/vision/env/get_zone_state').b_arm_clear, 6)
        r = call(node, '/vision/env/get_zone_state')
        assert r.success and (r.a_has_object, r.b_has_object, r.c_object_count) == (False, False, 0), r.message

        rig.set_scene(S.make_field_scene(a_objs=S.a_objects()))
        assert wait_until(lambda: call(node, '/vision/env/get_zone_state').a_has_object, 4)
        r = call(node, '/vision/env/capture_ranking')
        assert r.success and r.count == 8, r.message
        x, y = S.A_POSITIONS[S.A_PERM[2]]            # 第 3 小的物件；A 區平面座標要加上位移
        ox, oy = S.FIELD_A_OFFSET_MM
        g = call(node, '/vision/env/get_rank', x_mm=float(x + ox), y_mm=float(y + oy), peek_only=True)
        assert g.success and g.rank == 3 and g.distance_mm < 5.0, g.message

        rig.set_scene(S.make_field_scene(b_arm=True))
        assert wait_until(lambda: not call(node, '/vision/env/get_zone_state').b_arm_clear, 3)
    finally:
        rig.stop()


# --------------------------------------------------------------------------- #
# 手眼相機 1 節點（與環境相機節點串接）
# --------------------------------------------------------------------------- #
def _arm1_setup(tmp_path, env_scenes, **arm1_over):
    """env 節點先拍好快照；arm1 的 get_rank client 接到 env 節點的 service 回呼。"""
    H_plane = plane_homography(S.ENV_ZONES['a'], *S.ZONE_SIZE_MM)           # 像素 -> A 站平面 mm
    T = np.array([[0.001, 0, 0.10], [0, 0.001, 0.05], [0, 0, 1.0]])         # mm -> 手臂 1 基座 m（平移 0.10, 0.05）
    save_h(tmp_path, 'a_station_homography', H_plane, 'mm')
    save_h(tmp_path, 'arm1_homography', T @ H_plane, 'm')

    env = make_node(env_mod.EnvVisionNode, 'env_vision_node', 'env_vision.yaml', tmp_path)
    env_rig = Rig(env, env_scenes['a8']).start()
    assert wait_until(lambda: len(env._zones) == 3 and env._boot_done)
    assert call(env, '/vision/env/capture_ranking').success

    arm1 = make_node(arm1_mod.Arm1VisionNode, 'arm1_vision_node', 'arm1_vision.yaml', tmp_path,
                     **{'env_call_timeout_s': 0.5, 'square.min_area_px': 150.0,   # 合成圖裡的物件比真實手眼相機看到的小
                        **arm1_over})
    arm1.cli_rank.handler = lambda req: env.services['/vision/env/get_rank'](req, SRV['GetRank'].Response())
    arm1_rig = Rig(arm1, env_scenes['a8']).start()
    return env, env_rig, arm1, arm1_rig


def _pick(arm1, **fields):
    return call(arm1, '/vision/arm1/get_pick_target', **fields)


def test_arm1_picks_smallest_rank_with_base_coordinates(tmp_path, env_scenes):
    env, env_rig, arm1, arm1_rig = _arm1_setup(tmp_path, env_scenes)
    try:
        r = _pick(arm1, peek_only=True)
        assert r.success, r.message
        assert r.rank == 1
        x, y = S.A_POSITIONS[S.A_PERM[0]]                      # 最小（20mm）物件的真實位置
        assert abs(r.x - (0.10 + x / 1000.0)) < 0.004 and abs(r.y - (0.05 + y / 1000.0)) < 0.004
        assert abs(r.z - 0.03) < 1e-9                           # 站台高度 0 + 物件高度 30mm
        assert 'peek_only' in r.message

        r = _pick(arm1, target_rank=5, peek_only=True)
        x5, y5 = S.A_POSITIONS[S.A_PERM[4]]
        assert r.success and r.rank == 5 and abs(r.x - (0.10 + x5 / 1000.0)) < 0.004

        r = _pick(arm1, target_rank=9, peek_only=True)
        assert not r.success and '不在可見物件中' in r.message
        assert not _pick(arm1, target_rank=-1).success
    finally:
        arm1_rig.stop(); env_rig.stop()


def test_arm1_marks_taken_and_never_returns_the_same_object_twice(tmp_path, env_scenes):
    env, env_rig, arm1, arm1_rig = _arm1_setup(tmp_path, env_scenes)
    try:
        r = _pick(arm1)                                         # 預設：取走 rank 1 並通知 env 標記
        assert r.success and r.rank == 1
        # 物件還在畫面上（還沒真的被拿走）：env 說它已被取走 -> 不猜測，回報失敗
        r2 = _pick(arm1)
        assert not r2.success and '無法配對' in r2.message and '已被標記為取走' in r2.message
        r3 = _pick(arm1, target_rank=2)                         # 指定其他 rank 仍可取
        assert r3.success and r3.rank == 2

        # 物件被實際拿走（畫面上少了 rank 1、rank 2）-> 預設取剩下最小的 rank 3
        gone = S.make_env_scene(a_objs=S.a_objects(skip=[0, 1]))
        arm1_rig.set_scene(gone)
        time.sleep(0.4)
        r4 = _pick(arm1)
        assert r4.success and r4.rank == 3

        # 重試復原：清掉標記後，被誤標記的物件可再被查到
        call(env, '/vision/env/clear_taken')
        arm1_rig.set_scene(env_scenes['a8'])
        time.sleep(0.4)
        r5 = _pick(arm1, peek_only=True)
        assert r5.success and r5.rank == 1
    finally:
        arm1_rig.stop(); env_rig.stop()


def test_arm1_reports_reach_homography_and_env_failures(tmp_path, env_scenes):
    env, env_rig, arm1, arm1_rig = _arm1_setup(tmp_path, env_scenes)
    try:
        arm1.set_param('reach.max_m', 0.2)
        arm1._refresh_cfg(time.monotonic() + 10)
        r = _pick(arm1)
        assert not r.success and '超出可達範圍' in r.message
        # 超出可達範圍時不可標記為已取走：恢復範圍後仍可取到 rank 1
        arm1.set_param('reach.max_m', 0.35)
        arm1._refresh_cfg(time.monotonic() + 20)
        assert _pick(arm1).rank == 1

        arm1.cli_rank.handler = None                              # env 節點沒啟動
        r = _pick(arm1)
        assert not r.success and '未就緒' in r.message

        def slow(req):
            time.sleep(1.0)
            return SRV['GetRank'].Response()
        arm1.cli_rank.handler = slow                              # env 回應太慢
        t0 = time.time()
        r = _pick(arm1)
        assert not r.success and '逾時' in r.message and time.time() - t0 < 1.5
        assert arm1.cli_rank.removed >= 1                         # 逾時後有清掉 pending request

        arm1_rig.set_scene(S.make_env_scene())                    # A 區沒東西
        time.sleep(0.4)
        arm1.cli_rank.handler = lambda req: env.services['/vision/env/get_rank'](req, SRV['GetRank'].Response())
        r = _pick(arm1)
        assert not r.success and '沒有偵測到橘色正方形' in r.message
    finally:
        arm1_rig.stop(); env_rig.stop()


def _publish(node, topic):
    """手動觸發節點的發布計時器（Rig 只跑處理計時器 timers[0]），回傳剛發布的訊息。"""
    node.timers[1][1]()
    return node.publishers[topic].msgs[-1]


def test_arm1_topic_publishes_smallest_rank_without_marking_taken(tmp_path, env_scenes):
    env, env_rig, arm1, arm1_rig = _arm1_setup(tmp_path, env_scenes)
    try:
        assert wait_until(lambda: _publish(arm1, '/vision/arm1/pick_target').success)
        m = _publish(arm1, '/vision/arm1/pick_target')
        x, y = S.A_POSITIONS[S.A_PERM[0]]
        assert m.rank == 1 and m.header.frame_id == 'arm1_base'
        assert abs(m.x - (0.10 + x / 1000.0)) < 0.004 and abs(m.y - (0.05 + y / 1000.0)) < 0.004
        assert abs(m.z - 0.03) < 1e-9 and list(m.visible_ranks) == list(range(1, 9))
        assert _publish(arm1, '/vision/arm1/pick_target').rank == 1   # 一直發布也不會把 rank 1 標記成已取走

        assert _pick(arm1).rank == 1                                  # service 照舊：取走 rank 1 並標記
        m = _publish(arm1, '/vision/arm1/pick_target')                # rank 1 還在畫面上但已標記 -> 不猜測
        assert not m.success and '無法確定最小者' in m.message

        arm1_rig.set_scene(S.make_env_scene(a_objs=S.a_objects(skip=[0])))   # rank 1 被實際拿走
        assert wait_until(lambda: _publish(arm1, '/vision/arm1/pick_target').rank == 2, 4)
    finally:
        arm1_rig.stop(); env_rig.stop()


def test_arm1_topic_reports_failure_when_no_images(tmp_path):
    arm1 = make_node(arm1_mod.Arm1VisionNode, 'arm1_vision_node', 'arm1_vision.yaml', tmp_path)
    m = _publish(arm1, '/vision/arm1/pick_target')
    assert not m.success and m.message and m.rank == 0


def test_arm1_without_homography_or_images_fails_gracefully(tmp_path, env_scenes):
    arm1 = make_node(arm1_mod.Arm1VisionNode, 'arm1_vision_node', 'arm1_vision.yaml', tmp_path,
                     **{'service_timeout_s': 0.3})
    r = _pick(arm1)                                               # 完全沒有影像
    assert not r.success and r.message
    rig = Rig(arm1, env_scenes['a8']).start()
    try:
        assert wait_until(lambda: arm1.detections.peek()[0] is not None)
        r = _pick(arm1)                                           # 有影像但 homography 還是範本（valid=false）
        assert not r.success and '尚未標定' in r.message
    finally:
        rig.stop()


# --------------------------------------------------------------------------- #
# 手眼相機 2 節點
# --------------------------------------------------------------------------- #
ASC = [28, 30, 32, 34, 36, 38, 40, 42]


def _arm2_node(tmp_path, with_h=True, **over):
    if with_h:
        H = plane_homography(S.ARM2_C, *S.C_SIZE_MM)
        T = np.array([[0.001, 0, 0.20], [0, 0.001, -0.10], [0, 0, 1.0]])   # mm -> 手臂 2 基座 m
        save_h(tmp_path, 'arm2_homography', T @ H, 'm')
    return make_node(arm2_mod.Arm2VisionNode, 'arm2_vision_node', 'arm2_vision.yaml', tmp_path,
                     **{'c_station_size_mm': list(S.C_SIZE_MM), **over})


def test_arm2_grid_state_gives_occupancy_and_place_coordinates(tmp_path):
    node = _arm2_node(tmp_path)
    rig = Rig(node, S.make_arm2_scene([28, None, 32, 34, None, 38, 40, 42])).start()
    try:
        assert wait_until(lambda: node.results.peek()[0] is not None and node.results.peek()[0].valid)
        r = call(node, '/vision/arm2/get_grid_state')
        assert r.success, r.message
        assert list(r.occupied) == [True, False, True, True, False, True, True, True]
        assert not r.all_full
        # 格位 1 中心 = 平面 (25,25)mm、格位 8 中心 = (175,75)mm  -> 基座 = 平面/1000 + (0.20, -0.10)
        assert abs(r.place_x[0] - 0.225) < 0.004 and abs(r.place_y[0] - (-0.075)) < 0.004
        assert abs(r.place_x[7] - 0.375) < 0.004 and abs(r.place_y[7] - (-0.025)) < 0.004
        assert all(abs(z - 0.03) < 1e-9 for z in r.place_z)

        rig.set_scene(S.make_arm2_scene(ASC))
        assert wait_until(lambda: call(node, '/vision/arm2/get_grid_state').all_full, 4)
    finally:
        rig.stop()


def test_arm2_topic_matches_grid_state_service(tmp_path):
    node = _arm2_node(tmp_path)
    m = _publish(node, '/vision/arm2/grid_state')                 # 還沒有影像：照樣發布失敗與原因
    assert not m.success and m.message and list(m.occupied) == [False] * 8
    rig = Rig(node, S.make_arm2_scene([28, None, 32, 34, None, 38, 40, 42])).start()
    try:
        assert wait_until(lambda: _publish(node, '/vision/arm2/grid_state').success)
        m = _publish(node, '/vision/arm2/grid_state')
        r = call(node, '/vision/arm2/get_grid_state')
        assert m.header.frame_id == 'arm2_base'
        assert list(m.occupied) == list(r.occupied) == [True, False, True, True, False, True, True, True]
        assert all(abs(a - b) < 1e-6 for a, b in zip(m.place_x, r.place_x))
        assert abs(m.place_x[0] - 0.225) < 0.004 and not m.all_full
    finally:
        rig.stop()


def test_arm2_verify_place_order_check(tmp_path):
    node = _arm2_node(tmp_path)
    rig = Rig(node, S.make_arm2_scene(ASC)).start()
    try:
        assert wait_until(lambda: node.results.peek()[0] is not None and node.results.peek()[0].valid)
        r = call(node, '/vision/arm2/verify_place', slot=3)
        assert r.success and r.placed and r.order_ok and list(r.wrong_slots) == []

        rig.set_scene(S.make_arm2_scene([28, 30, 38, 34, 36, 32, 40, 42]))     # 格位 3 與 6 放錯
        assert wait_until(lambda: not call(node, '/vision/arm2/verify_place', slot=3).order_ok, 4)
        r = call(node, '/vision/arm2/verify_place', slot=3)
        assert r.success and r.placed and not r.order_ok and list(r.wrong_slots) == [3, 6]
        assert '3' in r.message and '6' in r.message

        rig.set_scene(S.make_arm2_scene([28, None, 32, None, None, 38, None, 42]))   # 部分放置，順序正確
        assert wait_until(lambda: not call(node, '/vision/arm2/verify_place', slot=2).placed, 4)
        r = call(node, '/vision/arm2/verify_place', slot=2)
        assert r.success and not r.placed and r.order_ok
        r = call(node, '/vision/arm2/verify_place', slot=3)
        assert r.success and r.placed

        for bad in (0, 9, -1):
            r = call(node, '/vision/arm2/verify_place', slot=bad)
            assert not r.success and '1..8' in r.message
    finally:
        rig.stop()


def test_arm2_refuses_to_guess_when_objects_touch(tmp_path):
    node = _arm2_node(tmp_path)
    rig = Rig(node, S.make_arm2_scene([42, 42, None, None, None, None, None, None],
                                      extra_objs=[(50, 25, 20, 0)])).start()
    try:
        assert wait_until(lambda: node.results.peek()[0] is not None and node.results.peek()[0].valid)
        r = call(node, '/vision/arm2/verify_place', slot=1)
        assert not r.success and '無法量測邊長' in r.message and '不猜測' in r.message
        assert r.placed       # 占用狀態仍如實回報
    finally:
        rig.stop()


def test_arm2_without_homography_still_reports_occupancy_but_not_success(tmp_path):
    node = _arm2_node(tmp_path, with_h=False)
    rig = Rig(node, S.make_arm2_scene(ASC)).start()
    try:
        assert wait_until(lambda: node.results.peek()[0] is not None and node.results.peek()[0].valid)
        r = call(node, '/vision/arm2/get_grid_state')
        assert not r.success and '尚未標定' in r.message
        assert list(r.occupied) == [True] * 8 and r.all_full
        assert call(node, '/vision/arm2/verify_place', slot=1).success    # 驗證放置不需要座標
    finally:
        rig.stop()


def test_arm2_zone_lost_and_image_timeout(tmp_path):
    node = _arm2_node(tmp_path, **{'corner_hold_s': 0.3, 'service_timeout_s': 0.4})
    rig = Rig(node, S.make_arm2_scene(ASC)).start()
    try:
        assert wait_until(lambda: node.results.peek()[0] is not None and node.results.peek()[0].valid)
        blank = S.make_arm2_scene(ASC)
        blank[:] = S.WOOD
        rig.set_scene(blank)
        assert wait_until(lambda: not call(node, '/vision/arm2/get_grid_state').success, 4)
        r = call(node, '/vision/arm2/get_grid_state')
        assert not r.success and 'C 區定位失敗' in r.message
        assert list(r.occupied) == [False] * 8

        rig.set_scene(S.make_arm2_scene(ASC))
        assert wait_until(lambda: call(node, '/vision/arm2/get_grid_state').success, 4)
        rig.paused = True                                               # 影像中斷
        time.sleep(0.7)
        r = call(node, '/vision/arm2/get_grid_state')
        assert not r.success and '過期' in r.message
    finally:
        rig.stop()


def test_arm2_rotation_parameter_aligns_slot_numbers(tmp_path):
    sides = [28, None, 32, 34, 36, None, 40, 42]
    node = _arm2_node(tmp_path, **{'warp_rotation_k': 2})
    rig = Rig(node, S.make_arm2_scene(sides, rotation_k=2)).start()
    try:
        assert wait_until(lambda: node.results.peek()[0] is not None and node.results.peek()[0].valid)
        r = call(node, '/vision/arm2/get_grid_state')
        assert list(r.occupied) == [s is not None for s in sides]
    finally:
        rig.stop()


def test_arm2_bad_grid_config_is_rejected_at_startup(tmp_path):
    try:
        _arm2_node(tmp_path, **{'grid.rows': 3, 'grid.cols': 3})
        assert False, '應該要丟 ValueError'
    except ValueError as e:
        assert '必須等於 8' in str(e)


# --------------------------------------------------------------------------- #
# debug 視窗的繪圖路徑（show_debug=true）
# --------------------------------------------------------------------------- #
def test_debug_images_are_produced_when_show_debug_is_true(tmp_path, env_scenes):
    node = make_node(env_mod.EnvVisionNode, 'env_vision_node', 'env_vision.yaml', tmp_path, show_debug=True)
    rig = Rig(node, env_scenes['a8']).start()
    try:
        assert wait_until(lambda: node.get_debug_image() is not None and len(node._zones) == 3)
        assert call(node, '/vision/env/capture_ranking').success      # 之後畫面上會有 rank 標示
        time.sleep(0.4)
        img = node.get_debug_image()
        assert img.shape == (480, 640, 3) and img.dtype == np.uint8
        assert not np.array_equal(img, env_scenes['a8'])              # 確實畫了東西
    finally:
        rig.stop()

    H = plane_homography(S.ENV_ZONES['a'], *S.ZONE_SIZE_MM)
    save_h(tmp_path, 'a_station_homography', H, 'mm')
    arm1 = make_node(arm1_mod.Arm1VisionNode, 'arm1_vision_node', 'arm1_vision.yaml', tmp_path, show_debug=True,
                     **{'square.min_area_px': 150.0})
    rig = Rig(arm1, env_scenes['a8']).start()
    try:
        assert wait_until(lambda: arm1.get_debug_image() is not None)
        assert arm1.get_debug_image().shape == (480, 640, 3)
    finally:
        rig.stop()

    arm2 = _arm2_node(tmp_path, show_debug=True)
    rig = Rig(arm2, S.make_arm2_scene(ASC)).start()
    try:
        assert wait_until(lambda: arm2.get_debug_image() is not None)
        img = arm2.get_debug_image()
        assert img.shape[1] == 640 and img.shape[0] > 480            # 上：原始影像；下：校正後的格位圖
    finally:
        rig.stop()
