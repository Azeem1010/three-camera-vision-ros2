"""最小化的假 ROS 2 介面（rclpy / sensor_msgs / std_srvs / vision_interfaces），讓三個節點能在沒有 ROS 的
機器上跑端到端測試。

重點：模擬 rclpy 訊息欄位的「嚴格型別檢查」（float 欄位不能塞 int、bool 不能塞 numpy.bool_、
定長陣列長度必須吻合…），這類錯誤在真實 ROS 才會丟 AssertionError，這裡先抓出來。
它不模擬 executor/QoS；計時器與訂閱回呼由測試自己呼叫（見 test_nodes.py 的 Rig）。

用法：  import fake_ros; fake_ros.install()   # 必須在 import 節點模組之前
"""
from __future__ import annotations

import copy
import sys
import threading
import types
from types import SimpleNamespace


# --------------------------------------------------------------------------- #
# 嚴格型別的訊息類別
# --------------------------------------------------------------------------- #
def _check(t: str, v, name: str) -> None:
    def bad():
        raise AssertionError(f"The '{name}' field must be of type '{t}' (got {type(v).__name__}: {v!r})")

    scalar = {'bool': lambda x: isinstance(x, bool),
              'int': lambda x: isinstance(x, int) and not isinstance(x, bool),
              'float': lambda x: isinstance(x, float),
              'str': lambda x: isinstance(x, str)}
    if t in scalar:
        if not scalar[t](v):
            bad()
        return
    base, _, tail = t.partition('[')
    size = tail.rstrip(']')
    if not isinstance(v, (list, tuple)) or isinstance(v, str):
        bad()
    if size and len(v) != int(size):
        bad()
    if not all(scalar[base](x) for x in v):
        bad()


class _Msg:
    _fields: dict = {}

    def __init__(self, **kw):
        object.__setattr__(self, '_vals', {k: copy.deepcopy(d) for k, (t, d) in self._fields.items()})
        for k, v in kw.items():
            setattr(self, k, v)

    def __setattr__(self, k, v):
        if k not in self._fields:
            raise AttributeError(f'{type(self).__name__} 沒有欄位 {k}')
        _check(self._fields[k][0], v, k)
        self._vals[k] = v

    def __getattr__(self, k):
        try:
            return object.__getattribute__(self, '_vals')[k]
        except KeyError:
            raise AttributeError(k)


def _cls(name, fields):
    return type(name, (_Msg,), {'_fields': fields})


def _srv(name, req, res):
    return type(name, (), {'Request': _cls(name + '_Request', req), 'Response': _cls(name + '_Response', res)})


_OK = {'success': ('bool', False), 'message': ('str', '')}
SRV = {
    'RecalibrateZones': _srv('RecalibrateZones', {}, dict(_OK)),
    'GetZoneState': _srv('GetZoneState', {}, {
        'a_has_object': ('bool', False), 'b_has_object': ('bool', False), 'b_arm_clear': ('bool', False),
        'c_object_count': ('int', 0), **_OK}),
    'CaptureRanking': _srv('CaptureRanking', {}, {'count': ('int', 0), 'low_confidence': ('bool', False), **_OK}),
    'GetRank': _srv('GetRank', {'x_mm': ('float', 0.0), 'y_mm': ('float', 0.0), 'peek_only': ('bool', False)},
                    {'rank': ('int', 0), 'distance_mm': ('float', 0.0), **_OK}),
    'GetPickTarget': _srv('GetPickTarget', {'target_rank': ('int', 0), 'peek_only': ('bool', False)},
                          {'rank': ('int', 0), 'x': ('float', 0.0), 'y': ('float', 0.0), 'z': ('float', 0.0), **_OK}),
    'GetGridState': _srv('GetGridState', {}, {
        'occupied': ('bool[8]', [False] * 8), 'place_x': ('float[8]', [0.0] * 8),
        'place_y': ('float[8]', [0.0] * 8), 'place_z': ('float[8]', [0.0] * 8),
        'all_full': ('bool', False), **_OK}),
    'VerifyPlace': _srv('VerifyPlace', {'slot': ('int', 0)}, {
        'placed': ('bool', False), 'order_ok': ('bool', False), 'wrong_slots': ('int[]', []), **_OK}),
}
Trigger = _srv('Trigger', {}, dict(_OK))


class Image:
    def __init__(self, height=0, width=0, encoding='', step=0, data=b''):
        self.height, self.width, self.encoding, self.step, self.data = height, width, encoding, step, data


# --------------------------------------------------------------------------- #
# 假的 Node / Client / Future
# --------------------------------------------------------------------------- #
class FakeLogger:
    def __init__(self):
        self.records = []   # (level, text)

    def _log(self, lvl, msg, **kw):
        self.records.append((lvl, str(msg)))

    def info(self, msg, **kw): self._log('info', msg, **kw)
    def warning(self, msg, **kw): self._log('warning', msg, **kw)
    warn = warning
    def error(self, msg, **kw): self._log('error', msg, **kw)
    def fatal(self, msg, **kw): self._log('fatal', msg, **kw)
    def debug(self, msg, **kw): self._log('debug', msg, **kw)

    def text(self, level=None):
        return '\n'.join(t for lv, t in self.records if level in (None, lv))


class FakeFuture:
    def __init__(self):
        self._lock = threading.Lock()
        self._done = False
        self._cbs = []
        self._result = None
        self._exc = None

    def set_result(self, r):
        with self._lock:
            self._result, self._done = r, True
            cbs = list(self._cbs)
        for cb in cbs:
            cb(self)

    def set_exception(self, e):
        with self._lock:
            self._exc, self._done = e, True
            cbs = list(self._cbs)
        for cb in cbs:
            cb(self)

    def add_done_callback(self, cb):
        with self._lock:
            if not self._done:
                self._cbs.append(cb)
                return
        cb(self)

    def result(self): return self._result
    def exception(self): return self._exc


class FakeClient:
    """handler(request) -> response。call_async 在另一條執行緒完成 future，模擬「回應由別的 executor 執行緒處理」。"""

    def __init__(self, srv_type, name, group):
        self.srv_type, self.srv_name, self.group = srv_type, name, group
        self.handler = None
        self.removed = 0

    def service_is_ready(self): return self.handler is not None

    def call_async(self, req):
        fut = FakeFuture()

        def run():
            try:
                fut.set_result(self.handler(req))
            except Exception as e:  # noqa: BLE001
                fut.set_exception(e)
        threading.Thread(target=run, daemon=True).start()
        return fut

    def remove_pending_request(self, fut): self.removed += 1


class FakeNode:
    overrides: dict = {}    # 測試在建立節點前先放入 {節點名稱: {參數名: 值}}

    def __init__(self, node_name, automatically_declare_parameters_from_overrides=False, **kw):
        self.node_name = node_name
        self._params = dict(FakeNode.overrides.get(node_name, {}))
        self._logger = FakeLogger()
        self.subs, self.timers, self.services, self.clients = [], [], {}, {}

    def get_logger(self): return self._logger
    def list_parameters(self, prefixes, depth): return SimpleNamespace(names=list(self._params))
    def get_parameters(self, names): return [SimpleNamespace(name=n, value=self._params[n]) for n in names]
    def set_param(self, name, value): self._params[name] = value   # 模擬 ros2 param set

    def create_subscription(self, msg_type, topic, cb, qos, callback_group=None):
        self.subs.append((topic, cb))
        return object()

    def create_timer(self, period, cb, callback_group=None):
        self.timers.append((period, cb))
        return object()

    def create_service(self, srv_type, name, cb, callback_group=None):
        self.services[name] = cb
        return object()

    def create_client(self, srv_type, name, callback_group=None):
        c = FakeClient(srv_type, name, callback_group)
        self.clients[name] = c
        return c

    def destroy_node(self): pass


def install() -> None:
    """把假模組塞進 sys.modules（需在 import 節點模組之前呼叫）。"""
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__path__ = []   # 當成 package，允許子模組 import
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    mod('rclpy', init=lambda *a, **k: None, shutdown=lambda *a, **k: None, ok=lambda: True)
    mod('rclpy.node', Node=FakeNode)
    mod('rclpy.callback_groups', MutuallyExclusiveCallbackGroup=type('MutuallyExclusiveCallbackGroup', (), {}),
        ReentrantCallbackGroup=type('ReentrantCallbackGroup', (), {}))
    mod('rclpy.executors', MultiThreadedExecutor=type('MultiThreadedExecutor', (), {
        '__init__': lambda self, num_threads=1: None, 'add_node': lambda self, n: None,
        'spin': lambda self: None, 'shutdown': lambda self: None}))
    mod('rclpy.qos', qos_profile_sensor_data=object())
    mod('rclpy.utilities', remove_ros_args=lambda args=None: list(args or []))
    mod('sensor_msgs', msg=None)
    mod('sensor_msgs.msg', Image=Image)
    mod('std_srvs', srv=None)
    mod('std_srvs.srv', Trigger=Trigger)
    mod('vision_interfaces', srv=None)
    mod('vision_interfaces.srv', **SRV)
