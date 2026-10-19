"""三個視覺節點共用的輔助工具。

刻意不在模組載入時 import rclpy，讓這裡的純 Python 邏輯（影像轉換、影格緩衝、去彈跳…）
可在沒有 ROS 的環境單獨測試。三個節點檔案彼此獨立，只共用這個工具模組（不共用任何執行期狀態）。
"""
from __future__ import annotations

import array
import os
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .vision_core.params_util import dataclass_from_flat


# --------------------------------------------------------------------------- #
# sensor_msgs/Image -> BGR（不依賴 cv_bridge，避免 Jetson 上 cv_bridge / numpy ABI 問題）
# --------------------------------------------------------------------------- #
def image_msg_to_bgr(msg) -> np.ndarray:
    """支援 usb_cam 常見輸出：rgb8 / bgr8 / rgba8 / bgra8 / mono8 / yuyv。不支援時拋 ValueError。"""
    enc = (msg.encoding or '').lower()
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    if h <= 0 or w <= 0:
        raise ValueError('空影像')
    buf = np.frombuffer(msg.data, dtype=np.uint8)

    def rows(ch: int) -> np.ndarray:
        need = h * step
        if buf.size < need or step < w * ch:
            raise ValueError(f'影像資料長度異常（{buf.size} < {need}，step={step}）')
        return buf[:need].reshape(h, step)[:, :w * ch]

    if enc == 'bgr8':
        return np.ascontiguousarray(rows(3).reshape(h, w, 3))
    if enc == 'rgb8':
        return np.ascontiguousarray(rows(3).reshape(h, w, 3)[:, :, ::-1])
    if enc in ('bgra8', 'rgba8'):
        a = np.ascontiguousarray(rows(4).reshape(h, w, 4))
        return cv2.cvtColor(a, cv2.COLOR_BGRA2BGR if enc == 'bgra8' else cv2.COLOR_RGBA2BGR)
    if enc in ('mono8', '8uc1'):
        return cv2.cvtColor(np.ascontiguousarray(rows(1).reshape(h, w)), cv2.COLOR_GRAY2BGR)
    if enc in ('yuv422_yuy2', 'yuyv'):
        return cv2.cvtColor(np.ascontiguousarray(rows(2).reshape(h, w, 2)), cv2.COLOR_YUV2BGR_YUY2)
    if enc in ('yuv422', 'uyvy'):
        return cv2.cvtColor(np.ascontiguousarray(rows(2).reshape(h, w, 2)), cv2.COLOR_YUV2BGR_UYVY)
    raise ValueError(f'不支援的影像編碼 {msg.encoding!r}')


# --------------------------------------------------------------------------- #
# 執行緒安全的資料容器
# --------------------------------------------------------------------------- #
class FrameBuffer:
    """最近 N 張影格（含到達時間）。影像回呼只負責 push，處理與 service 在別的執行緒讀取。"""

    def __init__(self, maxlen: int = 8):
        self._cv = threading.Condition()
        self._dq: deque = deque(maxlen=max(1, int(maxlen)))
        self._seq = 0

    @property
    def maxlen(self) -> int:
        return self._dq.maxlen

    def push(self, bgr: np.ndarray) -> None:
        with self._cv:
            self._seq += 1
            self._dq.append((self._seq, time.monotonic(), bgr))
            self._cv.notify_all()

    def latest(self) -> Optional[Tuple[int, float, np.ndarray]]:
        """(序號, 到達時間 monotonic, 影像)；尚未收到影像回傳 None。"""
        with self._cv:
            return self._dq[-1] if self._dq else None

    def latest_n(self, n: int) -> List[Tuple[int, float, np.ndarray]]:
        with self._cv:
            return list(self._dq)[-int(n):]

    def age(self) -> Optional[float]:
        with self._cv:
            return None if not self._dq else time.monotonic() - self._dq[-1][1]

    def wait_fresh(self, n: int, timeout_s: float) -> List[np.ndarray]:
        """等待「呼叫之後」到達的 n 張新影格；逾時回傳目前已收到的（可能不足 n 張）。"""
        n = min(int(n), self.maxlen)
        deadline = time.monotonic() + float(timeout_s)
        with self._cv:
            # 用序號而不是時間判斷「新」：Windows 的 monotonic 解析度約 15 ms，呼叫前剛到的影格時間可能與呼叫時相同
            seq0 = self._seq
            while True:
                fresh = [f for (s, _, f) in self._dq if s > seq0]
                if len(fresh) >= n:
                    return fresh[-n:]
                remain = deadline - time.monotonic()
                if remain <= 0:
                    return fresh
                self._cv.wait(remain)


class LatestValue:
    """保存「最新結果」及其產生時間；service 可等待一個夠新的結果（含逾時）。"""

    def __init__(self):
        self._cv = threading.Condition()
        self._v: Any = None
        self._t: Optional[float] = None

    def set(self, v: Any) -> None:
        with self._cv:
            self._v, self._t = v, time.monotonic()
            self._cv.notify_all()

    def clear(self) -> None:
        with self._cv:
            self._v, self._t = None, None

    def peek(self) -> Tuple[Any, Optional[float]]:
        """(值, 年齡秒)；沒有值時 (None, None)。"""
        with self._cv:
            return self._v, (None if self._t is None else time.monotonic() - self._t)

    def wait_fresh(self, max_age_s: float, timeout_s: float) -> Tuple[Any, str]:
        """等到有年齡 <= max_age_s 的結果。逾時回傳 (None, 原因)。"""
        deadline = time.monotonic() + float(timeout_s)
        with self._cv:
            while True:
                now = time.monotonic()
                if self._v is not None and (now - self._t) <= max_age_s:
                    return self._v, ''
                remain = deadline - now
                if remain <= 0:
                    if self._v is None:
                        return None, f'等待 {timeout_s:.1f}s 仍無結果（尚未收到影像或站台尚未定位？）'
                    return None, f'最新結果已過期 {now - self._t:.1f}s（影像中斷？）'
                self._cv.wait(remain)


class Debouncer:
    """布林狀態去彈跳：新值需連續出現 n 次才切換，可分別設定切到 True / False 所需次數。

    initial=None 表示採用第一筆輸入；給定 True/False 則以該值起始（例如手臂預設「在場」較安全）。
    """

    def __init__(self, n_to_true: int, n_to_false: int, initial: Optional[bool] = None):
        self.n_to_true = max(1, int(n_to_true))
        self.n_to_false = max(1, int(n_to_false))
        self.state: Optional[bool] = initial
        self._count = 0

    def update(self, raw: bool) -> bool:
        raw = bool(raw)
        if self.state is None:
            self.state, self._count = raw, 0
            return raw
        if raw == self.state:
            self._count = 0
            return self.state
        self._count += 1
        if self._count >= (self.n_to_true if raw else self.n_to_false):
            self.state, self._count = raw, 0
        return self.state


# --------------------------------------------------------------------------- #
# 參數
# --------------------------------------------------------------------------- #
def _py(v: Any) -> Any:
    if hasattr(v, 'tolist'):
        return v.tolist()
    if isinstance(v, (array.array, tuple)):
        return list(v)
    return v


def read_ros_params(node) -> Dict[str, Any]:
    """把節點目前所有 ROS 參數攤成 {'a.b.c': value}（參數皆由 YAML 自動宣告，不在程式內給預設值）。"""
    names = node.list_parameters([], 0).names
    return {p.name: _py(p.value) for p in node.get_parameters(names)}


class FlatParams:
    """扁平參數字典的型別安全讀取；缺少參數時丟出指向 config/*.yaml 的明確錯誤。"""

    def __init__(self, flat: Dict[str, Any]):
        self.flat = flat

    def raw(self, name: str) -> Any:
        v = self.flat.get(name)
        if v is None:
            raise KeyError(f'缺少參數 {name}（請檢查 config/*.yaml）')
        return v

    def f(self, name: str) -> float:
        return float(self.raw(name))

    def i(self, name: str) -> int:
        return int(self.raw(name))

    def b(self, name: str) -> bool:
        v = self.raw(name)
        return v.strip().lower() in ('1', 'true', 'yes', 'on') if isinstance(v, str) else bool(v)

    def s(self, name: str) -> str:
        return str(self.raw(name))

    def fl(self, name: str) -> List[float]:
        return [float(x) for x in self.raw(name)]

    def group(self, cls, prefix: str):
        return dataclass_from_flat(cls, self.flat, prefix)


# --------------------------------------------------------------------------- #
# Topic 發布
# --------------------------------------------------------------------------- #
def stamp(node, msg, frame_id: str):
    """填入 header（stamp = 目前時間，frame_id = 相機名稱），回傳 msg 方便串接 publish。"""
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = frame_id
    return msg


def image_problem(latest, now: float, timeout_s: float) -> str:
    """影像來源異常的原因（尚未收到 / 已中斷）；正常時回傳空字串。latest = FrameBuffer.latest()。"""
    if latest is None:
        return '尚未收到影像'
    age = now - latest[1]
    return f'影像中斷 {age:.1f}s' if age > timeout_s else ''


# --------------------------------------------------------------------------- #
# Service 內呼叫另一個 service（不在回呼中 spin）
# --------------------------------------------------------------------------- #
def call_service(client, request, timeout_s: float) -> Tuple[Any, str]:
    """call_async + threading.Event 等待（含逾時），回傳 (response 或 None, 錯誤訊息)。

    前提：節點使用 MultiThreadedExecutor，且 client 與呼叫它的 service 伺服端放在不同的
    MutuallyExclusiveCallbackGroup，回應才能由另一條執行緒處理，不會死鎖。
    """
    name = getattr(client, 'srv_name', '<service>')
    deadline = time.monotonic() + float(timeout_s)
    while not client.service_is_ready():
        if time.monotonic() >= deadline:
            return None, f'服務 {name} 未就緒（對端節點沒啟動？）'
        time.sleep(0.05)
    fut = client.call_async(request)
    done = threading.Event()
    fut.add_done_callback(lambda _f: done.set())
    if not done.wait(max(0.05, deadline - time.monotonic())):
        try:
            client.remove_pending_request(fut)
        except Exception:
            pass
        return None, f'呼叫 {name} 逾時（{timeout_s:.1f}s）'
    exc = fut.exception()
    if exc is not None:
        return None, f'呼叫 {name} 失敗: {exc!r}'
    return fut.result(), ''


# --------------------------------------------------------------------------- #
# 執行迴圈（含 debug 視窗）
# --------------------------------------------------------------------------- #
def has_display() -> bool:
    if os.name == 'nt':
        return True
    return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))


def spin_node(rclpy_mod, node, executor, show_debug: bool, window: str, gui_hz: float = 15.0) -> None:
    """執行節點。

    show_debug=False 或沒有 DISPLAY：單純 executor.spin()。
    否則 executor 在背景執行緒 spin，主執行緒負責全部 cv2 GUI 呼叫
    （HighGUI 只能由單一執行緒使用）；視窗按 q / Esc 只會關閉視窗，節點照常運作。
    """
    log = node.get_logger()
    use_gui = bool(show_debug)
    if use_gui and not has_display():
        log.warning('找不到 DISPLAY / WAYLAND_DISPLAY，已自動停用 debug 視窗')
        use_gui = False
    if not use_gui:
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        return

    th = threading.Thread(target=executor.spin, name='executor_spin', daemon=True)
    th.start()
    wait_ms = max(1, int(1000.0 / gui_hz))
    try:
        while th.is_alive() and rclpy_mod.ok():
            img = node.get_debug_image() if use_gui else None
            if img is None:
                time.sleep(wait_ms / 1000.0)
                continue
            try:
                cv2.imshow(window, img)
                key = cv2.waitKey(wait_ms) & 0xFF
            except cv2.error as e:
                log.warning(f'無法開啟 debug 視窗，已停用: {e}')
                use_gui = False
                continue
            if key in (27, ord('q')):
                cv2.destroyWindow(window)
                use_gui = False
                log.info('debug 視窗已關閉（節點仍在運作）')
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
