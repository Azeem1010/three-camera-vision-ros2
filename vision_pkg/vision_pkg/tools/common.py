"""工具程式共用：影像來源（ROS topic 或靜態圖片）與共同命令列參數。

工具程式只「訂閱」usb_cam 的影像 topic（與正式節點一致），不會自己開 /dev/video。
加上 --image 可完全離線使用（不需要 ROS / 相機），方便在筆電上用靜態圖片練習。
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import cv2
import numpy as np

from vision_pkg import calib_io


def strip_ros_args(argv: List[str]) -> List[str]:
    """去掉 --ros-args 之後的參數，讓 argparse 不會被 ROS 的參數干擾。"""
    try:
        from rclpy.utilities import remove_ros_args
        return remove_ros_args(args=argv)
    except ImportError:
        return argv


def add_source_args(ap: argparse.ArgumentParser, default_camera: str) -> None:
    ap.add_argument('--camera', choices=list(calib_io.CAMERA_NAMES), default=default_camera,
                    help='要看哪一顆鏡頭（決定預設 topic /<camera>_cam/image_raw 與檔名）')
    ap.add_argument('--topic', default='', help='影像 topic（預設依 --camera）')
    ap.add_argument('--image', default='', help='改用靜態圖片，不連 ROS（離線練習用）')
    ap.add_argument('--calib-dir', default='', help='校正檔目錄（預設 $VISION_CALIB_DIR 或 ~/.ros/vision_calib）')
    ap.add_argument('--undistort', action='store_true', help='套用內參去畸變（須與節點的 undistort 設定一致）')
    ap.add_argument('--intrinsics', default='', help='內參 YAML（預設 <calib_dir>/<camera>_cam_intrinsics.yaml）')


def imread(path: str) -> np.ndarray:
    """讀圖（支援 Windows 中文路徑）。"""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f'讀不到影像: {path}')
    return img


def require_display() -> None:
    from vision_pkg.node_common import has_display
    if not has_display():
        sys.exit('找不到 DISPLAY：這個工具需要圖形視窗。請在有螢幕的桌面執行，或用 ssh -X / VNC。')


class ImageSource:
    """read() 回傳最新一張 BGR 影像（還沒有則回傳 None）。"""

    def read(self, wait_s: float = 0.03) -> Optional[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class FileSource(ImageSource):
    def __init__(self, path: str, undistorter=None):
        img = imread(path)
        self.img = undistorter.apply(img) if undistorter else img

    def read(self, wait_s: float = 0.03) -> Optional[np.ndarray]:
        import time
        time.sleep(wait_s)
        return self.img.copy()


class TopicSource(ImageSource):
    def __init__(self, topic: str, undistorter=None, node_name: str = 'vision_tool'):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        from vision_pkg.node_common import image_msg_to_bgr
        self._rclpy = rclpy
        self._convert = image_msg_to_bgr
        self._und = undistorter
        self.frame: Optional[np.ndarray] = None
        rclpy.init(args=sys.argv)
        self.node = rclpy.create_node(node_name)
        self.node.create_subscription(Image, topic, self._cb, qos_profile_sensor_data)
        self.node.get_logger().info(f'訂閱 {topic}，等待影像…')

    def _cb(self, msg) -> None:
        try:
            f = self._convert(msg)
            self.frame = self._und.apply(f) if self._und else f
        except Exception as e:
            self.node.get_logger().warning(f'影像轉換失敗: {e}', throttle_duration_sec=5.0)

    def read(self, wait_s: float = 0.03) -> Optional[np.ndarray]:
        self._rclpy.spin_once(self.node, timeout_sec=wait_s)
        return None if self.frame is None else self.frame.copy()

    def close(self) -> None:
        try:
            self.node.destroy_node()
            if self._rclpy.ok():
                self._rclpy.shutdown()
        except Exception:
            pass


def open_source(args, node_name: str) -> ImageSource:
    calib_dir = calib_io.resolve_calib_dir(args.calib_dir)
    und = None
    if args.undistort:
        path = args.intrinsics or str(calib_io.intrinsics_path(calib_io.CAMERA_NAMES[args.camera], calib_dir))
        und = calib_io.Undistorter(path)
        print(f'已啟用去畸變: {path}')
    if args.image:
        return FileSource(args.image, und)
    topic = args.topic or f'/{calib_io.CAMERA_NAMES[args.camera]}/image_raw'
    return TopicSource(topic, und, node_name)
