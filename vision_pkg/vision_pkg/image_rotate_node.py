"""image_rotate_node：把鏡頭影像旋轉 90/180/270° 後重新發布（鏡頭裝反時用）。

usb_cam 與相機本身都沒有翻轉功能，所以 launch 會把 usb_cam 的輸出改到 <ns>/image_raw_unrotated，
再由本節點旋轉後發布到 <ns>/image_raw。視覺節點、標定工具、rqt_image_view 都看同一份轉正後的影像，
像素座標（homography、HSV、內參）因此一致。

參數：
  rotate_deg  0 / 90 / 180 / 270，順時針。0 = 原樣轉發。
"""
from __future__ import annotations

import numpy as np

# 每種編碼一個像素佔幾個 byte；YUYV 兩個像素共用色度，不能直接按像素旋轉，所以不支援
_BPP = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1, '8uc1': 1, 'mono16': 2, '16uc1': 2}
VALID_ROTATIONS = (0, 90, 180, 270)


def rotate_pixels(data: bytes, height: int, width: int, step: int, encoding: str, rotate_deg: int):
    """回傳 (資料 bytes, 新高度, 新寬度, 新 step)。不支援的編碼拋 ValueError。"""
    if rotate_deg not in VALID_ROTATIONS:
        raise ValueError(f'rotate_deg 只能是 {VALID_ROTATIONS}，收到 {rotate_deg}')
    bpp = _BPP.get((encoding or '').lower())
    if bpp is None:
        raise ValueError(f'旋轉不支援影像編碼 {encoding!r}（請讓 usb_cam 輸出 rgb8，例如 pixel_format: yuyv2rgb）')
    buf = np.frombuffer(data, dtype=np.uint8)
    if buf.size < height * step or step < width * bpp:
        raise ValueError(f'影像資料長度異常（{buf.size} < {height * step}，step={step}）')
    img = buf[:height * step].reshape(height, step)[:, :width * bpp].reshape(height, width, bpp)
    # np.rot90 的 k 是逆時針；順時針 rotate_deg -> k = -rotate_deg/90
    out = np.ascontiguousarray(np.rot90(img, k=-(rotate_deg // 90)))
    h2, w2 = out.shape[:2]
    return out.tobytes(), h2, w2, w2 * bpp


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    class ImageRotateNode(Node):
        def __init__(self):
            super().__init__('image_rotate')
            self.rotate_deg = int(self.declare_parameter('rotate_deg', 180).value)
            if self.rotate_deg not in VALID_ROTATIONS:
                raise SystemExit(f'rotate_deg 只能是 {VALID_ROTATIONS}，收到 {self.rotate_deg}')
            self.pub = self.create_publisher(Image, 'image_raw', qos_profile_sensor_data)
            self.create_subscription(Image, 'image_raw_unrotated', self._cb, qos_profile_sensor_data)
            self.get_logger().info(f'image_raw_unrotated -> 順時針旋轉 {self.rotate_deg}° -> image_raw')

        def _cb(self, msg: Image) -> None:
            if self.rotate_deg == 0:
                self.pub.publish(msg)
                return
            try:
                data, h, w, step = rotate_pixels(bytes(msg.data), msg.height, msg.width, msg.step,
                                                 msg.encoding, self.rotate_deg)
            except ValueError as e:
                self.get_logger().error(str(e), throttle_duration_sec=5.0)
                return
            out = Image()
            out.header = msg.header
            out.height, out.width, out.step = h, w, step
            out.encoding = msg.encoding
            out.is_bigendian = msg.is_bigendian
            out.data = data
            self.pub.publish(out)

    rclpy.init(args=args)
    node = ImageRotateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
