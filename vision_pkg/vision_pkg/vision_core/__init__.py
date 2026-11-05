"""vision_core：純 OpenCV / numpy 的影像演算法（不得 import rclpy）。

模組：
  geometry      角點排序、homography、多邊形遮罩、去畸變
  color_zone    HSV 分割、場地（粉紅）定位與三區均分、各區占用判斷
  cap_detect    藍色圓形瓶蓋偵測
  holes         C 區 8 個方孔偵測、固定編號、每孔是否已放置、放置順序檢查
  params_util   扁平參數字典 -> dataclass
所有閾值皆由呼叫端傳入，模組內不寫死數值。
"""
