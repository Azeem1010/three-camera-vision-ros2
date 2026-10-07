"""vision_core：純 OpenCV / numpy 的影像演算法（不得 import rclpy）。

模組：
  geometry      角點排序、homography、多邊形遮罩、去畸變
  color_zone    HSV 分割、最大輪廓、四邊形角點、透視校正、站台占用判斷
  square_detect 橘色正方形偵測與邊長量測
  size_ranker   尺寸排序（可替換為絕對尺寸對照表）
  grid          C 區 2x4 格位切割與占用判斷
  snapshot      A 區快照（偵測 + 排序）與座標配對
  params_util   扁平參數字典 -> dataclass
所有閾值皆由呼叫端傳入，模組內不寫死數值。
"""
