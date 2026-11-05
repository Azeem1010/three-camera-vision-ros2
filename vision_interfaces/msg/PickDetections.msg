# /vision/arm1/detections（手眼相機 1，每處理一張新影格發布一次）
# 各藍色瓶蓋的偵測結果與座標（瓶蓋大小相同、不排序）。
std_msgs/Header header  # frame_id = arm1_cam
bool valid
string message
int32 count
# 以下陣列長度 = count
float64[] pixel_x       # 瓶蓋中心（影像像素）
float64[] pixel_y
float64[] side_px       # 瓶蓋直徑（影像像素，等面積圓）
bool plane_valid        # a_station_homography 已標定；false 時 plane_* 為空陣列
float64[] plane_x_mm    # A 區平面座標
float64[] plane_y_mm
bool base_valid         # arm1_homography 已標定；false 時 base_x / base_y 為空陣列
float64[] base_x        # 手臂 1 基座座標 (m)
float64[] base_y
float64 base_z          # 場地高度 + 物件高度 (m)
