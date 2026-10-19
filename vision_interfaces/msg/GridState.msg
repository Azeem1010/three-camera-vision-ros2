# /vision/arm2/grid_state（手眼相機 2，每處理一張新影格發布一次）
# 格位 1..8：上排=靠近 B 區（1–4），下排 5–8，各排由左至右；陣列第 i 項 = 格位 i+1。
std_msgs/Header header  # frame_id = arm2_base（place_* 的座標系）
bool valid
string message
bool[8] occupied
float64[8] fill_ratio   # 格內橘色像素比例
float64[8] side_mm      # 物件邊長；未占用或無法量測時為 NaN
bool all_full
bool place_valid        # arm2_homography 已標定；false 時 place_* 全為 0
float64[8] place_x      # 各格中心的手臂 2 基座座標 (m)
float64[8] place_y
float64[8] place_z
# 排序檢查（同 verify_place）：有已占用但量不到邊長的格位時 order_checked=false，不猜測
bool order_checked
bool order_ok
int32[] wrong_slots
int32[] unmeasurable_slots
int32[] approx_slots    # 形狀不是標準正方形，邊長為近似值
