# /vision/arm2/grid_state（手眼相機 2，每處理一張新影格發布一次）
# 格位 1..8（固定）：使用者視角（B 區在右）最下排左 1、右 2，往上 3、4 … 最上排 7、8；陣列第 i 項 = 格位 i+1。
std_msgs/Header header  # frame_id = arm2_base（place_* 的座標系）
bool valid
string message
bool[8] occupied
float64[8] fill_ratio   # 孔內藍色（瓶蓋）像素比例
float64[8] side_mm      # 不使用（瓶蓋大小相同），固定 NaN
bool all_full
bool place_valid        # arm2_homography 已標定；false 時 place_* 全為 0
float64[8] place_x      # 各格中心的手臂 2 基座座標 (m)
float64[8] place_y
float64[8] place_z
# 放置順序檢查（同 verify_place）：已放置的格位須為 1..n 連續
bool order_checked
bool order_ok
int32[] wrong_slots
int32[] unmeasurable_slots   # 不使用，固定為空
int32[] approx_slots         # 不使用，固定為空
