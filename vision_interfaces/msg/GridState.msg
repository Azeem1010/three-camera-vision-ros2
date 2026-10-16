# /vision/arm2/grid_state：C 區 2x4 格位狀態（以處理頻率持續發布，內容同 get_grid_state 的回應）
std_msgs/Header header   # stamp = 發布時間；frame_id = arm2_base
bool[8] occupied         # 格位 1..8（上排=靠近 B 區，1–4；下排 5–8；各排由左至右）
float64[8] place_x       # 各格中心的手臂 2 基座座標 (m)
float64[8] place_y
float64[8] place_z
bool all_full
bool success
string message
