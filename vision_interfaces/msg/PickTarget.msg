# /vision/arm1/pick_target：手眼相機 1 目前會選中的瓶蓋（以處理頻率持續發布，選法同 get_pick_target）
std_msgs/Header header   # stamp = 發布時間；frame_id = arm1_base
bool success             # false 時其餘數值無意義，原因見 message
string message
int32 rank               # 不使用（不排序），固定 0
float64 x                # 手臂 1 基座座標 (m)；z = 場地高度 + 物件高度
float64 y
float64 z
int32[] visible_ranks    # 不使用，固定為空
