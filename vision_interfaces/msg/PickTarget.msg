# /vision/arm1/pick_target：手眼相機 1 看得到的物件中 rank 最小者（以處理頻率持續發布）
# 只是「查詢」結果（同 get_pick_target peek_only=true），不會通知環境相機標記「已取走」
std_msgs/Header header   # stamp = 發布時間；frame_id = arm1_base
bool success             # false 時其餘數值無意義，原因見 message
string message
int32 rank
float64 x                # 手臂 1 基座座標 (m)；z = 站台高度 + 物件高度
float64 y
float64 z
int32[] visible_ranks    # 目前畫面中配對得到的所有 rank
