# /vision/env/ranking（環境相機，transient_local：快照或「已取走」標記改變時發布，晚訂閱也收得到最後一筆）
# valid=false = 目前沒有可用快照（尚未 capture_ranking、上次拍攝失敗或站台重新定位後作廢）。
std_msgs/Header header  # frame_id = env_cam
bool valid
string message
bool low_confidence
# 以下陣列等長，第 i 項 = 第 i 個物件（依 rank 由小到大）
int32[] rank
float64[] x_mm          # A 站平面座標（原點=站台左上角，x 向右、y 向下）
float64[] y_mm
float64[] side_mm
bool[] taken            # 已被 get_rank / get_pick_target 標記為「已取走」
