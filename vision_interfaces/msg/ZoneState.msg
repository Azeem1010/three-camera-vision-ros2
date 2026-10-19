# /vision/env/zone_state（環境相機，每處理一張新影格發布一次，約 process_hz）
# 欄位與 GetZoneState.srv 相同；valid=false 時其餘欄位無意義，只看 message。
std_msgs/Header header  # stamp = 產生結果的時間；frame_id = env_cam
bool valid
string message
bool a_has_object
bool b_has_object
bool b_arm_clear        # true = 手臂已離開 B 區；無法判斷時一律 false（保守）
int32 c_object_count
