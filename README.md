# 雙機械手臂協同搬運系統 — 視覺（三顆鏡頭 / OpenCV / ROS 2 Jazzy）

只負責視覺：環境相機、手眼相機 1（Home_A）、手眼相機 2（Home_C）。任務協調器與手臂控制由其他組員負責，
彼此以 **ROS service** 溝通（請求/回應）；三個節點另外把同一份結果以 **topic** 持續發布，
供監看、記錄或想用訂閱方式的組員使用（見第 2.1 節）。不使用深度學習，OpenCV 用 apt 版（4.6 API）。

**場地與物件**

- **場地**：60×30 cm 粉紅色泡棉板（兩塊 30×30 拼成），沒有黑框、也沒有分隔線。由左到右分成 **C | B | A**，各 20×30 cm。
- **C 區**：有 8 個方孔（約 2.5–3 cm，孔底是白紙），排成 2 行 × 4 列，位在 C 區靠 B 區的那一側。
- **物件**：藍色圓形瓶蓋，直徑 20 mm，**8 個大小都一樣，所以不排序**。手臂 1 在 A 區夾**任何一個**都可以，
  放置順序完全由 C 區的格位編號決定。
- **C 區格位編號（固定）**：從使用者的位置看（站在場地長邊前，B 區在右手邊），最下排（靠近使用者）左邊是 1、右邊是 2，
  往上一排是 3、4，最上排是 7、8。依 1 → 8 的順序放。

```
   C 區          B 區 →
  [7] [8]
  [5] [6]
  [3] [4]
  [1] [2]
  ─── 使用者這側 ───
```

```
 usb_cam /env_cam ──► env_vision_node  ──► srv   /vision/env/{recalibrate_zones,get_zone_state,capture_ranking,get_rank,clear_taken}
                                       └─► topic /vision/env/{zone_state,ranking}
 usb_cam /arm1_cam ─► arm1_vision_node ──► srv   /vision/arm1/get_pick_target
                                       └─► topic /vision/arm1/{pick_target,detections}
 usb_cam /arm2_cam ─► arm2_vision_node ──► srv   /vision/arm2/{get_grid_state,verify_place}
                                       └─► topic /vision/arm2/grid_state
```

三個節點是三個獨立行程，彼此不互相呼叫，所以一顆鏡頭出錯不會影響其他節點。影像演算法全部放在 `vision_core/`，
不依賴 ROS，可以直接用靜態圖片測試。

---

## 1. 套件目錄結構

```
三鏡頭Opencv/
├── README.md
├── vision_interfaces/                  # ament_cmake：所有自訂 srv / msg
│   ├── CMakeLists.txt  package.xml
│   ├── srv/ RecalibrateZones  GetZoneState  CaptureRanking  GetRank
│   │        GetPickTarget  GetGridState  VerifyPlace   (.srv)
│   └── msg/ ZoneState  RankingSnapshot  PickDetections  GridState   (.msg，topic 用)
└── vision_pkg/                         # ament_python
    ├── package.xml  setup.py  setup.cfg  pytest.ini  resource/vision_pkg
    ├── launch/vision.launch.py         # 一次啟動 3 個 usb_cam + 3 個視覺節點（show_debug 啟動參數）
    ├── config/                         # 安裝到 share/vision_pkg/config
    │   ├── env_vision.yaml  arm1_vision.yaml  arm2_vision.yaml   # 各節點所有閾值/ROI/門檻/逾時
    │   ├── hsv.yaml                    # HSV 預設值（pink 場地 / blue 瓶蓋 / black 手臂）
    │   ├── usb_cam.yaml  cameras.yaml  # usb_cam 參數、三顆鏡頭的 /dev/v4l/by-id 路徑（待填）
    │   └── zones.yaml  arm1_homography.yaml  a_station_homography.yaml  arm2_homography.yaml   # 空範本
    ├── vision_pkg/
    │   ├── env_vision_node.py  arm1_vision_node.py  arm2_vision_node.py   # 三個節點，各一檔
    │   ├── node_common.py  calib_io.py  debug_draw.py                      # 共用工具（不含任何執行期共享狀態）
    │   ├── vision_core/                # 純函式（不含 ROS）
    │   │   ├── color_zone.py           # HSV 分割、粉紅場地定位（抗邊緣凸出物）、三區均分、各區占用/手臂判斷
    │   │   ├── cap_detect.py           # 藍色圓形瓶蓋偵測（圓度、填滿率、面積、邊界）
    │   │   ├── holes.py                # C 區 8 個方孔偵測、固定編號、每孔是否已放置、放置順序檢查
    │   │   └── geometry.py  params_util.py
    │   └── tools/                      # hsv_tuner  calibrate_intrinsics  calibrate_homography
    └── tests/                          # pytest：靜態圖片 + 假 ROS（見第 8 節）
```

執行期產生的檔案放在 **calib_dir**（參數 `calib_dir` > 環境變數 `VISION_CALIB_DIR` > `~/.ros/vision_calib`）：
`zones.yaml`、`hsv.yaml`（hsv_tuner 逐色覆寫套件預設）、`arm1_homography.yaml`、`a_station_homography.yaml`、
`arm2_homography.yaml`、`<cam>_intrinsics.yaml`。讀取時 calib_dir 優先，找不到才退回套件內建範本
（所以不需要 `--symlink-install` 也能寫入）。

## 2. srv 定義（`vision_interfaces/srv/`）

所有欄位都和原本一樣，協調器不必改介面。差別在於瓶蓋不排序，排序相關的 service 改成「保留但不排序」，詳見各項說明。

```
# RecalibrateZones.srv —— 重新找出粉紅場地並均分成 C|B|A（場地淨空時呼叫）
---
bool success
string message

# GetZoneState.srv
---
bool a_has_object       # A 區有沒有瓶蓋（藍色為主）
bool b_has_object
bool b_arm_clear        # true = 手臂已離開 B 區；無法判斷時一律 false（保守）
int32 c_object_count    # 環境相機看到的 C 區瓶蓋數
bool success
string message

# CaptureRanking.srv —— 不排序：只清點 A 區瓶蓋數。數量不是 8 只在 message 提醒，不會失敗
---
bool success
string message
int32 count
bool low_confidence     # 固定 false

# GetRank.srv —— 不排序：固定回 success=true、rank=0（保留介面）
float64 x_mm
float64 y_mm
bool peek_only
---
bool success
string message
int32 rank
float64 distance_mm

# GetPickTarget.srv —— 挑一個可夾的瓶蓋（可達範圍內、周圍淨空優先、離基座最近優先）
int32 target_rank       # 原樣帶回 rank（協調器可用來記「這次要放第幾格」），不影響選擇結果
bool peek_only          # 保留，沒有作用
---
bool success
string message
int32 rank              # = target_rank
float64 x               # 手臂 1 基座座標 (m)；z = 場地高度 + 物件高度
float64 y
float64 z

# GetGridState.srv
---
bool[8] occupied        # 格位 1..8（固定編號，見最上面的圖）
float64[8] place_x      # 各孔中心的手臂 2 基座座標 (m)
float64[8] place_y
float64[8] place_z
bool all_full
bool success
string message

# VerifyPlace.srv
int32 slot              # 1..8
---
bool placed             # 該格是否已放瓶蓋
bool order_ok           # 已放置的格位是否剛好是 1..n 連續（沒有跳號）
int32[] wrong_slots     # 跳號放置的格位（前面還有空格卻已經放了）
bool success
string message
```

`/vision/env/clear_taken`（`std_srvs/Trigger`）保留，但不再有作用，固定回 success。

**共通行為**：每個 service 都有逾時（`service_timeout_s`，預設 2 秒，待實測）。遇到影像中斷、場地未定位、
homography 未標定、方孔數量不對等情況，一律回 `success=false` 並在 `message` 說明原因，節點不會崩潰。
無法判斷時不會猜測，例如方孔沒找到 8 個、或孔位排列和 `view_rotation_k` 對不上時。

### 2.1 Topic（`vision_interfaces/msg/`）

| Topic | 型別 | 發布時機 | 內容 |
|---|---|---|---|
| `/vision/env/zone_state` | `ZoneState` | 每處理一張新影格（≈ `process_hz`） | 同 `get_zone_state` |
| `/vision/env/ranking` | `RankingSnapshot` | 開機發布一次（`transient_local`） | 固定 `valid=false`、message 說明「不排序」 |
| `/vision/arm1/pick_target` | `PickTarget` | 每個處理週期（獨立計時器） | 目前會選中的瓶蓋與基座 x/y/z（同 `get_pick_target`；`rank` 固定 0） |
| `/vision/arm1/detections` | `PickDetections` | 每處理一張新影格 | 各瓶蓋的像素 / A 區平面 mm / 手臂 1 基座 m 座標；`side_px` = 直徑 |
| `/vision/arm2/grid_state` | `GridState` | 每處理一張新影格 | 同 `get_grid_state`，另加各孔藍色比例與 `verify_place` 的順序檢查（`side_mm` 固定 NaN） |

共通規則：每則訊息都有 `header`、`valid`、`message`（`pick_target` 用 `success`，與 service 一致）。
`frame_id`：含基座座標的 `pick_target` / `grid_state` 為 `arm1_base` / `arm2_base`，其餘為相機名稱。`valid=false` 時只看 `message`
（例如「影像中斷 0.8s」「場地尚未定位」「C 區方孔定位失敗」），其餘欄位無意義；`zone_state` 無效時 `b_arm_clear` 一律 false。
homography 未標定時座標欄位以 `plane_valid` / `base_valid` / `place_valid` 標示，不影響其他欄位。

**安全性**：B 區放行等「要動手臂前」的判斷仍建議用 service 拿當下結果；用 topic 時請檢查 `valid` 與 `header.stamp` 是否夠新。

## 3. 介面速查

```bash
ros2 service call /vision/env/recalibrate_zones vision_interfaces/srv/RecalibrateZones
ros2 service call /vision/env/get_zone_state    vision_interfaces/srv/GetZoneState
ros2 service call /vision/env/capture_ranking   vision_interfaces/srv/CaptureRanking      # 清點 A 區瓶蓋數
ros2 service call /vision/arm1/get_pick_target  vision_interfaces/srv/GetPickTarget "{target_rank: 1}"
ros2 service call /vision/arm2/get_grid_state   vision_interfaces/srv/GetGridState
ros2 service call /vision/arm2/verify_place     vision_interfaces/srv/VerifyPlace "{slot: 3}"

ros2 topic echo /vision/env/zone_state
ros2 topic echo /vision/arm1/pick_target
ros2 topic echo /vision/arm1/detections
ros2 topic echo /vision/arm2/grid_state
ros2 topic hz   /vision/arm2/grid_state
```

典型一輪（k = 1..8）：`get_pick_target(target_rank=k)` → 手臂 1 夾起並搬運（進出 B 區前看 `get_zone_state.b_arm_clear`）
→ 手臂 2 放到 `get_grid_state` 給的 `place_*[k-1]` → `verify_place(slot=k)`（`placed` 且 `order_ok`）。
批次開始時可以先呼叫 `capture_ranking`，確認 A 區有 8 個瓶蓋。

## 4. 安裝、建置與執行（Jetson AGX Orin / Ubuntu 24.04 / ROS 2 Jazzy）

```bash
sudo apt update
sudo apt install ros-jazzy-usb-cam python3-opencv python3-numpy python3-yaml python3-pytest v4l-utils
python3 -c "import cv2; print(cv2.__version__)"      # 應為 apt 版（4.6.x）；若被 pip 的 opencv-python 蓋掉請移除它

mkdir -p ~/vision_ws/src && cp -r vision_interfaces vision_pkg ~/vision_ws/src/
cd ~/vision_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y      # 選用
colcon build --symlink-install                          # 會先建 vision_interfaces
source install/setup.bash

ls /dev/v4l/by-id/                                      # 查三顆鏡頭的穩定路徑（取 ...-video-index0）
v4l2-ctl -d /dev/v4l/by-id/<名稱> --list-formats-ext    # 確認支援 MJPG 640x480 30fps

ros2 launch vision_pkg vision.launch.py \
    env_cam_device:=/dev/v4l/by-id/<env> arm1_cam_device:=/dev/v4l/by-id/<arm1> arm2_cam_device:=/dev/v4l/by-id/<arm2> \
    show_debug:=true
```

- 鏡頭路徑可用 launch 參數或填 `config/cameras.yaml`（launch 參數優先）。**兩處皆空或路徑不存在時**，launch 會印出錯誤與
  `ls /dev/v4l/by-id/` 提示、略過該顆 usb_cam，其他鏡頭與視覺節點照常啟動（對應 service 會回「無影像」）。
- launch 參數：`show_debug`（預設 true；沒有 DISPLAY 時自動停用）、`calib_dir`、`undistort`（空=沿用 YAML）、`respawn`（預設 true）。
- usb_cam 命名空間 `/env_cam`、`/arm1_cam`、`/arm2_cam`；視覺節點只訂閱 `/<ns>/image_raw`，**不會自己開 /dev/video**。
- 單獨啟動一個節點：
  `ros2 run vision_pkg env_vision_node --ros-args --params-file $(ros2 pkg prefix vision_pkg)/share/vision_pkg/config/env_vision.yaml`
- 即時調參：`ros2 param set /env_vision_node match_threshold_mm 25.0`（約 1 秒內生效；設錯會被拒絕並沿用舊值）。
- debug 視窗（視窗按 `q`/Esc 只會關視窗）：
  - env：三區框線、各區狀態、瓶蓋輪廓。
  - arm1：A 區 ROI、每顆瓶蓋的圓圈，下方兩行是座標 x / y，被選中的那顆標 `PICK`。
  - arm2：8 個方孔、格位編號；孔中央兩行是放置座標 x / y，孔下方是藍色比例。
  - 座標單位：homography 已標定時是手臂基座座標（公尺）；未標定時改顯示像素座標，左上角會註明目前是哪一種。
    標定完可以拿手臂 TCP 去碰瓶蓋或孔中心，對照畫面上的數字檢查準不準。

## 5. 首次上線的標定步驟順序

1. **硬體與光線**：三顆鏡頭固定好、光源固定、場地放好；場地淨空。先跑 launch，用 `ros2 topic hz /env_cam/image_raw`
   與 `ros2 run rqt_image_view rqt_image_view` 確認三路影像都有。環境相機要讓場地在影像中是**橫的**，
   預設由左至右是 C|B|A；如果影像裡是 A|B|C，就設 `field.reverse: true`。
2. **鎖定曝光/白平衡**（建議）：在 `config/usb_cam.yaml` 把 `auto_white_balance`/`autoexposure` 設 false 並填手動值（相機需支援）。
   三顆鏡頭共用同一份 `hsv.yaml`，所以三台的色彩表現要盡量一致。
3. **（建議）內參校正**，三顆各一次：
   `ros2 run vision_pkg calibrate_intrinsics --camera env --cols 9 --rows 6 --square-mm 25`（arm1、arm2 同理）。
   存成 camera_info 格式 `<calib_dir>/<cam>_cam_intrinsics.yaml`，launch 會自動帶給 usb_cam。若決定啟用去畸變：
   launch 加 `undistort:=true`，**之後 calibrate_homography 也要加 `--undistort`，兩邊必須一致**（先決定再標 homography）。
4. **HSV 調整**：`ros2 run vision_pkg hsv_tuner --camera env --color pink`（鍵 1–3 切 pink/blue/black，`s` 存）。
   - `pink`：整塊場地要白、桌面和孔底白紙要黑。
   - `blue`：瓶蓋要白，場地不能有雜點。
   - 再到 arm1/arm2 的影像確認 pink、blue 也能用。調完節點會自動重新載入，不必重啟。
5. **把實測值填進 config**（目前都是預設值，YAML 內標「待實測」）：
   - `object_height_mm` / `station_height_mm`、`reach.min_m` / `reach.max_m`、`pick.min_clearance_m`
   - 手眼相機的 `cap.*` 和 `hole.*`：單位是原始影像像素，依相機高度而定。看 debug 視窗量瓶蓋、方孔在影像中的大小再填。
   - `hole.occupied_threshold`、逾時。
6. **環境相機場地定位**：場地淨空 →
   `ros2 service call /vision/env/recalibrate_zones vision_interfaces/srv/RecalibrateZones`。看 env 的 debug 視窗三個框是否貼齊場地，
   並確認 C 區在有方孔的那一側。成功會寫入 `<calib_dir>/zones.yaml`，之後重啟沿用；偵測失敗時讀上次存檔並警告。
7. **手眼相機 1 標定**（手臂在 Home_A，**標記放在瓶蓋頂面高度**）：
   - `ros2 run vision_pkg calibrate_homography --target arm1`：空白鍵凍結 → 點 ≥ 6–8 個已知點（用手臂 TCP 碰標記讀基座 x y），
     每點在終端機輸入 `x y`（公尺）。看每點誤差（目標 < 3 mm），`s` 存檔。
   - （選用）`calibrate_homography --target a_station --corners`：依序點 A 區左上、右上、右下、左下。標了之後只偵測 A 區範圍內的瓶蓋；
     不標就偵測整張影像，所以 A 區以外不能有藍色東西。
   - 驗證：`get_pick_target`，核對 x/y/z；debug 視窗標 `PICK` 的就是被選中的瓶蓋。
8. **手眼相機 2 標定**（手臂在 Home_C）：`calibrate_homography --target arm2`（同上，點在放置高度）。
   接著看 arm2 的 debug 視窗：8 個孔都要框到，**1 號要在使用者視角的左下**。不對就把 `view_rotation_k` 改成 1/2/3，
   意思是手眼相機影像要順時針轉幾個 90° 才會變成使用者視角。設錯時節點會回報失敗，不會給出錯的編號。
9. **整合驗證**：
   - `get_zone_state`：包含手臂進出 B 區。
   - `get_pick_target` → 搬運 → `get_grid_state` → `verify_place`：故意跳號放一格，應回 `order_ok=false` 和 `wrong_slots`。

## 6. 已知限制與調參建議

**尚未在真機驗證。** 本專案是在沒有 ROS / Jetson / 相機的 Windows 電腦上寫的。演算法與三個節點都用「合成靜態圖片 + 假 rclpy」
驗證過，合成場景依照實際場地照片做（粉紅場地、C 區 8 孔、下緣凸出的手臂底座、藍色瓶蓋），也測了 service 逾時與型別檢查。
但 `colcon build`、usb_cam 實際輸出、真實光源下的 HSV 與各閾值都還沒實測。
launch 內的 usb_cam 設定（執行檔 `usb_cam_node_exe`、`pixel_format: mjpeg2rgb`、topic `image_raw`）是依 ros-jazzy-usb-cam 的慣例寫的，
第一次上機請用 `ros2 pkg executables usb_cam`、`ros2 topic list` 確認。

**場地（環境相機）**
- env 取「粉紅色最大色塊」當場地，再從輪廓凸包挑出面積最大的四邊形當四個角，所以邊上的小凸出物不會把角點拉歪，
  例如手臂底座那塊泡棉板。但如果有**大塊**粉紅色東西連在場地邊上（手臂底座的泡棉墊太大、或貼著場地擺），角點仍可能被拉走。
  請看 debug 視窗確認；必要時把那些泡棉換色或挪開。
- 三區是在場地平面上均分（各 200 mm），不依賴分隔線。兩塊板子的接縫由 `field_mask.close_ksize` 補起來。
- C 區有方孔，所以淨空基準（場地色可見比例）會比 A/B 低約 1 成，這是正常的。

**瓶蓋 / 占用判斷**
- 一顆 20 mm 瓶蓋只佔一區約 0.5% 面積，所以「有沒有瓶蓋」主要看區內藍色比例（`occupancy.object_ratio_thresh`，約半顆瓶蓋），
  而不是場地被遮擋的比例。藍色的雜物（筆、膠帶、藍色線材）會被當成瓶蓋，場地上請不要放。
- `c_object_count` 與 `capture_ranking.count` 是藍色塊數：兩顆瓶蓋相黏會少算。
- `b_arm_clear`：粉紅色可見面積驟降（≤ 基準 60%）且黑色 ≥ 15% 才算手臂。手臂進入立刻回報；離開需連續 5 次（0.5 s）確認才回 true
  （安全優先，無法判斷一律 false）。手臂有非黑色部位、強陰影、背光時請調 `occupancy.*`。這只是視覺輔助，
  B 區互斥的最終保證不應只靠視覺。

**取件（手眼相機 1）**
- 不排序，所以每次挑「可達範圍內 → 周圍淨空（最近的其他瓶蓋 ≥ `pick.min_clearance_m`）→ 離基座最近」的那一個。
  都不夠淨空時仍選最近的，message 會註明。
- 被排除的藍色區塊（例如兩顆相黏）不會被選中，但會算進「周圍淨空」的距離計算。
- 手眼相機假設手臂**停在 Home 姿態**。節點不知道手臂在哪裡，手臂移動中的畫面會產生錯誤結果，
  所以協調器應在手臂到位、穩定（建議 ≥ 0.3 s）後再呼叫。
- homography 只對應「一個平面」，標定點請放在瓶蓋頂面（夾取）高度。

**放置（手眼相機 2）**
- 方孔 = 粉紅遮罩裡被包圍的洞。手眼相機畫面必須把 8 個孔完整拍進來，而且孔的四周要是粉紅色。
  孔被夾爪遮住時，會沿用最近一次的孔位 `hole_hold_s` 秒，超過就回報失敗。
- `field_mask.close_ksize` 必須小於方孔在影像中的邊長，否則孔會被補掉。
- 占用 = 孔內（內縮 `margin_ratio`）藍色比例 ≥ `hole.occupied_threshold`。20 mm 瓶蓋在約 28 mm 方孔中約 0.4–0.6。
  瓶蓋沒放正、斜靠在孔壁時比例會下降。
- 「順序正確」的定義：已放置的格位剛好是 1..n。只要前面還有空格，後面已放的格位就算 `wrong_slots`。

**顏色**
- 粉紅泡棉的飽和度不高（S 約 60–90），白紙、灰桌面的 S 更低，`pink` 的 S 下限是主要分界。皮膚、紅色物品也可能落在 pink 範圍內，
  請讓它們遠離場地邊緣，避免連成一塊。
- 光線變化是最大風險來源：固定光源、鎖曝光與白平衡、避免日光直射與反光。

**參數對照（常用）**：`field.*` / `field_mask.*`（場地定位）、`occupancy.*`（有無瓶蓋/手臂）、`object_count.*`、
`cap.*`（arm1 瓶蓋判定）、`pick.min_clearance_m`、`hole.*` / `view_rotation_k` / `hole_hold_s`（arm2 方孔）、`debounce.*`。

## 7. 設計決定（規格內未明確、我自行採用的）

- **場地定位**：env 開機取 5 張影格逐點中位數；粉紅最大色塊 → 凸包上面積最大的內接四邊形 → 均分三區。
  失敗的區沿用現有 ROI 或上次存檔並記錄警告。
- **不排序**：瓶蓋大小相同。`capture_ranking` / `get_rank` / `clear_taken` / `ranking` topic 保留介面但不排序；
  `get_pick_target` 原樣帶回 `target_rank`，讓協調器原本「第 k 輪」的流程不用改。
- **方孔編號**：只用 8 個孔中心的相對位置，用主軸找出 4 排的方向、再依「由下往上、由左至右」編號，所以不需要看到 C 區外框。
  影像方向由 `view_rotation_k` 指定；孔位不像 4×2 格子、或長軸方向不符時一律回報失敗。
- **執行緒**：三個節點都用 `MultiThreadedExecutor`；影像、處理計時器、service 伺服端各一個 MutuallyExclusiveCallbackGroup
  （arm1 另有 topic 發布計時器一組）。三個節點之間不互相呼叫 service。
- **影像轉換**不依賴 cv_bridge（避免 Jetson 上 numpy ABI 問題）。內參校正預設固定 k3=0（`--with-k3` 可放開）。

## 8. 測試

```bash
cd vision_pkg && python3 -m pytest tests -q
```

不需要 ROS，只要有 numpy / OpenCV / PyYAML / pytest 就能跑。測試內容：
- 用合成靜態圖片（`tests/data/*.png`，缺少時自動生成；`python tests/make_test_images.py` 可以重新輸出）測：
  場地定位（含凸出的手臂底座）、三區占用/手臂判斷、瓶蓋偵測、方孔偵測與編號（含相機旋轉 0–3）、放置順序檢查。
- `tests/fake_ros.py` 用假 rclpy 對三個節點做端到端測試。
- 另外也測 launch 邏輯與標定工具（離線、腳本化 GUI）。
