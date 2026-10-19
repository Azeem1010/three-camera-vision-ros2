# 雙機械手臂協同搬運系統 — 視覺（三顆鏡頭 / OpenCV / ROS 2 Jazzy）

只負責視覺：環境相機、手眼相機 1（Home_A）、手眼相機 2（Home_C）。任務協調器與手臂控制由其他組員負責，
彼此以 **ROS service** 溝通（請求/回應）；三個節點另外把同一份結果以 **topic** 持續發布，
供監看、記錄或想用訂閱方式的組員使用（見第 2.1 節）。不使用深度學習，OpenCV 用 apt 版（4.6 API）。

```
 usb_cam /env_cam ──► env_vision_node  ──► srv   /vision/env/{recalibrate_zones,get_zone_state,capture_ranking,get_rank,clear_taken}
                                       └─► topic /vision/env/{zone_state,ranking}
 usb_cam /arm1_cam ─► arm1_vision_node ──► srv   /vision/arm1/get_pick_target ──(client)──► /vision/env/get_rank
                                       └─► topic /vision/arm1/{pick_target,detections}
 usb_cam /arm2_cam ─► arm2_vision_node ──► srv   /vision/arm2/{get_grid_state,verify_place}
                                       └─► topic /vision/arm2/grid_state
```

三個節點是三個獨立行程（一顆鏡頭出錯不影響其他節點）；影像演算法全部在 `vision_core/`，不含 ROS，可用靜態圖片單獨測試。

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
    │   ├── hsv.yaml                    # HSV 預設值（blue/green/c_zone/orange/black）
    │   ├── usb_cam.yaml  cameras.yaml  # usb_cam 參數、三顆鏡頭的 /dev/v4l/by-id 路徑（待填）
    │   └── zones.yaml  arm1_homography.yaml  a_station_homography.yaml  arm2_homography.yaml   # 空範本
    ├── vision_pkg/
    │   ├── env_vision_node.py  arm1_vision_node.py  arm2_vision_node.py   # 三個節點，各一檔
    │   ├── node_common.py  calib_io.py  debug_draw.py                      # 共用工具（不含任何執行期共享狀態）
    │   ├── vision_core/                # 純函式（不含 ROS）
    │   │   ├── color_zone.py           # HSV 分割、最大輪廓、approxPolyDP 四角點、透視校正、占用判斷、站台定位
    │   │   ├── square_detect.py        # 橘色正方形偵測（4 頂點、長寬比≤1.15、填滿率≥0.85）與邊長量測
    │   │   ├── size_ranker.py          # 相對排序；介面可替換成「絕對尺寸對照表」
    │   │   ├── grid.py                 # C 區 2x4 切格、每格橘色比例、各格邊長
    │   │   ├── snapshot.py             # A 區快照（偵測+排序）與座標配對
    │   │   └── geometry.py  params_util.py
    │   └── tools/                      # hsv_tuner  calibrate_intrinsics  calibrate_homography
    └── tests/                          # pytest：靜態圖片 + 假 ROS（見第 8 節）
```

執行期產生的檔案放在 **calib_dir**（參數 `calib_dir` > 環境變數 `VISION_CALIB_DIR` > `~/.ros/vision_calib`）：
`zones.yaml`、`hsv.yaml`（hsv_tuner 逐色覆寫套件預設）、`arm1_homography.yaml`、`a_station_homography.yaml`、
`arm2_homography.yaml`、`<cam>_intrinsics.yaml`。讀取時 calib_dir 優先，找不到才退回套件內建範本
（所以不需要 `--symlink-install` 也能寫入）。

## 2. srv 定義（`vision_interfaces/srv/`）

```
# RecalibrateZones.srv
---
bool success
string message

# GetZoneState.srv
---
bool a_has_object
bool b_has_object
bool b_arm_clear        # true = 手臂已離開 B 區；無法判斷時一律 false（保守）
int32 c_object_count
bool success
string message

# CaptureRanking.srv
---
bool success
string message
int32 count
bool low_confidence

# GetRank.srv
float64 x_mm            # A 站平面座標（原點=站台左上角，x 向右、y 向下）
float64 y_mm
bool peek_only          # false（預設）= 配對成功即標記「已取走」；true = 只查詢
---
bool success
string message
int32 rank
float64 distance_mm

# GetPickTarget.srv
int32 target_rank       # 0 = 預設取目前看得到的 rank 最小者
bool peek_only          # false（預設）= 回傳目標後通知環境相機標記「已取走」
---
bool success
string message
int32 rank
float64 x               # 手臂 1 基座座標 (m)；z = 站台高度 + 物件高度
float64 y
float64 z

# GetGridState.srv
---
bool[8] occupied        # 格位 1..8（上排=靠近 B 區，1–4；下排 5–8；各排由左至右）
float64[8] place_x      # 各格中心的手臂 2 基座座標 (m)
float64[8] place_y
float64[8] place_z
bool all_full
bool success
string message

# VerifyPlace.srv
int32 slot              # 1..8
---
bool placed             # 該格是否已放置
bool order_ok           # 已占用格位的邊長是否隨編號嚴格遞增
int32[] wrong_slots     # 排錯的格位編號（最長遞增子序列以外者）
bool success
string message
```

規格外多加的兩項（都不影響規格內欄位）：`peek_only`（讓重試/預覽不會把物件標成已取走）與
`/vision/env/clear_taken`（`std_srvs/Trigger`，清除所有「已取走」標記，供重試/復原用）。

**共通行為**：每個 service 都有逾時（`service_timeout_s`，預設 2 秒，待實測）；影像中斷、站台未定位、
homography 未標定、偵測數量不符等情況一律回 `success=false` 與明確的 `message`，節點不會崩潰；
無法判斷時不猜測（例如快照不是 8 個物件、格內物件黏在一起量不到邊長）。

### 2.1 Topic（`vision_interfaces/msg/`）

service 全部保留、行為不變；topic 是同一份處理結果的廣播，不會觸發任何動作（拍快照、標記已取走仍只能用 service）。

| Topic | 型別 | 發布時機 | 內容 |
|---|---|---|---|
| `/vision/env/zone_state` | `ZoneState` | 每處理一張新影格（≈ `process_hz`） | 同 `get_zone_state` |
| `/vision/env/ranking` | `RankingSnapshot` | 快照成功/失敗/作廢、已取走標記改變時（`transient_local`，晚訂閱也收得到最後一筆） | 各物件 rank、A 站平面座標、邊長、是否已取走 |
| `/vision/arm1/pick_target` | `PickTarget` | 每個處理週期（獨立計時器） | rank 最小的目標 + 基座 x/y/z、可見 rank（同 `get_pick_target(target_rank=0, peek_only=true)`，**不標記已取走**） |
| `/vision/arm1/detections` | `PickDetections` | 每處理一張新影格 | 各正方形的像素 / A 站平面 mm / 手臂 1 基座 m 座標（**不含 rank**） |
| `/vision/arm2/grid_state` | `GridState` | 每處理一張新影格 | 同 `get_grid_state`，再加各格比例、邊長（NaN = 無）與 `verify_place` 的排序檢查 |

共通規則：每則訊息都有 `header`、`valid`、`message`（`pick_target` 用 `success`，與 service 一致）。
`frame_id`：含基座座標的 `pick_target` / `grid_state` 為 `arm1_base` / `arm2_base`，其餘為相機名稱。`valid=false` 時只看 `message`
（例如「影像中斷 0.8s」「站台尚未定位」「C 區定位失敗」），其餘欄位無意義；`zone_state` 無效時 `b_arm_clear` 一律 false。
homography 未標定時座標欄位以 `plane_valid` / `base_valid` / `place_valid` 標示，不影響其他欄位。

手眼相機 1 有兩個 topic：`pick_target` 直接給「下一個要夾的目標」，但每個週期都要向環境相機查每個物件的 rank
（約 10 Hz × 物件數次 `get_rank` 呼叫）；`detections` 只給座標、不呼叫任何 service，需要 rank 時用
`plane_x_mm / plane_y_mm` 對照 `/vision/env/ranking`。`pick_target` **不會標記「已取走」**：真正要夾取時，
仍要呼叫一次 `get_pick_target`（peek_only=false），否則物件還在畫面上時無法區分「已取走」與「還沒取」。

**安全性**：B 區放行等「要動手臂前」的判斷仍建議用 service 拿當下結果；用 topic 時請檢查 `valid` 與 `header.stamp` 是否夠新。

## 3. 介面速查

```bash
ros2 service call /vision/env/recalibrate_zones vision_interfaces/srv/RecalibrateZones
ros2 service call /vision/env/get_zone_state    vision_interfaces/srv/GetZoneState
ros2 service call /vision/env/capture_ranking   vision_interfaces/srv/CaptureRanking
ros2 service call /vision/env/get_rank          vision_interfaces/srv/GetRank "{x_mm: 100.0, y_mm: 100.0, peek_only: true}"
ros2 service call /vision/env/clear_taken       std_srvs/srv/Trigger
ros2 service call /vision/arm1/get_pick_target  vision_interfaces/srv/GetPickTarget "{target_rank: 0, peek_only: true}"
ros2 service call /vision/arm2/get_grid_state   vision_interfaces/srv/GetGridState
ros2 service call /vision/arm2/verify_place     vision_interfaces/srv/VerifyPlace "{slot: 3}"

ros2 topic echo /vision/env/zone_state
ros2 topic echo /vision/env/ranking --qos-durability transient_local
ros2 topic echo /vision/arm1/pick_target
ros2 topic echo /vision/arm1/detections
ros2 topic echo /vision/arm2/grid_state
ros2 topic hz   /vision/arm2/grid_state
```

典型一輪：`capture_ranking`（批次開始，8 個物件都在 A 區）→ 每次取件前 `get_pick_target(target_rank=k)`
→ 手臂 1 搬運（B 區進出前看 `get_zone_state.b_arm_clear`）→ 手臂 2 放入 C 區後 `verify_place(slot=k)`。

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
- debug 視窗：env 畫三區框線/狀態/物件輪廓/rank（`r3x` = 已取走）；arm1 畫 A 區 ROI、正方形框與邊長/rank；
  arm2 上半是原始影像（角點 0..3），下半是校正後 2x4 格（編號、占用比例、邊長）。視窗按 `q`/Esc 只會關視窗。

## 5. 首次上線的標定步驟順序

1. **硬體與光線**：三顆鏡頭固定好、光源固定、站台貼好；桌面淨空。先跑 launch，用 `ros2 topic hz /env_cam/image_raw`
   與 `ros2 run rqt_image_view rqt_image_view` 確認三路影像都有。
2. **鎖定曝光/白平衡**（建議）：在 `config/usb_cam.yaml` 把 `auto_white_balance`/`autoexposure` 設 false 並填手動值（相機需支援）。
   三顆鏡頭共用同一份 `hsv.yaml`，所以三台的色彩表現要盡量一致。
3. **（建議）內參校正**，三顆各一次：
   `ros2 run vision_pkg calibrate_intrinsics --camera env --cols 9 --rows 6 --square-mm 25`（arm1、arm2 同理）。
   存成 camera_info 格式 `<calib_dir>/<cam>_cam_intrinsics.yaml`，launch 會自動帶給 usb_cam。若決定啟用去畸變：
   launch 加 `undistort:=true`，**之後 calibrate_homography 也要加 `--undistort`，兩邊必須一致**（先決定再標 homography）。
4. **HSV 調整**：`ros2 run vision_pkg hsv_tuner --camera env --color black`（鍵 1–5 切 blue/green/c_zone/orange/black，`s` 存）。
   env 相機調 black（外框）、c_zone（白色內部）、orange；再到 arm1/arm2 影像確認 orange、c_zone 也可用。節點自動重新載入，不必重啟。
5. **把實測值填進 config**（目前都是預設值，YAML 內已標「待實測」）：站台尺寸 `zone_size_mm`/`a_station_size_mm`/`c_station_size_mm`、
   `object_height_mm`/`station_height_mm`、`reach.min_m/max_m`、`match_threshold_mm`、`ranker.confidence_margin_px`、
   `grid.occupied_threshold`、逾時。
6. **環境相機站台定位**：桌面淨空 →
   `ros2 service call /vision/env/recalibrate_zones vision_interfaces/srv/RecalibrateZones`。看 env 的 debug 視窗三個框是否貼齊站台；
   成功會寫入 `<calib_dir>/zones.yaml`（之後重啟沿用，偵測失敗時讀上次存檔並警告）。
7. **環境相機快照驗證**：8 個物件擺進 A 區（互不相連、離站台邊緣至少數 mm）→ `capture_ranking`。
   失敗時 `message` 會列出被排除的區塊與原因；`low_confidence=true` 表示相鄰尺寸差小於信心門檻。
8. **手眼相機 1 標定**（手臂在 Home_A，**標記放在物件頂面高度**）：
   - `ros2 run vision_pkg calibrate_homography --target a_station --corners`：依序點 A 站左上、右上、右下、左下（座標自動帶入 mm）。
   - `ros2 run vision_pkg calibrate_homography --target arm1`：空白鍵凍結 → 點 ≥ 6–8 個已知點（用手臂 TCP 碰標記讀基座 x y），
     每點在終端機輸入 `x y`（公尺）。看每點誤差（目標 < 3 mm），`s` 存檔。
   - 驗證：`get_pick_target` 帶 `peek_only: true`，核對 x/y/z 與 `rank`；`get_rank` 的 `distance_mm` 反映兩台相機座標的吻合程度。
9. **手眼相機 2 標定**（手臂在 Home_C）：`calibrate_homography --target arm2`（同上，點在放置高度）。
   接著看 arm2 的 debug 視窗：下半圖的「TOP ROW」必須是**靠近 B 區的那一排**，不是就把 `warp_rotation_k` 改成 1/2/3；
   確認 `c_station_size_mm` 與格位、`grid.occupied_threshold` 合理（見第 6 節）。
10. **整合驗證**：`get_zone_state`（含手臂進出 B 區）、`get_pick_target` → `get_rank` 串接、`get_grid_state`/`verify_place`
    （故意放錯兩格，應回 `wrong_slots`）。

## 6. 已知限制與調參建議

**尚未在真機驗證。** 本專案是在沒有 ROS / Jetson / 相機的 Windows 電腦上寫的：演算法與三個節點用「合成靜態圖片 + 假 rclpy」
驗證過（含 service 逾時、跨節點呼叫、型別檢查）；但 `colcon build`、usb_cam 實際輸出、真實光源下的 HSV 與各閾值都還沒實測。
launch 內的 usb_cam 設定（執行檔 `usb_cam_node_exe`、`pixel_format: mjpeg2rgb`、topic `image_raw`）是依 ros-jazzy-usb-cam 的慣例寫的，
第一次上機請用 `ros2 pkg executables usb_cam`、`ros2 topic list` 確認。

**幾何 / 座標**
- homography 只對應「一個平面」。立方體頂面比站台高，視差會讓座標偏移（相機離中心越遠、物件越高越明顯）。
  標定點請放在物件頂面高度；環境相機與手眼相機 1 的視差不同，若 `get_rank` 回報的 `distance_mm` 常接近門檻，
  請放寬 `match_threshold_mm`（但要小於物件最小間距的一半）或把環境相機裝高、接近正俯視。
- **場地（env，`zone_locate_mode: field`）**：60×30 cm 黑色外框、內部白色，由左至右 A|B|C 各 20×30 cm。env 找黑框
  （`field.edge` = outer 量外緣 / inner 量內緣），在場地平面上均分三區；`roi_erode_px` 要大於框線/分隔線在影像中的寬度。
  相機須讓場地在影像中是橫向；若影像中是 C|B|A 設 `field.reverse: true`。舊的「三塊彩色站台」做法改 `zone_locate_mode: color`。
- C 區 20×30 cm 切 2×4 格，`c_station_size_mm: [300, 200]`，每格 **75×100 mm**。占用判斷是「橘色像素 / 格內面積 ≥ 0.3」，
  這代表小立方體（邊長 < ≈48 mm）在這種格子裡會被判成未占用。請依實際物件尺寸調 `grid.occupied_threshold`。
- arm2 以白色找 C 區外框：若 C 區和 B 區之間**沒有黑色分隔線**，白色會連成一片，C 區會定位錯誤（請確認場地有分隔線）。
- 手眼相機假設手臂**停在 Home 姿態**。節點不知道手臂在哪裡，手臂移動中的畫面會產生錯誤結果；協調器應在手臂到位、穩定
  （建議 ≥ 0.3 s）後再呼叫。

**排序精度**
- 640×480 下 200 mm 站台約佔 200–300 px，約 1 mm/px。預設門檻 3 px（校正後影像，2 px/mm → 1.5 mm）；
  相鄰尺寸建議至少差 3–4 mm。快照會平均 3 張影格降雜訊。需要更高精度就把相機靠近、或改用 `ranker.type: absolute_table`。
- arm2 視角下離相機近的格子物件看起來較大；若尺寸差距小於這個位置偏差，`verify_place` 的順序檢查可能誤判。
- 物件互相接觸（黏成一塊）或貼到站台邊緣時，該區塊會被排除或標為「無法量測」。`capture_ranking` 會失敗並說明、`verify_place`
  回 `success=false`——這是刻意不猜測。請在擺放時留邊距。

**手臂 / 占用判斷**
- `b_arm_clear`：綠色可見面積驟降（≤ 基準 60%）且黑色 ≥ 15% 才算手臂。手臂進入立刻回報；離開需連續 5 次（0.5 s）確認才回 true
  （安全優先，無法判斷一律 false）。手臂有非黑色部位、強陰影、背光時請調 `occupancy.*`。這只是視覺輔助，
  B 區互斥的最終保證不應只靠視覺。
- `c_object_count` 是橘色塊數，物件相連會少算。
- 站台上的橘色物件會被併入遮罩來求 C 區（arm2）外框，所以放滿物件也能定位；但 env 的站台定位請在桌面淨空時做。

**取件**
- `get_pick_target` 預設取「目前看得到」的最小 rank；若真正最小的物件被遮住/沒偵測到，會回傳下一小的，請檢查回傳的 `rank`。
  只要有看得到卻配對不到的物件，預設模式會失敗而不是猜。
- 已取走標記只在 `capture_ranking` 成功後存在記憶體中；env 節點重啟會遺失快照。

**顏色**
- 木頭桌面（H≈10–20）與橘色接近：`orange` 的 S/V 下限要比木頭高，並用 hsv_tuner 在實際光源下確認。
  C 區預設白色（低飽和度、高亮度）容易被反光/白紙干擾，顏色未定案前建議先貼一個專用色。
- 光線變化是最大風險來源：固定光源、鎖曝光與白平衡、避免日光直射與反光。

**參數對照（常用）**：`occupancy.*`（有無物件/手臂）、`square.*`（正方形判定；env/arm2 單位為校正後像素，arm1 為原始像素）、
`ranker.confidence_margin_px`、`match_threshold_mm`、`grid.*`、`warp_rotation_k`、`corner_hold_s`、`debounce.*`。

## 7. 設計決定（規格內未明確、我自行採用的）

- **站台定位**：env 開機取 5 張影格逐點中位數；各站台獨立成功/失敗，失敗者沿用現有 ROI 或上次存檔並記錄警告。
- **站台占用**：被遮擋比例（相對淨空基準）為主、橘色比例為輔、大面積黑色且無橘色視為手臂而非物件。
- **排序器介面**：`SizeRanker.rank()/check_order()`；`ranker.type` 設 `relative` 或 `absolute_table`（已實作，待實測對照表），節點不用改。
  `verify_place` 以最長遞增子序列找出排錯的格位。
- **執行緒**：三個節點都用 `MultiThreadedExecutor`；影像、處理計時器、service 伺服端各一個 MutuallyExclusiveCallbackGroup，
  arm1 另有 client 專用 group，呼叫 env 服務用 `call_async` + `threading.Event`（含逾時），回呼中不 spin。
- **影像轉換**不依賴 cv_bridge（避免 Jetson 上 numpy ABI 問題）。內參校正預設固定 k3=0（`--with-k3` 可放開）。

## 8. 測試

```bash
cd vision_pkg && python3 -m pytest tests -q
```

不需要 ROS（只要 numpy / OpenCV / PyYAML / pytest）。內容：以合成靜態圖片（`tests/data/*.png`，缺少時自動生成；
`python tests/make_test_images.py` 可重新輸出）測三區定位、正方形偵測、相對排序、格位判斷、占用/手臂判斷；
`tests/fake_ros.py` 以假 rclpy 對三個節點做端到端測試；另含 launch 邏輯與標定工具（離線/腳本化 GUI）測試。
