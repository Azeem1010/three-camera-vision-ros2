cmake_minimum_required(VERSION 3.8)
project(vision_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 所有自訂 service（資料交換一律走 service，不使用自訂 topic）
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/RecalibrateZones.srv"
  "srv/GetZoneState.srv"
  "srv/CaptureRanking.srv"
  "srv/GetRank.srv"
  "srv/GetPickTarget.srv"
  "srv/GetGridState.srv"
  "srv/VerifyPlace.srv"
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
