cmake_minimum_required(VERSION 3.8)
project(vision_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# 自訂 service（協調器的請求/回應）與 msg（三個節點持續發布的狀態 topic）
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ZoneState.msg"
  "msg/PickTarget.msg"
  "msg/RankingSnapshot.msg"
  "msg/PickDetections.msg"
  "msg/GridState.msg"
  "srv/RecalibrateZones.srv"
  "srv/GetZoneState.srv"
  "srv/CaptureRanking.srv"
  "srv/GetRank.srv"
  "srv/GetPickTarget.srv"
  "srv/GetGridState.srv"
  "srv/VerifyPlace.srv"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
