cmake_minimum_required(VERSION 3.8)
project(vision_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# service：協調器主動查詢；msg：手眼相機以處理頻率持續發布的 topic（內容同對應 service）
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PickTarget.msg"
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
