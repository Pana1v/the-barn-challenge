cmake_minimum_required(VERSION 3.0.2)
project(navigation_pkg)

set(CMAKE_CXX_STANDARD 17)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  visualization_msgs
  robot_localization
)
find_package(Eigen3 REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(nav_astar_cpp src/nav_astar_node.cpp)
target_link_libraries(nav_astar_cpp ${catkin_LIBRARIES} Eigen3::Eigen)

install(TARGETS nav_astar_cpp
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
