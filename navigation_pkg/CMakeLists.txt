cmake_minimum_required(VERSION 3.0.2)
project(navigation_pkg)

set(CMAKE_CXX_STANDARD 17)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  tf
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  nav_core
  costmap_2d
  pluginlib
)
find_package(Eigen3 REQUIRED)

catkin_package(
  LIBRARIES astar_global_planner
  CATKIN_DEPENDS roscpp tf nav_msgs nav_core costmap_2d pluginlib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# C++ standalone node
add_executable(nav_astar_cpp src/nav_astar_node.cpp)
target_link_libraries(nav_astar_cpp ${catkin_LIBRARIES} Eigen3::Eigen)

# A* global planner plugin for move_base
add_library(astar_global_planner src/astar_global_planner.cpp)
target_link_libraries(astar_global_planner ${catkin_LIBRARIES})

# Python node (kept for benchmarking)
catkin_install_python(PROGRAMS scripts/nav_astar.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(TARGETS nav_astar_cpp
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS astar_global_planner
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(FILES astar_planner_plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
