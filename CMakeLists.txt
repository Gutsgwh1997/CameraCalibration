cmake_minimum_required(VERSION 2.8)
project(calibration_mono)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)
add_definitions(-O3)
add_definitions(-std=c++11)

find_package(Boost COMPONENTS filesystem REQUIRED)
find_package(OpenCV  REQUIRED)

include_directories("/usr/include/eigen3")
include_directories(${Boost_LIBRARY_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(CameraCalibration calibration_mono.cpp)
target_link_libraries(CameraCalibration ${OpenCV_LIBS} ${Boost_LIBRARIES})
