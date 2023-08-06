cmake_minimum_required(VERSION 3.0.0)
project(ReadWebCamThread VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# add opencv lib
# Add the executable (main executable)
add_library(custom_libray config_reader.cpp display_frame.cpp ini_reader.cpp camloader.cpp)

add_executable(ReadWebCamThread main.cpp)

target_link_libraries(ReadWebCamThread ${OpenCV_LIBS})
target_link_libraries(ReadWebCamThread custom_libray)

# target_link_libraries(ReadWebCamThread ${TensorRT_LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
