cmake_minimum_required(VERSION 3.0.0)
project(yolov5onnx VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# find tensorrt and include its header and library
set(TensorRT_DIR C:/Program Files/TensorRT-8.5.3.1)
find_library(TensorRT_LIBRARIES nvinfer HINTS ${TensorRT_DIR}/lib)
include_directories(${TensorRT_DIR}/include)

add_executable(yolov5onnx main.cpp)

target_link_libraries(yolov5onnx ${OpenCV_LIBS})
target_link_libraries(yolov5onnx ${TensorRT_LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
