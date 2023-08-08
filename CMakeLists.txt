cmake_minimum_required(VERSION 3.0.0)
project(ReadWebCamThread VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# add opencv lib
# Add the executable (main executable)
# it will create library file in build folder but you have to keep all the source file in SOURCE Directory
# add_library(custom_libray config_reader.cpp ini_reader.cpp camloader.cpp yolov5_model.cpp)

# Adding include directory
include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB SOURCE_FILES src/*.cpp)
add_executable(ReadWebCamThread main.cpp ${SOURCE_FILES})

target_link_libraries(ReadWebCamThread ${OpenCV_LIBS})
# target_link_libraries(ReadWebCamThread custom_libray)

# target_link_libraries(ReadWebCamThread ${TensorRT_LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
