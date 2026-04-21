cmake_minimum_required(VERSION 3.10)

project(Yolov9Funditec)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(OpenCV REQUIRED)

add_executable(detect main.cpp YOLOv9/YOLOv9.cpp)

# Add targets (paths)
target_include_directories(detect PRIVATE # -I
    ${OpenCV_INCLUDE_DIRS}
    /home/lola/YOLOv9-cpp/onnxruntime-linux-x64-1.17.0/include
    /home/lola/YOLOv9-cpp/YOLOv9
)
target_link_libraries(detect PRIVATE 
${OpenCV_LIBS}
/home/lola/YOLOv9-cpp/onnxruntime-linux-x64-1.17.0/lib/libonnxruntime.so
) 
set_target_properties(detect PROPERTIES
    BUILD_RPATH "/home/lola/YOLOv9-cpp/onnxruntime-linux-x64-1.17.0/lib"
    INSTALL_RPATH "/home/lola/YOLOv9-cpp/onnxruntime-linux-x64-1.17.0/lib"
)