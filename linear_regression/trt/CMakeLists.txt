cmake_minimum_required(VERSION 3.8)

project(linear)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Ofast -Wfatal-errors -pthread -fopenmp  -D_MWAITXINTRIN_H_INCLUDED")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--extended-lambda)

##################################DEP

find_package(Threads REQUIRED)

# CUDA
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda-10.2/include)
link_directories(/usr/local/cuda-10.2/lib64)

# TensorRT
include_directories(/usr/local/TensorRT-7.1.3.4/include)
link_directories(/usr/local/TensorRT-7.1.3.4/lib)

# OpenCV
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIRS})

##################################PROJ

add_executable(linear linear_regression.cpp engine_build.cpp engine_run.cpp)
target_link_libraries(linear nvinfer)
target_link_libraries(linear cudart)
target_link_libraries(linear opencv_world)

add_definitions(-O2 -pthread)

