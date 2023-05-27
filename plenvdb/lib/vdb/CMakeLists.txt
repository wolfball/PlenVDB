project(plenvdb)
cmake_minimum_required(VERSION 2.8)
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")

find_package(CUDA)
find_package(PythonLibs 3.7 REQUIRED)

include_directories(
    ${CMAKE_INSTALL_PREFIX}/include
    ${PYTHON_INCLUDE_DIRS}
    /home/hyan/.conda/envs/plenvdb/lib/python3.7/site-packages/pybind11/include # change to your pybind path
)

LINK_DIRECTORIES(
    ${CMAKE_INSTALL_PREFIX}/lib 
    /usr/lib/x86_64-linux-gnu
    /usr/local/cuda/lib64
)

# add_executable
cuda_add_library(
    plenvdb SHARED  
    plenvdb.cpp
    plenvdb.h
    plenvdb.cuh
    plenvdb.cu
    densityvdb.cu
    colorvdb.cu
    renderer.cu
    OPTIONS -arch sm_70
)

target_link_libraries(
    plenvdb
    libtbb.so 
    libpthread.so
    libopenvdb.so
    ${PYTHON_LIBRARIES} 
    ${CUDA_LIBRARIES}
    -lcublas -lcurand
)
set_target_properties(plenvdb PROPERTIES PREFIX "")
