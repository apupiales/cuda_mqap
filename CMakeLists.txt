cmake_minimum_required(VERSION 3.24)
project(cuda_mqap LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default: GeForce RTX 20xx (Turing). Override with -DCMAKE_CUDA_ARCHITECTURES="75;86;89".
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

add_library(mqap_core STATIC
    src/instance.cpp
    src/fitness.cu
    src/nsga2.cu
    src/operators.cu
    src/local_search.cu
    src/solver.cu
)
target_include_directories(mqap_core PUBLIC include)
# Debug builds synchronize after every kernel to report errors at the failing launch.
target_compile_definitions(mqap_core PUBLIC $<$<CONFIG:Debug>:MQAP_SYNC_CHECK>)
target_compile_options(mqap_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_executable(cuda_mqap src/main.cpp)
target_link_libraries(cuda_mqap PRIVATE mqap_core)

enable_testing()
add_executable(test_kernels tests/test_kernels.cu)
target_link_libraries(test_kernels PRIVATE mqap_core)
add_test(NAME kernels COMMAND test_kernels ${CMAKE_SOURCE_DIR}/mQAPData)
