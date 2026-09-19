# Copyright (C) 2019-2026 Andres Pupiales Arevalo <apupiales@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 3.24)
project(cuda_mqap LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default: native code for sm_75/80/86/89 plus sm_89 PTX (JIT-compiled for newer GPUs).
# Override with e.g. -DCMAKE_CUDA_ARCHITECTURES=native
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real 86-real 89)
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
