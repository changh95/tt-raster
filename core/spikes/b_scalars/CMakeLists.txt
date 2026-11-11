# SPDX-License-Identifier: Apache-2.0

add_executable(spike_b spike_b.cpp)
target_link_libraries(spike_b PRIVATE TT::Metalium)
target_compile_definitions(spike_b PRIVATE TT_RASTER_SPIKE_B_KERNELS="${CMAKE_CURRENT_SOURCE_DIR}/kernels/")

add_executable(mad_bench mad_bench.cpp)
target_link_libraries(mad_bench PRIVATE TT::Metalium)
target_compile_definitions(mad_bench PRIVATE TT_RASTER_SPIKE_B_KERNELS="${CMAKE_CURRENT_SOURCE_DIR}/kernels/")
