# SPDX-License-Identifier: Apache-2.0

add_executable(spike_a spike_a.cpp)
target_link_libraries(spike_a PRIVATE TT::Metalium)
target_compile_definitions(spike_a PRIVATE TT_RASTER_SPIKE_A_KERNELS="${CMAKE_CURRENT_SOURCE_DIR}/kernels/")
