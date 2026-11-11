# SPDX-License-Identifier: Apache-2.0

add_executable(spike_d spike_d.cpp)
target_link_libraries(spike_d PRIVATE TT::Metalium)
target_compile_definitions(spike_d PRIVATE TT_RASTER_SPIKE_D_KERNELS="${CMAKE_CURRENT_SOURCE_DIR}/kernels/")
