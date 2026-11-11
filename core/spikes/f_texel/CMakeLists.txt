# SPDX-License-Identifier: Apache-2.0

add_executable(spike_f spike_f.cpp)
target_link_libraries(spike_f PRIVATE TT::Metalium)
target_compile_definitions(spike_f PRIVATE TT_RASTER_SPIKE_F_KERNELS="${CMAKE_CURRENT_SOURCE_DIR}/kernels/")
