# SPDX-License-Identifier: Apache-2.0

add_executable(spike_c spike_c.cpp)
target_link_libraries(spike_c PRIVATE TT::Metalium)
target_compile_definitions(spike_c PRIVATE TT_RASTER_SPIKE_C_KERNELS="${CMAKE_CURRENT_SOURCE_DIR}/kernels/")
