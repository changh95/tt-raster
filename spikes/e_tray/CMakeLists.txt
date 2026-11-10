# SPDX-License-Identifier: Apache-2.0

add_executable(spike_e spike_e.cpp)
target_link_libraries(spike_e PRIVATE TT::Metalium)
target_compile_definitions(spike_e PRIVATE TT_RASTER_SPIKE_E_KERNELS="${CMAKE_CURRENT_SOURCE_DIR}/kernels/")
