# SPDX-License-Identifier: Apache-2.0

# Host side of the Tensix renderer. Needs tt-metal.
add_library(ttraster_tensix STATIC tensix_renderer.cpp multichip.cpp)
target_link_libraries(ttraster_tensix PUBLIC ttraster_core TT::Metalium)
target_compile_definitions(ttraster_tensix PUBLIC TT_RASTER_TENSIX_KERNELS="${CMAKE_CURRENT_SOURCE_DIR}/kernels/")
