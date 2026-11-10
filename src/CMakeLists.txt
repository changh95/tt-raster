# SPDX-License-Identifier: Apache-2.0

# Host-side library shared by the CPU reference, the tools, the tests and the Tensix renderer's
# host code. No tt-metal dependency.
find_package(ZLIB REQUIRED)

add_library(ttraster_core STATIC
    ttraster/frame.cpp
    ttraster/setup.cpp
    ttraster/tile.cpp
    ttraster/reference.cpp
    ttraster/oracle.cpp
    ttraster/scenes.cpp
    ttraster/image.cpp)
target_include_directories(ttraster_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ttraster_core PUBLIC ZLIB::ZLIB)
target_compile_features(ttraster_core PUBLIC cxx_std_20)
# Fused operations are written explicitly (std::fmaf); never let the compiler contract anything else.
target_compile_options(ttraster_core PUBLIC -ffp-contract=off)
target_compile_options(ttraster_core PRIVATE -Wall -Wextra)

if(TT_RASTER_BUILD_DEVICE)
    add_subdirectory(tensix)
endif()
