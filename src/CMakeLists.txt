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
    ttraster/image.cpp
    scene/scene.cpp)
target_include_directories(ttraster_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ttraster_core PUBLIC ZLIB::ZLIB)
target_compile_features(ttraster_core PUBLIC cxx_std_20)
set_target_properties(ttraster_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Fused operations are written explicitly (std::fmaf); never let the compiler contract anything else.
target_compile_options(ttraster_core PUBLIC -ffp-contract=off)
target_compile_options(ttraster_core PRIVATE -Wall -Wextra)

if(TT_RASTER_BUILD_DEVICE)
    add_subdirectory(tensix)
endif()

# C interface: libttraster.so (CPU reference backend; Tensix backend when built with tt-metal).
add_library(ttraster SHARED capi/ttraster_capi.cpp)
target_link_libraries(ttraster PRIVATE ttraster_core)
target_include_directories(ttraster PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(ttraster PROPERTIES POSITION_INDEPENDENT_CODE ON)
