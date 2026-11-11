# SPDX-License-Identifier: Apache-2.0

# The renderer core: frame description, CPU reference rasterizer, shader interpreter, the Tensix renderer (when built with
# tt-metal), the C frame API (libttraster.so) and the tools, tests and spikes that belong to them. No OpenGL, Vulkan, Isaac Lab
# or Newton code lives here; those are api/ and plugins/.
find_package(ZLIB REQUIRED)

add_library(ttraster_core STATIC
    core/src/ttraster/frame.cpp
    core/src/ttraster/shader.cpp
    core/src/ttraster/setup.cpp
    core/src/ttraster/tile.cpp
    core/src/ttraster/reference.cpp
    core/src/ttraster/backend.cpp
    core/src/ttraster/oracle.cpp
    core/src/ttraster/scenes.cpp
    core/src/ttraster/image.cpp
    core/src/ttraster/parallel.cpp)
target_include_directories(ttraster_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ttraster_core PUBLIC ZLIB::ZLIB)
target_compile_features(ttraster_core PUBLIC cxx_std_20)
set_target_properties(ttraster_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Fused operations are written explicitly (std::fmaf); never let the compiler contract anything else.
target_compile_options(ttraster_core PUBLIC -ffp-contract=off)
target_compile_options(ttraster_core PRIVATE -Wall -Wextra)

if(TT_RASTER_BUILD_DEVICE)
    add_subdirectory(src/tensix)
endif()

# C interface: libttraster.so (CPU reference backend; Tensix backend when built with tt-metal).
add_library(ttraster SHARED core/src/capi/ttraster_capi.cpp)
target_link_libraries(ttraster PRIVATE ttraster_core)
target_include_directories(ttraster PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(ttraster PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(TT_RASTER_BUILD_DEVICE)
    target_link_libraries(ttraster PRIVATE ttraster_tensix)
    target_compile_definitions(ttraster PRIVATE TT_RASTER_WITH_TENSIX=1)
endif()

add_subdirectory(tests)
add_subdirectory(tools)
if(TT_RASTER_BUILD_DEVICE)
    add_subdirectory(spikes)
endif()
