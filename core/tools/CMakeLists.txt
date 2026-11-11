# SPDX-License-Identifier: Apache-2.0

# Host tools: render a scene with the CPU reference, compare images.
add_executable(ttr_render ttr_render.cpp)
target_link_libraries(ttr_render PRIVATE ttraster_core)

add_executable(ttr_diff ttr_diff.cpp)
target_link_libraries(ttr_diff PRIVATE ttraster_core)

add_executable(ttr_compare ttr_compare.cpp)
target_link_libraries(ttr_compare PRIVATE ttraster_core)

# Device tool: needs tt-metal.
if(TT_RASTER_BUILD_DEVICE)
    add_executable(probe_device probe_device.cpp)
    target_link_libraries(probe_device PRIVATE TT::Metalium)

    add_executable(ttr_device ttr_device.cpp)
    target_link_libraries(ttr_device PRIVATE ttraster_tensix)
    target_compile_definitions(ttr_device PRIVATE TT_RASTER_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../tests/golden/")

    add_executable(ttr_multichip ttr_multichip.cpp)
    target_link_libraries(ttr_multichip PRIVATE ttraster_tensix)

    add_executable(ttr_shader_ops ttr_shader_ops.cpp)
    target_link_libraries(ttr_shader_ops PRIVATE ttraster_tensix)

    add_executable(ttr_shader_bench ttr_shader_bench.cpp)
    target_link_libraries(ttr_shader_bench PRIVATE ttraster_tensix)

    add_executable(ttr_shader_fuzz ttr_shader_fuzz.cpp)
    target_link_libraries(ttr_shader_fuzz PRIVATE ttraster_tensix)
endif()
