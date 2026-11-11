# SPDX-License-Identifier: Apache-2.0

# The Isaac Lab plugin: the batched-camera scene layer on top of the renderer core, its C interface (libttraster_scene.so),
# the Python bindings and the Isaac Lab renderer package.
add_library(ttraster_scene STATIC src/scene/scene.cpp)
target_include_directories(ttraster_scene PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ttraster_scene PUBLIC ttraster_core)
set_target_properties(ttraster_scene PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(ttraster_scene PRIVATE -Wall -Wextra)

add_library(ttraster_scene_c SHARED src/capi/ttraster_scene_capi.cpp)
set_target_properties(ttraster_scene_c PROPERTIES OUTPUT_NAME ttraster_scene POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ttraster_scene_c PRIVATE ttraster_scene)
if(TT_RASTER_BUILD_DEVICE)
    target_link_libraries(ttraster_scene_c PRIVATE ttraster_tensix)
    target_compile_definitions(ttraster_scene_c PRIVATE TT_RASTER_WITH_TENSIX=1)
endif()

function(ttr_scene_test name)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} PRIVATE ttraster_scene)
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/core/tests)
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS fast)
endfunction()
ttr_scene_test(test_scene)
ttr_scene_test(test_pipeline)
