# SPDX-License-Identifier: Apache-2.0

# The Gaussian-splatting plugin: scene, projection, PLY, CPU reference renderer and the renderer through the core's backends.
add_library(ttraster_gaussian STATIC src/gs/project.cpp src/gs/ply.cpp src/gs/render.cpp)
target_include_directories(ttraster_gaussian PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ttraster_gaussian PUBLIC ttraster_core)
set_target_properties(ttraster_gaussian PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(ttraster_gaussian PRIVATE -Wall -Wextra)

add_executable(ttr_gsplat tools/ttr_gsplat.cpp)
target_link_libraries(ttr_gsplat PRIVATE ttraster_gaussian)
if(TT_RASTER_BUILD_DEVICE)
    target_link_libraries(ttr_gsplat PRIVATE ttraster_tensix)
    target_compile_definitions(ttr_gsplat PRIVATE TT_RASTER_WITH_TENSIX=1)
endif()

add_executable(test_gaussian tests/test_gaussian.cpp)
target_link_libraries(test_gaussian PRIVATE ttraster_gaussian)
target_include_directories(test_gaussian PRIVATE ${CMAKE_SOURCE_DIR}/core/tests)
add_test(NAME test_gaussian COMMAND test_gaussian)
set_tests_properties(test_gaussian PROPERTIES LABELS fast)

add_library(ttraster_gaussian_c SHARED src/capi/ttraster_gs_capi.cpp)
set_target_properties(ttraster_gaussian_c PROPERTIES OUTPUT_NAME ttraster_gaussian POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ttraster_gaussian_c PRIVATE ttraster_gaussian)
if(TT_RASTER_BUILD_DEVICE)
    target_link_libraries(ttraster_gaussian_c PRIVATE ttraster_tensix)
    target_compile_definitions(ttraster_gaussian_c PRIVATE TT_RASTER_WITH_TENSIX=1)
endif()
