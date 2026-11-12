# SPDX-License-Identifier: Apache-2.0

# Host tests: no device needed. `fast` tests run in about a second; `golden` renders every scene at
# 1920x1088 and compares with core/tests/golden.
function(ttr_test name label)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE ttraster_core)
        target_compile_definitions(${name} PRIVATE TT_RASTER_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden/")
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS ${label})
endfunction()

ttr_test(test_setup fast)
ttr_test(test_edge_exactness fast)
ttr_test(test_image fast)
ttr_test(test_scenes_small fast)
ttr_test(test_shading fast)
ttr_test(test_classify fast)
ttr_test(test_viewports fast)
ttr_test(test_reuse fast)
ttr_test(test_parallel fast)
ttr_test(test_perspective fast)
ttr_test(test_state fast)
ttr_test(test_shader fast)
ttr_test(test_golden golden)

# The shader manifest lives in the Tensix library; the test opens no device.
if(TT_RASTER_BUILD_DEVICE)
    add_executable(test_shader_warm test_shader_warm.cpp)
    target_link_libraries(test_shader_warm PRIVATE ttraster_tensix)
    add_test(NAME test_shader_warm COMMAND test_shader_warm)
    set_tests_properties(test_shader_warm PROPERTIES LABELS fast)
endif()
