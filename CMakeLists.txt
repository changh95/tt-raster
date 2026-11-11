# SPDX-License-Identifier: Apache-2.0
#
# tt-raster. Layout: core/ (the renderer), api/ (OpenGL and Vulkan drivers), plugins/ (Isaac Lab, ...).
# Two parts of the build:
#   * the host library, tools and tests (plain C++20 plus zlib, no tt-metal);
#   * the device spikes and the Tensix renderer, built against a built tt-metal checkout.
# Use scripts/build.sh. Without TT_METAL_HOME only the host part is built.

cmake_minimum_required(VERSION 3.22...4.2)
project(tt_raster LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Where tt-metal was built and installed (build_metal.sh installs into its build dir).
set(TT_METAL_HOME "$ENV{TT_METAL_HOME}" CACHE PATH "tt-metal checkout (empty: build the host part only)")
if(TT_METAL_HOME)
    set(TT_RASTER_BUILD_DEVICE ON)
else()
    set(TT_RASTER_BUILD_DEVICE OFF)
endif()

enable_testing()

if(TT_RASTER_BUILD_DEVICE)
    set(TT_METAL_BUILD_DIR "${TT_METAL_HOME}/build_Release" CACHE PATH "tt-metal build directory")
    # The package config in the build dir root is not self-contained; the installed one under lib/cmake is.
    list(APPEND CMAKE_PREFIX_PATH "${TT_METAL_BUILD_DIR}" "${TT_METAL_BUILD_DIR}/lib/cmake")
    set(tt-metalium_DIR "${TT_METAL_BUILD_DIR}/lib/cmake/tt-metalium")
    find_package(tt-metalium REQUIRED)
endif()

# core/     the renderer (library, tools, tests, device spikes)
# api/      graphics API drivers (api/gl, api/vulkan): built with Mesa by scripts, not by this CMake project
# plugins/  consumers of the core that are not graphics APIs (plugins/isaaclab; later a Gaussian-splatting renderer)
add_subdirectory(core)
option(TT_RASTER_BUILD_PLUGINS "Build the plugins" ON)
if(TT_RASTER_BUILD_PLUGINS)
    add_subdirectory(plugins/isaaclab)
endif()
