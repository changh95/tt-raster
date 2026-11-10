# SPDX-License-Identifier: Apache-2.0
#
# tt-raster, built out of tree against a built tt-metal checkout.
# Use scripts/build.sh; it passes the toolchain file and the tt-metal location.

cmake_minimum_required(VERSION 3.22...4.2)
project(tt_raster LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Where tt-metal was built and installed (build_metal.sh installs into its build dir).
set(TT_METAL_HOME "$ENV{TT_METAL_HOME}" CACHE PATH "tt-metal checkout")
if(NOT TT_METAL_HOME)
    message(FATAL_ERROR "Set TT_METAL_HOME to a built tt-metal checkout")
endif()
set(TT_METAL_BUILD_DIR "${TT_METAL_HOME}/build_Release" CACHE PATH "tt-metal build directory")
# The package config in the build dir root is not self-contained; the installed one under lib/cmake is.
list(APPEND CMAKE_PREFIX_PATH "${TT_METAL_BUILD_DIR}" "${TT_METAL_BUILD_DIR}/lib/cmake")
set(tt-metalium_DIR "${TT_METAL_BUILD_DIR}/lib/cmake/tt-metalium")

find_package(tt-metalium REQUIRED)

add_subdirectory(tools)
add_subdirectory(spikes)
