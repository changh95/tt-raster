# SPDX-License-Identifier: Apache-2.0

add_executable(probe_device probe_device.cpp)
target_link_libraries(probe_device PRIVATE TT::Metalium)
