// SPDX-License-Identifier: Apache-2.0
//
// probe_device: open the visible chip(s) and print what the device API reports.
// Used for Phase 0.1 (worker grid, DRAM, L1) and for the "can separate processes
// hold separate chips" check (--hold keeps the device open).
//
//   TT_VISIBLE_DEVICES=0 probe_device unit [--hold SECONDS]
//   TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 probe_device mesh [--hold SECONDS]
//
// Chip selection is done with TT_VISIBLE_DEVICES, never in code, so the same
// binary serves single-chip and multi-chip runs.

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>

using namespace tt;
using namespace tt::tt_metal;

static void print_device(IDevice* d) {
    const CoreCoord logical = d->logical_grid_size();
    const CoreCoord worker = d->compute_with_storage_grid_size();
    std::cout << "  chip " << d->id() << ": logical grid " << logical.x << "x" << logical.y << ", worker grid "
              << worker.x << "x" << worker.y << " (" << worker.x * worker.y << " cores), DRAM channels "
              << d->num_dram_channels() << " x " << (d->dram_size_per_channel() >> 20) << " MiB, L1 per core "
              << (d->l1_size_per_core() >> 10) << " KiB\n";
}

int main(int argc, char** argv) {
    std::string mode = "unit";
    int hold_s = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "unit" || a == "mesh") {
            mode = a;
        } else if (a == "--hold" && i + 1 < argc) {
            hold_s = std::atoi(argv[++i]);
        } else {
            std::cerr << "usage: probe_device [unit|mesh] [--hold SECONDS]\n";
            return 2;
        }
    }

    const char* vis = std::getenv("TT_VISIBLE_DEVICES");
    std::cout << "TT_VISIBLE_DEVICES=" << (vis ? vis : "<unset>") << "\n";
    std::cout << "available devices: " << GetNumAvailableDevices() << ", PCIe devices: " << GetNumPCIeDevices()
              << "\n";
    std::cout << "system mesh shape: " << distributed::SystemMesh::instance().shape() << "\n";

    const auto t0 = std::chrono::steady_clock::now();
    std::shared_ptr<distributed::MeshDevice> mesh;
    if (mode == "unit") {
        mesh = distributed::MeshDevice::create_unit_mesh(0);
    } else {
        mesh = distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::SystemMesh::instance().shape()));
    }
    const auto t1 = std::chrono::steady_clock::now();
    std::cout << "opened " << mode << " mesh " << mesh->shape() << " with " << mesh->num_devices() << " device(s) in "
              << std::chrono::duration<double>(t1 - t0).count() << " s\n";
    for (IDevice* d : mesh->get_devices()) {
        print_device(d);
    }

    if (hold_s > 0) {
        std::cout << "holding the device(s) for " << hold_s << " s\n" << std::flush;
        std::this_thread::sleep_for(std::chrono::seconds(hold_s));
    }

    mesh->close();
    std::cout << "closed\n";
    return 0;
}
