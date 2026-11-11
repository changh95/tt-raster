// SPDX-License-Identifier: Apache-2.0
//
// Writes the output planes of every tile block to DRAM: block k of this core, plane p goes to
// page (first_block + k) * (number of planes) + rank of the plane among those in RASTER_PLANE_MASK. The
// planes come from CBs c_16 (colour), c_17 (depth), c_18 (id), c_19 (G), c_20 (B).
// Runtime args: DRAM buffer address, first block, number of blocks.

#include <cstdint>

#include "raster_layout.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t first_block = get_arg_val<uint32_t>(1);
    const uint32_t n_blocks = get_arg_val<uint32_t>(2);

    constexpr uint32_t kMask = RASTER_PLANE_MASK;
    constexpr uint32_t cb_plane[5] = {tt::CBIndex::c_16, tt::CBIndex::c_17, tt::CBIndex::c_18, tt::CBIndex::c_19, tt::CBIndex::c_20};
    constexpr uint32_t kPlanes = __builtin_popcount(kMask);
    constexpr auto args = TensorAccessorArgs<0>();
    const auto dst = TensorAccessor(args, addr);

    for (uint32_t k = 0; k < n_blocks; k++) {
        uint32_t rank = 0;
        for (uint32_t p = 0; p < 5; p++) {
            if (!(kMask & (1u << p))) {
                continue;
            }
            cb_wait_front(cb_plane[p], 1);
            noc_async_write_page((first_block + k) * kPlanes + rank, dst, get_read_ptr(cb_plane[p]));
            noc_async_write_barrier();
            cb_pop_front(cb_plane[p], 1);
            rank++;
        }
    }
}
