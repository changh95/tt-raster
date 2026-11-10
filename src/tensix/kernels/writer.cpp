// SPDX-License-Identifier: Apache-2.0
//
// Writes the output planes of every tile block to DRAM: block k of this core, plane p goes to
// page (first_block + k) * RASTER_OUT_PLANES + p of the output buffer. The planes come from CBs
// c_16 (colour), c_17 (depth), c_18 (id); RASTER_OUT_PLANES is 1 (colour only) or 3.
// Runtime args: DRAM buffer address, first block, number of blocks.

#include <cstdint>

#include "raster_layout.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t first_block = get_arg_val<uint32_t>(1);
    const uint32_t n_blocks = get_arg_val<uint32_t>(2);

    constexpr uint32_t kPlanes = RASTER_OUT_PLANES;
    constexpr uint32_t cb_plane[ttr::layout::kOutPlanes] = {tt::CBIndex::c_16, tt::CBIndex::c_17, tt::CBIndex::c_18};
    constexpr auto args = TensorAccessorArgs<0>();
    const auto dst = TensorAccessor(args, addr);

    for (uint32_t k = 0; k < n_blocks; k++) {
        for (uint32_t p = 0; p < kPlanes; p++) {
            cb_wait_front(cb_plane[p], 1);
            noc_async_write_page((first_block + k) * kPlanes + p, dst, get_read_ptr(cb_plane[p]));
            noc_async_write_barrier();
            cb_pop_front(cb_plane[p], 1);
        }
    }
}
