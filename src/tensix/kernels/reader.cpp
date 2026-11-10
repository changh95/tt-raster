// SPDX-License-Identifier: Apache-2.0
//
// Streams a core's pages of tile blocks from DRAM into CB c_0, in order.
// Runtime args: DRAM buffer address, first page, number of pages.

#include <cstdint>

#include "raster_layout.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t first_page = get_arg_val<uint32_t>(1);
    const uint32_t n_pages = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_tri = tt::CBIndex::c_0;
    constexpr auto args = TensorAccessorArgs<0>();
    const auto src = TensorAccessor(args, addr);

    for (uint32_t p = 0; p < n_pages; p++) {
        cb_reserve_back(cb_tri, 1);
        noc_async_read_page(first_page + p, src, get_write_ptr(cb_tri));
        noc_async_read_barrier();
        cb_push_back(cb_tri, 1);
    }
}
