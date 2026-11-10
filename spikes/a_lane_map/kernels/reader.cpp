// SPDX-License-Identifier: Apache-2.0
//
// Reads n_tiles FP32 tiles (pages) from an interleaved DRAM buffer into CB c_0.

#include <cstdint>

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto src = TensorAccessor(src_args, src_addr);

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_reserve_back(cb_in, 1);
        noc_async_read_page(i, src, get_write_ptr(cb_in));
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
