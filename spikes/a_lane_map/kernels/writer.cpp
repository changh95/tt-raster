// SPDX-License-Identifier: Apache-2.0
//
// Writes n_tiles FP32 tiles from CB c_16 to an interleaved DRAM buffer.

#include <cstdint>

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr auto dst_args = TensorAccessorArgs<0>();
    const auto dst = TensorAccessor(dst_args, dst_addr);

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_out, 1);
        noc_async_write_page(i, dst, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
