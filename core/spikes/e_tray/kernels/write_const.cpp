// SPDX-License-Identifier: Apache-2.0
//
// Spike E trivial program: write 16 known words to page 0 of a DRAM buffer.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb = tt::CBIndex::c_0;
    constexpr auto dst_args = TensorAccessorArgs<0>();
    const auto dst = TensorAccessor(dst_args, dst_addr);

    const uint32_t l1 = get_write_ptr(cb);
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t i = 0; i < 16; i++) {
        p[i] = 0xC0FFEE00u + i;
    }
    noc_async_write_page(0, dst, l1);
    noc_async_write_barrier();
}
