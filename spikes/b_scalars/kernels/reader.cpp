// SPDX-License-Identifier: Apache-2.0
//
// Reads one scalar tile into CB c_1 and n_coef coefficient tiles into CB c_2.

#include <cstdint>

void kernel_main() {
    const uint32_t scal_addr = get_arg_val<uint32_t>(0);
    const uint32_t coef_addr = get_arg_val<uint32_t>(1);
    const uint32_t n_coef = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_scal = tt::CBIndex::c_1;
    constexpr uint32_t cb_coef = tt::CBIndex::c_2;

    constexpr auto scal_args = TensorAccessorArgs<0>();
    constexpr auto coef_args = TensorAccessorArgs<scal_args.next_compile_time_args_offset()>();
    const auto scal = TensorAccessor(scal_args, scal_addr);
    const auto coef = TensorAccessor(coef_args, coef_addr);

    cb_reserve_back(cb_scal, 1);
    noc_async_read_page(0, scal, get_write_ptr(cb_scal));
    noc_async_read_barrier();
    cb_push_back(cb_scal, 1);

    for (uint32_t i = 0; i < n_coef; i++) {
        cb_reserve_back(cb_coef, 1);
        noc_async_read_page(i, coef, get_write_ptr(cb_coef));
        noc_async_read_barrier();
        cb_push_back(cb_coef, 1);
    }
}
