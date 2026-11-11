// SPDX-License-Identifier: Apache-2.0
//
// Spike D receiver: poison the buffer, announce readiness, wait for the multicast flag, ack, wait
// for the final flag, then verify the buffer and write a result record to DRAM.
//
// Record (16 words): [ok, mismatches, logical x, logical y, first bad index, first bad value].

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

static inline uint32_t pattern(uint32_t i) { return ((i + 1) * 2654435761u) ^ 0xA5A5A5A5u; }

void kernel_main() {
    const uint32_t skip = get_arg_val<uint32_t>(0);  // 1 on the sender core when it lies inside the rectangle
    if (skip) {
        return;
    }
    const uint32_t sender_x = get_arg_val<uint32_t>(1);
    const uint32_t sender_y = get_arg_val<uint32_t>(2);
    const uint32_t ready_addr = get_semaphore(get_arg_val<uint32_t>(3));
    const uint32_t flag_addr = get_semaphore(get_arg_val<uint32_t>(4));
    const uint32_t ack_addr = get_semaphore(get_arg_val<uint32_t>(5));
    const uint32_t out_addr = get_arg_val<uint32_t>(6);
    const uint32_t page = get_arg_val<uint32_t>(7);
    const uint32_t bytes = get_arg_val<uint32_t>(8);
    const uint32_t lx = get_arg_val<uint32_t>(9);
    const uint32_t ly = get_arg_val<uint32_t>(10);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr);

    const uint32_t buf = get_write_ptr(0);
    const uint32_t scratch = get_write_ptr(1);
    volatile tt_l1_ptr uint32_t* b = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
    for (uint32_t i = 0; i < bytes / 4; i++) {
        b[i] = 0xDEADBEEFu;
    }
    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_addr);
    noc_semaphore_set(flag, 0);

    noc_semaphore_inc(get_noc_addr(sender_x, sender_y, ready_addr), 1);
    noc_semaphore_wait(flag, 1);
    noc_semaphore_inc(get_noc_addr(sender_x, sender_y, ack_addr), 1);
    noc_semaphore_wait(flag, 2);

    uint32_t bad = 0, first_idx = 0, first_val = 0;
    for (uint32_t i = 0; i < bytes / 4; i++) {
        const uint32_t w = b[i];
        if (w != pattern(i)) {
            if (bad == 0) {
                first_idx = i;
                first_val = w;
            }
            bad++;
        }
    }
    volatile tt_l1_ptr uint32_t* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    rec[0] = (bad == 0) ? 1u : 0u;
    rec[1] = bad;
    rec[2] = lx;
    rec[3] = ly;
    rec[4] = first_idx;
    rec[5] = first_val;
    noc_async_write_page(page, out, scratch);
    noc_async_write_barrier();
}
