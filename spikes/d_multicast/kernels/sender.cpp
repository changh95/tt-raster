// SPDX-License-Identifier: Apache-2.0
//
// Spike D sender: multicast one 4 KB buffer to a rectangle of cores with a semaphore handshake,
// then time (1) the multicast-plus-acknowledge round trip and (2) a train of back-to-back multicasts.
//
// Receivers prepare their buffer and bump the ready semaphore; the sender then multicasts the data
// and a flag value of 1; each receiver acks on the ack semaphore. After the bandwidth train the sender
// multicasts flag value 2, which tells the receivers to verify the final buffer contents.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

static inline uint32_t wall_clock_lo() {
    volatile tt_reg_ptr uint32_t* p = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    return p[0];
}

static inline uint32_t pattern(uint32_t i) { return ((i + 1) * 2654435761u) ^ 0xA5A5A5A5u; }

void kernel_main() {
    const uint32_t start_x = get_arg_val<uint32_t>(0);
    const uint32_t start_y = get_arg_val<uint32_t>(1);
    const uint32_t end_x = get_arg_val<uint32_t>(2);
    const uint32_t end_y = get_arg_val<uint32_t>(3);
    const uint32_t num_dests = get_arg_val<uint32_t>(4);    // cores in the rectangle (includes the sender if inside)
    const uint32_t n_receivers = get_arg_val<uint32_t>(5);  // cores that ack (rectangle minus the sender)
    const uint32_t ready_addr = get_semaphore(get_arg_val<uint32_t>(6));
    const uint32_t flag_addr = get_semaphore(get_arg_val<uint32_t>(7));
    const uint32_t ack_addr = get_semaphore(get_arg_val<uint32_t>(8));
    const uint32_t loopback = get_arg_val<uint32_t>(9);
    const uint32_t repeats = get_arg_val<uint32_t>(10);
    const uint32_t out_addr = get_arg_val<uint32_t>(11);
    const uint32_t bytes = get_arg_val<uint32_t>(12);
    const uint32_t result_page = get_arg_val<uint32_t>(13);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr);

    const uint32_t buf = get_write_ptr(0);      // CB c_0: same L1 address on every core of the program
    const uint32_t scratch = get_write_ptr(1);  // CB c_1: 64-byte result record
    volatile tt_l1_ptr uint32_t* b = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
    for (uint32_t i = 0; i < bytes / 4; i++) {
        b[i] = pattern(i);
    }

    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_addr);
    volatile tt_l1_ptr uint32_t* ack = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ack_addr);

    noc_semaphore_wait(ready, n_receivers);  // every receiver has cleared its buffer and flag
    noc_semaphore_set(ready, 0);

    const uint64_t data_mc = get_noc_multicast_addr(start_x, start_y, end_x, end_y, buf);
    const uint64_t flag_mc = get_noc_multicast_addr(start_x, start_y, end_x, end_y, flag_addr);

    auto send_data = [&]() {
        if (loopback) {
            noc_async_write_multicast_loopback_src(buf, data_mc, bytes, num_dests);
        } else {
            noc_async_write_multicast(buf, data_mc, bytes, num_dests);
        }
    };
    auto send_flag = [&](uint32_t value) {
        *flag = value;
        if (loopback) {
            noc_semaphore_set_multicast_loopback_src(flag_addr, flag_mc, num_dests);
        } else {
            noc_semaphore_set_multicast(flag_addr, flag_mc, num_dests);
        }
    };

    // Phase 1: data + flag, then wait for every ack.
    const uint32_t t0 = wall_clock_lo();
    send_data();
    send_flag(1);
    noc_semaphore_wait(ack, n_receivers);
    const uint32_t t1 = wall_clock_lo();
    noc_async_write_barrier();

    // Phase 2: a train of back-to-back multicasts, completion measured at the sender.
    const uint32_t t2 = wall_clock_lo();
    for (uint32_t r = 0; r < repeats; r++) {
        send_data();
    }
    noc_async_write_barrier();
    const uint32_t t3 = wall_clock_lo();

    send_flag(2);
    noc_async_write_barrier();

    volatile tt_l1_ptr uint32_t* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    rec[0] = t1 - t0;
    rec[1] = t3 - t2;
    rec[2] = repeats;
    rec[3] = bytes;
    rec[4] = num_dests;
    noc_async_write_page(result_page, out, scratch);
    noc_async_write_barrier();
}
