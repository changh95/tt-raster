// SPDX-License-Identifier: Apache-2.0
//
// Spike F kernel: how fast can a Tensix core fetch and filter texels with its data-movement RISC-V?
//
//   mode 0: the texture is in L1 (circular buffer c_0). Per pixel: coordinates stepped in 16.16 fixed point, nearest (1 load), bilinear
//           (4 loads, 3 packed RGBA8 lerps), or gather only (filter 2: the 4 texels and the 2 weights are stored for the SFPU to filter).
//   mode 1: the texture is in DRAM as 64-byte pages of 4x4 texels. Per pixel `pages` NoC reads of one page each (1: the footprint is in
//           one page; 4: it crosses a page corner), `batch` reads in flight before a barrier, texels not filtered (the cost of the fetch).
//
// Result record (page = core index): [0] elapsed wall clock ticks, [1] pixels, [2] checksum.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t kMode = get_compile_time_arg_val(0);
constexpr uint32_t kBilinear = get_compile_time_arg_val(1);
constexpr uint32_t kTexLog2 = get_compile_time_arg_val(2);
constexpr uint32_t kBatch = get_compile_time_arg_val(3);
constexpr uint32_t kPages = get_compile_time_arg_val(4);

static inline uint64_t wall_clock() {
    volatile tt_reg_ptr uint32_t* lo = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    volatile tt_reg_ptr uint32_t* hi = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_H);
    uint32_t h = hi[0], l = lo[0];
    if (hi[0] != h) {
        h = hi[0];
        l = lo[0];
    }
    return (static_cast<uint64_t>(h) << 32) | l;
}

static inline uint32_t lerp8(uint32_t a, uint32_t b, uint32_t f) {
    const uint32_t rb = (((a & 0x00FF00FFu) * (256u - f) + (b & 0x00FF00FFu) * f) >> 8) & 0x00FF00FFu;
    const uint32_t ga = (((a >> 8) & 0x00FF00FFu) * (256u - f) + ((b >> 8) & 0x00FF00FFu) * f) & 0xFF00FF00u;
    return rb | ga;
}

void kernel_main() {
    const uint32_t npix = get_arg_val<uint32_t>(0);
    uint32_t u = get_arg_val<uint32_t>(1);
    uint32_t v = get_arg_val<uint32_t>(2);
    const uint32_t du = get_arg_val<uint32_t>(3);
    const uint32_t dv = get_arg_val<uint32_t>(4);
    const uint32_t out_addr = get_arg_val<uint32_t>(5);
    const uint32_t tex_addr = get_arg_val<uint32_t>(6);
    const uint32_t core_index = get_arg_val<uint32_t>(7);

    constexpr uint32_t W = 1u << kTexLog2, MASK = W - 1;
    constexpr auto out_args = TensorAccessorArgs<5>();
    const auto out = TensorAccessor(out_args, out_addr);
    uint32_t acc = 0;
    uint64_t t0, t1;

    if constexpr (kMode == 0) {
        volatile tt_l1_ptr uint32_t* tex = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));
        volatile tt_l1_ptr uint32_t* ring = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(2));
        for (uint32_t i = 0; i < W * W; i++) {
            tex[i] = (i + 1) * 2654435761u;
        }
        t0 = wall_clock();
        for (uint32_t i = 0; i < npix; i++) {
            const uint32_t x = (u >> 16) & MASK, y = (v >> 16) & MASK;
            if constexpr (kBilinear == 2) {
                const uint32_t fx = (u >> 8) & 255u, fy = (v >> 8) & 255u;
                const uint32_t x1 = (x + 1) & MASK, y1 = (y + 1) & MASK;
                volatile tt_l1_ptr uint32_t* o = ring + ((i & 255u) << 3);
                o[0] = tex[(y << kTexLog2) + x];
                o[1] = tex[(y << kTexLog2) + x1];
                o[2] = tex[(y1 << kTexLog2) + x];
                o[3] = tex[(y1 << kTexLog2) + x1];
                o[4] = fx;
                o[5] = fy;
                acc += o[0];
            } else if constexpr (kBilinear == 1) {
                const uint32_t fx = (u >> 8) & 255u, fy = (v >> 8) & 255u;
                const uint32_t x1 = (x + 1) & MASK, y1 = (y + 1) & MASK;
                const uint32_t c00 = tex[(y << kTexLog2) + x], c10 = tex[(y << kTexLog2) + x1];
                const uint32_t c01 = tex[(y1 << kTexLog2) + x], c11 = tex[(y1 << kTexLog2) + x1];
                acc += lerp8(lerp8(c00, c10, fx), lerp8(c01, c11, fx), fy);
            } else {
                acc += tex[(y << kTexLog2) + x];
            }
            u += du;
            v += dv;
        }
        t1 = wall_clock();
    } else {
        constexpr auto tex_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
        const auto tex = TensorAccessor(tex_args, tex_addr);
        const uint32_t slots = get_write_ptr(0);
        volatile tt_l1_ptr uint32_t* sl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slots);
        constexpr uint32_t row_pages = W >> 2, n_pages = row_pages * row_pages;
        t0 = wall_clock();
        for (uint32_t i = 0; i < npix; i += kBatch) {
            for (uint32_t k = 0; k < kBatch; k++) {
                const uint32_t x = (u >> 16) & MASK, y = (v >> 16) & MASK;
                const uint32_t page = (y >> 2) * row_pages + (x >> 2);
                uint32_t dst = slots + k * kPages * 64;
                noc_async_read(tex.get_noc_addr(page), dst, 64);
                if constexpr (kPages == 4) {
                    noc_async_read(tex.get_noc_addr((page + 1) % n_pages), dst + 64, 64);
                    noc_async_read(tex.get_noc_addr((page + row_pages) % n_pages), dst + 128, 64);
                    noc_async_read(tex.get_noc_addr((page + row_pages + 1) % n_pages), dst + 192, 64);
                }
                u += du;
                v += dv;
            }
            noc_async_read_barrier();
            acc += sl[0] + sl[(kBatch - 1) * kPages * 16];
        }
        t1 = wall_clock();
    }

    const uint32_t rec = get_write_ptr(1);
    volatile tt_l1_ptr uint32_t* r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rec);
    r[0] = static_cast<uint32_t>(t1 - t0);
    r[1] = npix;
    r[2] = acc;
    for (uint32_t i = 3; i < 16; i++) {
        r[i] = 0;
    }
    noc_async_write(rec, out.get_noc_addr(core_index), 64);
    noc_async_write_barrier();
}
