# Phase 02 reading notes: Blackhole SFPU, Dest, scalars, L1 and NoC (tt-metal v0.81.1)

Read-only research. No hardware was touched and no compiler or device program was run; everything below comes from reading files.

## Conventions

- `TM` = `tt-metal` (`git describe` = `v0.81.1`). Every path written without a prefix is relative to `TM`.
- `ISA:` = the `tenstorrent/tt-isa-documentation` repo, shallow clone at commit `f1b8bbefd6e233bee95fd70d6c2826ce901c7c25` (2026-11-08). Cited as `ISA:<repo-relative path>:<line>`. Online: `https://github.com/tenstorrent/tt-isa-documentation/blob/main/<path>`.
- Several BlackholeA0 ISA pages are stubs that point at the WormholeB0 page: `LReg.md`, `SFPCONFIG.md`, `SFPSETCC.md` (stated identical), `FloatBitPatterns.md`, `NoC/Alignment.md` (stated "similar, but not identical"). For those I cite the WormholeB0 page and say so.
- INFERRED = my own deduction, not stated in any source. "Installed code wins over docs" was applied; conflicts are listed in the last section.
- sfpi headers used for evidence live in the checkout at `runtime/sfpi/include/` (sfpi 7.89.0 per `runtime/sfpi-version.cmake`).

---

## V1. SFPU width and lane mapping onto Dest

**Answer.** The SFPU is 32 lanes of 32 bit (LReg viewed as a 4x8 lane grid). One `SFPLOAD`/`SFPSTORE` moves 32 datums between an LReg and Dest: 4 consecutive Dest rows by the 8 columns of one parity, lane L going to Dest row `(Addr&~3) + L/8`, column `2*(L&7) + ((Addr>>1)&1)`. sfpi `dst_reg[i]` encodes `Addr = 2*i` and `dst_reg++` adds 2 to the Dst RWC, so one `++` step flips between the even-column and odd-column halves of the current 4-row group and every second step moves down 4 rows: 8 steps cover 16 rows = one 16x16 face (256 datums), 32 indices cover a 32x32 tile (64 rows), and the LLK moves to the next face with `+16` rows (two `inc_dst_addr<8>`). With the usual unpack layout (tile t, face f at Dest rows `64t+16f ...`, rows row-major inside a face, face order TL, TR, BL, BR), vector k (0..7) of a face and lane L hold face element row `4*(k>>1) + (L>>3)`, column `2*(L&7) + (k&1)` (INFERRED composition; the unpacker/packer layout part is not in the SFPU docs).

**Evidence.**
- 32 lanes x 32 bit: `ISA:BlackholeA0/TensixTile/TensixCoprocessor/VectorUnit.md:3`; 4x8 lane grid: `ISA:WormholeB0/TensixTile/TensixCoprocessor/LReg.md:30-35`.
- Lane to (row, column): `ISA:BlackholeA0/TensixTile/TensixCoprocessor/SFPLOAD.md:85-111` ("top 8 bits of Addr end up selecting an aligned group of four rows ... next bit selects between even and odd columns, and the low bit goes unused"; `Row = (Addr & ~3) + (Lane / 8)`; `Column = (Lane & 7) * 2`; `Column += 1` if `(Addr & 2)`). Identical for store: `SFPSTORE.md:86-106`. Dest is 1024 rows x 16 columns of 16 bit or 512 x 16 of 32 bit: `Dst.md:3`.
- sfpi address encoding: `runtime/sfpi/include/sfpi_constants.h:16-17` (`SFP_DESTREG_STRIDE = 2`, `SFP_DESTREG_COUNT = 0x400/2`), `runtime/sfpi/include/sfpi_classes.h:390-398` (`operator[]` gives `vReg(ix * SFP_DESTREG_STRIDE)`; `operator++` is `__builtin_rvtt_ttincrwc(0, SFP_DESTREG_STRIDE * ix, 0, 0)`).
- "64/SFP_DESTREG_STRIDE = 32 rows per tile": comment at `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h:119-120`; `dst_tile_size_sfpi = 32` at `:120-122`.
- Faces: `_llk_math_eltwise_sfpu_apply_vector_mode_` calls the SFPU function once per face (4 faces for `VectorMode::RC`, faces 0+1 for `R`, faces 0+2 for `C`) and steps `+16` rows between faces (`inc_dst_addr<8>` twice): `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_sfpu_common.h:28-32, 46-85`. Every example function loops 8 times per face: `tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_custom_add.h:21`, `.../ckernel_sfpu_smoothstep.h:14-15`.
- Tile base and face base in Dest: `tt_metal/tt-llk/tt_llk_blackhole/common/inc/cmath_common.h:263-279` (`dst_index = tile_index << DstTileSizeLog2[32x32]` = `tile*64`, plus half-bank base), `:340-346` (`get_dest_index_in_faces = (dst_index << 2) + face_index`, 16-row faces back to back). Datacopy moves 8 rows per step with `dest.incr = 8`: `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_datacopy.h:358-360, 450-456`.
- Host face order inside a tile (what lands in L1, then Dest): see X2 (`tilize_utils.cpp:313-316`).
- `remap_addrs`/`swizzle_32b` (which change the Dest row to physical-row mapping, `ISA:BlackholeA0/TensixTile/TensixCoprocessor/Dst.md:23-40`) are only toggled for untilize: `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_common.h:70-82`.
- `vConstTileId` / LReg[15] holds `2*lane` (0, 2, ..., 62): `ISA:WormholeB0/TensixTile/TensixCoprocessor/LReg.md:15`, `runtime/sfpi/include/sfpi.h:616`.

**Confidence.** High for lane to (Dest row, column) and for the address arithmetic. Medium for the Dest row/column to tile (row, column) step, since it rests on the unpacker/packer convention.

**Needs hardware measurement.**
1. Round trip: unpack a known tile, have the SFPU store `vConstTileId`-derived values (plus the iteration index) back, pack out, and read which tile element each (k, lane) hit. Do it in BF16 Dest and in FP32 Dest, SyncHalf and SyncFull, for tile index 0 and index >0.
2. Confirm face order (TL, TR, BL, BR) and that `dst_reg[0..7]` is face 0 (the prose doc says otherwise, see "Discrepancies").
3. Confirm `remap_addrs`/`swizzle_32b` are off in the non-untilize path (the reset value was not found in the files read).

---

## V2. Dest capacity (16-bit vs FP32, half-sync)

**Answer.** Blackhole Dest is `DEST_REGISTER_FULL_SIZE = 64*16 = 1024` rows (16-bit view), half = 512; a 32x32 tile is 64 rows. 16-bit mode: 16 tiles (full sync) or 8 tiles (half sync); FP32 mode halves the row count: 8 tiles (full sync) or 4 tiles (half sync). Host controls: `ComputeConfig::fp32_dest_acc_en` (default false) and `ComputeConfig::dst_full_sync_en` (default false, which means half-sync double buffering); the JIT turns them into `DST_ACCUM_MODE` and `DST_SYNC_MODE`.

**Evidence.**
- Constants: `tt_metal/hw/inc/internal/tt-1xx/blackhole/tensix_types.h:187-197` (`DEST_FACE_WIDTH = 16`, `DEST_FACE_HEIGHT = 16`, `DEST_REGISTER_FULL_SIZE = 64 * DEST_FACE_HEIGHT`, `DEST_REGISTER_HALF_SIZE = FULL / 2`, `DEST_REGISTER_FULL_SIZE_BYTES = FULL * 2 * 16`, `..._HALF_SIZE_BYTES = FULL_BYTES / 2`). `DEST_NUM_TILES_FP16 = (FULL * 16) / (32*32)` = 16, half = 8: `tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_defs.h:126-127`.
- Tile-count formula: `tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel.h:857-863`: `SYNC_MODE == SyncHalf ? (ACCUM_MODE ? HALF >> 1 : HALF) : (ACCUM_MODE ? FULL >> 1 : FULL)`, then `>> DstTileSizeLog2[32x32]`; `DstTileSizeLog2 = {6, 5, 4}` at `:835-839`. Runtime twin reading `ALU_ACC_CTRL_Fp32_enabled` at `:872-880`; used in `LLK_ASSERT(dst_index < ...)` at `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_sfpu_common.h:42`.
- Host formula: `ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.cpp:142-150` (`FULL*16/(tile area)`; `/2` if `!dst_full_sync_en`; `/2` if `fp32_dest_acc_en`).
- Table in the docs (8/16 for 16-bit, 4/8 for FP32): `docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst:122-148`; "Accessing Dst register tiles beyond the number available ... undefined behavior" at `:170`.
- Config plumbing: `tt_metal/api/tt-metalium/kernel_types.hpp:108-110` (`fp32_dest_acc_en`, `dst_full_sync_en`, `unpack_to_dest_mode`); `tt_metal/jit_build/genfiles.cpp:1131-1137` (`constexpr bool DST_ACCUM_MODE`, `#define DST_SYNC_MODE DstSync::Sync{Full|Half}`). Hardware bits set by `_llk_math_hw_configure_`: `ALU_ACC_CTRL_Fp32_enabled` and `ALU_ACC_CTRL_SFPU_Fp32_enabled`, `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_common.h:40-41`. Mid-kernel toggle (`set_fp32_dest_acc<>`, `restore_fp32_dest_acc<>`): `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:150-250`.
- Full-precision FP32 into Dest needs `UnpackToDestMode::UnpackToDestFp32` in `ComputeConfig::unpack_to_dest_mode` (otherwise FP32 operands go through SrcA at reduced precision): `tt_metal/api/tt-metalium/base_types.hpp:37-50`.
- Half-sync flips the bank on every commit: `dest_offset_id` toggles and `DEST_TARGET_REG_CFG_MATH_Offset` is reset to 0 or `DEST_REGISTER_HALF_SIZE` (`tt_metal/tt-llk/tt_llk_blackhole/common/inc/cmath_common.h:305-311`, `tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel.h:407-416`, `llk_math_common.h:107-115`). The semaphore seed is 1 for SyncFull and 2 for SyncHalf: `llk_math_common.h:157-169`.

**Confidence.** High.

**Needs hardware measurement.** Optional: write distinct sentinels to tiles 0..N-1 in each of the four (dtype, sync) modes and pack them out to confirm the usable counts 16/8 and 8/4 (the behavior past the limit is documented as undefined).

---

## V3. One SFPU function reading/writing several Dest tiles (binary/ternary)

**Answer.** Yes. Binary and ternary SFPU wrappers pass several `dst_index` values to the SFPU function and start the Dest base at 0; the function then addresses each tile with a fixed per-tile offset of 32 sfpi vectors (= 64 rows = `Imm10` of `64*idx`). The per-iteration `dst_reg++` (or the per-face `+16` rows) is shared, so all operands advance together. Unary ops differ: the wrapper itself sets the Dest base to the single tile.

**Evidence.**
- Unary sets the base to the tile: `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h:15` (`_llk_math_eltwise_sfpu_start_(dst_index)`), `llk_math_eltwise_sfpu_common.h:17-21`. Binary and ternary start at 0 and forward all indices: `llk_math_eltwise_binary_sfpu_params.h:21-24`, `llk_math_eltwise_ternary_sfpu_params.h:22-25`.
- Example (installed experimental op): `tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_custom_add.h:14-26` (`in0_base_idx = dst_index_in0 * 32`, `a = dst_reg[in0_base_idx + i]`, `b = dst_reg[in1_base_idx + i]`, `dst_reg[out_base_idx + i] = a + b`). Kernel side: `tt_metal/programming_examples/custom_sfpi_add/kernels/compute/tiles_add.cpp:38-41` (`SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, my_add_tile_face, idx_dst0, idx_dst1, idx_out0, VectorMode::RC)`), call at `:69` (`my_add_tile(0, 1, 0)`).
- In-tree production binary op: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h:119-122` (`dst_reg[dst_index_in0 * dst_tile_size_sfpi]`, `dst_reg[dst_index_in1 * ...]`) with `sfpi::dst_reg++` at `:209`.
- Ternary with raw instructions: `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_where.h:25-32, 36-45` (`offset = (dst_index * 32) << 1`, three `TT_SFPLOAD(..., offset0/1/2)` and `TT_SFPSTORE(..., offset3)`).
- Operand index limit: `LLK_ASSERT` per index against `get_dest_max_tiles_rt` in `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_macros.h:25-40`. `Imm10` is 10 bits (`ISA:BlackholeA0/TensixTile/TensixCoprocessor/SFPLOAD.md:19`), enough for tile 15 (`15*64 + 14 = 974`).

**Confidence.** High.

**Needs hardware measurement.** None for correctness. INFERRED: keep tile indices compile-time (template parameters) so the compiler can emit immediate `.ttinsn` words; with runtime indices the RISC-V must assemble each instruction word. Worth a quick cycle-count check if the rasterizer's inner loop uses runtime tile indices.

---

## V4. Keeping tiles in Dest across a long loop

**Answer.** Yes: everything between `tile_regs_acquire()` and `tile_regs_commit()` is one section, in which any number of SFPU/FPU/unpack operations (and `cb_wait_front`/`cb_pop_front`) may run; the matmul example runs a `Kt` loop there. The limits are: Dest contents do not survive a section (the packer-side release clears them), only 8 (16-bit) or 4 (FP32) tiles are addressable in half-sync, `commit` and `release` must always be called (even if nothing is packed), and the packer only gets the data after `commit`.

**Evidence.**
- Long accumulation loop inside one section: `tt_metal/programming_examples/matmul/matmul_single_core/kernels/compute/mm.cpp:57-75` ("Make sure registers can be used ... This also sets the registers to zero", `tile_regs_acquire()`, loop over `kt` with `cb_wait_front`, `matmul_tiles(..., 0)`, `cb_pop_front`, then `tile_regs_commit(); tile_regs_wait();`). Several SFPU ops chained in one section: `tt_metal/programming_examples/sfpu_eltwise_chain/kernels/compute/compute.cpp:27-69`.
- API: `tt_metal/hw/inc/api/compute/reg_api.h:50-108` (`tile_regs_acquire` = `llk_math_wait_for_dest_available`; `tile_regs_commit` = `llk_math_dest_section_done`; `tile_regs_wait` = `llk_packer_wait_for_math_done`; `tile_regs_release` = `llk_pack_dest_section_done`).
- Mechanism: acquire is `SEMWAIT(... MATH_PACK, STALL_ON_MAX)` (`tt_metal/tt-llk/tt_llk_blackhole/common/inc/cmath_common.h:300-303`). Commit posts the semaphore after the SFPU drains (`WAIT_SFPU`, `cmath_common.h:240-244`) and, in half-sync, flips the Dest half (`llk_math_common.h:107-115`). Pack wait: `llk_pack_common.h:25-33`. Release: `STALLWAIT(STALL_MATH, PACK)`, then `ZEROACC` of the whole Dest or the current half, then `SEMGET` to hand it back (`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack_common.h:64-95`, default `clear_dest = true`).
- "Even if a kernel does not pack any data, `tile_regs_commit` and `tile_regs_release` must still be called in sequence ... undefined behavior": `docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst:118-120`. The next kernel's start-up spins `while (semaphore_read(MATH_PACK) > 0)`, so a never-released section would hang it: `llk_math_common.h:153-156`.
- Packer reads of rows that Matrix/Vector Unit or unpackers wrote need the right ordering ("wait for the packer first" / "STALLWAIT"): `ISA:BlackholeA0/TensixTile/TensixCoprocessor/Dst.md:110-120` (instruction-scheduling section; the LLK already does this).
- Fresh Dest rows after `ZEROACC` are "undefined" and `SFPLOAD` of an undefined row is `UndefinedBehavior`; the SFPU must `SFPSTORE` its own initial values (zeros) before reading: `ISA:BlackholeA0/TensixTile/TensixCoprocessor/Dst.md:52`.
- Host-side watchdog for long kernels is configurable: `tt_metal/llrt/rtoptions.cpp:1400-1404` (`TT_METAL_OPERATION_TIMEOUT_SECONDS`).

**Confidence.** High for the protocol; medium for "no practical limit on loop length" (no hardware timer was found in the code read, the host timeout above is the only one).

**Needs hardware measurement.** Run a section with, e.g., 1e5 to 1e6 SFPU passes and then pack, to confirm there is no hang or watchdog trip and to get cycles per pass; confirm results with `dst_full_sync_en = true` (8 FP32 tiles) vs false (4).

---

## V5. How runtime scalars reach SFPU code

**Answer (A, runtime/compile arg to an SFPU parameter).** The scalar lives in L1 (runtime args, written by the host) or in code (compile-time args), is read into a RISC-V register by the math TRISC with `get_arg_val<>()`, is passed as a plain C++ argument into the SFPU function, and the RISC-V pushes `SFPLOADI` instructions that carry its two 16-bit halves in the instruction immediate. Cost per 32-bit scalar: 2 SFPU instructions (low half, high half; 1 instruction = 1 cycle each) once per call (hoist it out of the row loop), plus RISC-V work to assemble and `sw` each runtime instruction word (INFERRED, a few cycles); a value that fits BF16/FP16 (`sFloat16b`) or 16-bit int costs 1 `SFPLOADI`, and a compile-time BF16-representable factor can ride in the `Imm16` of `SFPMULI`/`SFPADDI` for free.
**Answer (B, replicated tile).** A tile whose elements all equal the scalar is written into a CB page in L1 by a data-movement RISC-V (or arrives by NoC), `copy_tile` unpacks it into a Dest tile (use unpack-to-dest for exact FP32), and the SFPU needs 1 `SFPLOAD` (1 cycle) to get the value into all 32 lanes, hoisted out of the loop; the price is the L1 fill (the example writes all 1024 elements with 1024 stores), a tile of L1, and an unpack whose cycle cost is not in the ISA docs. It is the right tool only when each lane or each row needs a different value.
**Answer (C, constant registers).** `LReg[8]` = 0.8373, `LReg[9]` = 0, `LReg[10]` = 1.0, `LReg[15]` = lane*2 are fixed; `LReg[11..14]` are programmable only via `SFPCONFIG`, which broadcasts 8 lanes of `LReg[0]` (so up to 8 distinct values across `lane & 7`, INFERRED use) to 32 lanes. Programming one costs 3 SFPU instructions (2x `SFPLOADI` into `LReg[0]` + 1 `SFPCONFIG`, latency up to 2 cycles) and then every use is a free register operand; sfpi reserves `LReg[11]` as -1.0, leaving `vConstFloatPrgm0..2` = `LReg[12..14]`.

**Evidence.**
- (A) Existing ops with scalar params: `tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h:29-44` (`add_unary_tile(idst, uint32_t param1)`: "fp32 value scalar encoded as uint32"), implemented at `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binop_with_unary.h:27-33` (`const sfpi::vFloat parameter = Converter::as_float(param);` then `val + parameter` per row). Others: `fill_tile(idst, float param0)` (`tt_metal/hw/inc/api/compute/eltwise_unary/fill.h:29-32`, body `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_fill.h:15-27`, `vFloat fill_val = value;` hoisted above the loop), `clamp_tile(idst, param0, param1)` (`eltwise_unary/clamp.h:30-39`), and the example `my_smoothstep_tiles(idx, edge0, edge1, inv_delta)` (`tt_metal/programming_examples/custom_sfpi_smoothstep/kernels/compute/tiles_smoothstep.cpp:30-33`, `.../ckernel_sfpu_smoothstep.h:13-22`).
- Runtime arg to compute kernel: `tt_metal/programming_examples/custom_sfpi_add/kernels/compute/tiles_add.cpp:44` (`get_arg_val<uint32_t>(0)`), host `SetRuntimeArgs(program, compute, core, {n_tiles})` at `tt_metal/programming_examples/custom_sfpi_add/custom_sfpi_add.cpp:155`. Runtime args are fixed per launch from the host; per-triangle values inside one launch have to come through a CB tile (B) or an L1 read by the math TRISC (INFERRED, plumbing not verified).
- Cost statement: "Scalar values may be converted to vectors ... In the general case this takes 2 instructions, to load the low and high halves of a 32-bit value ... `sFloat16b` ... ensuring a single load immediate": `docs/source/tt-metalium/tt_metal/apis/kernel_apis/sfpu/llk.rst:502-513`. `vFloat(float)` lowers through `__builtin_rvtt_sfpxloadi(..., -32)`: `runtime/sfpi/include/sfpi_funcs.h:531-532`.
- Runtime instruction words: `TT_SFPLOADI` pushes a RISC-V-computed word with a store (`TT_INSN` = `instrn_buffer[0] = ENCODING`), while `TTI_*` is a compile-time `.ttinsn`: `tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_ops.h:8-13, 663-666`. Explicit helper: `_sfpu_load_imm32_` = `TT_SFPLOADI(dest, 10, val & 0xFFFF)` + `TT_SFPLOADI(dest, 8, val >> 16)`: `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_load_config.h:18-22`. `SFPLOADI` modes (BF16, FP16, 16-bit int, upper/lower half): `ISA:WormholeB0/TensixTile/TensixCoprocessor/SFPLOADI.md:25-32`; `SFPMULI`/`SFPADDI` take a BF16 `Imm16`: `ISA:BlackholeA0/TensixTile/TensixCoprocessor/VectorUnit.md:13,17`.
- (B) Example: `tt_metal/programming_examples/sfpu_eltwise_chain/kernels/dataflow/reader.cpp:34-41` (reader RISC-V fills a "ones" tile with 1024 `uint16_t` stores and `cb_push_back`s it), `.../compute/compute.cpp:37-38` (`copy_tile(src, 0, 0); copy_tile(ones, 0, 1)`). Broadcast unpack variants: `unary_bcast<BroadcastType::...>(icb, tile, dst)` in `tt_metal/hw/inc/api/compute/bcast.h:34-105`. RISC-V store rate to L1: 1 store/cycle if stores coalesce into aligned 128-bit blocks, else 1 per 5 cycles: `ISA:BlackholeA0/TensixTile/BabyRISCV/README.md:85`.
- (B') Dest is also mapped into RISC-V address space at `0xFFBD_8000` (`ISA:BlackholeA0/TensixTile/TensixCoprocessor/Dst.md:122-125`; tt-metal itself only uses it for debug prints, `tt_metal/hw/inc/api/debug/dprint_tensix.h:164, 198`); one element per store/load, so not useful beyond a handful of scalars (INFERRED).
- (C) Fixed/programmable constants: `ISA:WormholeB0/TensixTile/TensixCoprocessor/LReg.md:11-15`; `SFPCONFIG` writes `LReg[11..14]` from `LReg[0][0:8]` broadcast: `ISA:WormholeB0/TensixTile/TensixCoprocessor/SFPCONFIG.md:5, 68-84`, latency "up to 2 cycles" in `ISA:BlackholeA0/TensixTile/TensixCoprocessor/VectorUnit.md:100`. sfpi names: `runtime/sfpi/include/sfpi.h:605-616` (`vConstFloatPrgm0` = LReg 12 ... `vConstTileId` = LReg 15), "These may be assigned to, but that is an expensive operation" at `sfpi.h:60-62`. The cost model from the tree itself: "One SFPCONFIG (+2 SFPLOADI) per call buys one instruction per row" for `vConstFloatPrgm1 = threshold`: `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_relu.h:60, 73`; clamp parks both bounds in `vConstIntPrgm0`: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_max_min.h:15`. Programmable constants persist until another op reprograms them; `LReg[11]` must stay -1.0 for sfpi code (`_restore_lconst_neg1_`, `ckernel_sfpu_load_config.h:45-49`).

**Confidence.** High for what each mechanism is and the instruction counts of A and C; medium for B (unpack cost) and for RISC-V-side overheads.

**Needs hardware measurement.** (1) Cycles for one runtime `vFloat(float)` load including the RISC-V instruction-word assembly, vs one `SFPCONFIG` constant program. (2) Cycles for `copy_tile` of a replicated tile into Dest (BF16 and unpack-to-dest FP32) and for `unary_bcast<SCALAR>`. (3) How a per-triangle scalar block in L1 is best handed to the math TRISC (CB tile vs direct L1 read) and whether the CB pointer plumbing works on the math thread. (4) Whether FP32 scalars survive bit-exactly through route B.

---

## V6. SFPU instruction issue rate and dependent-instruction stalls

**Answer.** All SFPU instructions listed (SFPADD/SFPMAD/SFPMUL/SFPMULI/SFPADDI/SFPLUT included) have IPC 1; `SFPLOAD`, `SFPSTORE`, `SFPLOADI`, `SFPMOV`, compares, bit ops and conversions have 1-cycle latency, and the MAD family (SFPMAD, SFPADD, SFPMUL, SFPMULI, SFPADDI, SFPLUT) has 2-cycle latency. The unit accepts only one instruction per cycle (the five sub-units load/simple/MAD/round/store can overlap only through `SFPLOADMACRO`, up to 5 per cycle); hardware auto-stalls one cycle if the very next instruction reads an SFPMAD result, so a dependent MAD chain runs at 1 per 2 cycles and independent work should be interleaved. A handful of instruction/hazard combinations are not tracked by the auto-stall and need a manual `SFPNOP`, and `SFPSWAP`/some `SFPSHFT2` modes make the next instruction wait.

**Evidence.**
- IPC and latency table: `ISA:BlackholeA0/TensixTile/TensixCoprocessor/VectorUnit.md:13-17` (SFPADDI/SFPADD/SFPMAD/SFPMUL/SFPMULI: IPC 1, latency 2 cycles), `:23-30` (SFPMOV, SFPSETSGN, SFPABS, SFPARECIP, SFPGT, SFPLE, SFPLZ, SFPSETCC: 1 cycle), `:77-81` (SFPLOADI/SFPLOAD/SFPSTORE: IPC 1, 1 cycle), `:22` (SFPSWAP: IPC <= 1, 2 cycles). One instruction per cycle: `:112` ("can only accept one instruction per cycle from the outside world"); up to five via `SFPLOADMACRO`: `ISA:BlackholeA0/TensixTile/TensixCoprocessor/SFPLOADMACRO.md:13`. Clock 1.35 GHz: `VectorUnit.md:6`; peak 64 FP32 op/cycle per SFPU = 0.0864 TFLOP/s: `SFPMAD.md:93`.
- Auto-stall: `SFPMAD.md:70` ("hardware will ensure that on the next cycle, the Vector Unit (SFPU) does not execute an instruction which reads from any location written to by the SFPMAD ... automatically stall the thread for one cycle"); indirect modes assume all LRegs read/written (`:70`); not applied inside `SFPLOADMACRO` sequences (`:73`). Untracked cases needing a manual one-cycle gap (e.g. `SFPAND/SFPOR` with `USE_VB`, `SFPIADD`, `SFPSHFT`, `SFPSETEXP/MAN/SGN` modifier bug, `SFPSWAP` non-swap modes, half-register `SFPLOADI`/`SFPLOAD` after a MAD): `SFPMAD.md:75-89`. `SFPSTOCHRND` float-to-int falsely stalls unless `VB = 9`: `SFPSTOCHRND_FloatInt.md:33`.
- `SFPSWAP`: next cycle accepts only `SFPNOP` (else 1 stall cycle; `SFPSWAP`+`SFPNOP` = 2 cycles instead of 3): `ISA:BlackholeA0/TensixTile/TensixCoprocessor/SFPSWAP.md:110`. Some `SFPSHFT2` modes the same: `SFPSHFT2.md:183`. `SFPLUTFP32` stalls conservatively: `SFPLUTFP32.md:149-153`.
- Dest hazard with the Matrix Unit: at least 3 unrelated Tensix instructions between an FPU write to a Dest region and an `SFPLOAD` of it: `ISA:BlackholeA0/TensixTile/TensixCoprocessor/SFPLOAD.md:203`. The LLK already inserts `STALLWAIT(STALL_SFPU, MATH)` before an SFPU section (`llk_math_eltwise_sfpu_common.h:20`).
- Front end: the SFPU is fed by RISC-V pushes (one `sw` per Tensix instruction); max dequeue 1 instruction per Tensix thread per cycle even though 4 `.ttinsn` can be fused per RISC-V cycle; MOP/REPLAY expanders let the backend see 1/cycle without a RISC-V push per instruction: `ISA:BlackholeA0/TensixTile/BabyRISCV/PushTensixInstruction.md:5, 19`, `ISA:WormholeB0/TensixTile/TensixCoprocessor/README.md:38`. The SFPI compiler uses `SFPREPLAY` for repeating sequences (works best for unrolled loops with < 32 instructions): `docs/source/tt-metalium/tt_metal/apis/kernel_apis/sfpu/llk.rst:909-930`. A fully tuned `where` reaches 3 cycles per input row of 32 values with `SFPLOADMACRO`: `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_where.h:50`.
- INFERRED arithmetic: a pass over one 32x32 tile with n SFPU instructions per vector costs about `32*n` cycles when not RISC-V bound (so add = load, load, add, store = 4 per vector = ~128 cycles = ~95 ns at 1.35 GHz).
- No documented hazard or latency for `SFPSTORE` immediately followed by `SFPLOAD` of the same Dest rows; the in-tree binary op does exactly that (`ckernel_sfpu_binary.h:140-142`).

**Confidence.** High for IPC/latency/stall rules; medium for end-to-end tile throughput (INFERRED).

**Needs hardware measurement.** Cycles per tile for the real kernel (RISC-V-bound or SFPU-bound?), the effect of `SFPREPLAY`, SFPSTORE then SFPLOAD to the same rows, and whether sfpi-generated `.ttinsn` streams actually fuse 4/cycle.

---

## V7. L1 caching on the baby RISC-Vs and cross-core/NoC visibility

**Answer.** The ISA has a per-core L0 data cache for L1 loads (4 lines x 16 B, write-through behavior, not coherent with NoC or other cores, flushed by `fence`/atomics), but tt-metal v0.81.1 firmware turns it OFF by default (CSR bit 3 set) and only enables it with `TT_METAL_ENABLE_L1_DATA_CACHE_RISCVS`. Regardless, the framework idiom is to call `invalidate_l1_cache()` (= `fence` on Blackhole) on every iteration of a poll loop on a `volatile` L1 pointer. The Blackhole semaphore wait is the 6-line loop quoted below.

**Evidence.**
- ISA cache description: `ISA:BlackholeA0/TensixTile/BabyRISCV/MemoryOrdering.md:57-61` ("4 lines of 16 bytes each ... The cache is _not_ coherent: if other clients write to L1, doing so will not invalidate the L0 of any baby RISCV ... polling loops are encouraged to either contain a `fence` instruction or use an atomic instruction"); fence flushes the cache: `:69-76`; CSR bit: `ISA:BlackholeA0/TensixTile/BabyRISCV/CSRs.md:58` (`DisLowCash` bit 3 "If set, the L0 data cache is disabled"), periodic flush bit `:72`. Reset state is all clear (cache enabled): `CSRs.md:80`.
- tt-metal default: `tt_metal/hw/inc/internal/firmware_common.h:322-334` ("The cache is default disabled ... can be enabled by setting `TT_METAL_ENABLE_L1_DATA_CACHE_RISCVS`", `#if defined(ENABLE_L1_DATA_CACHE) set_l1_data_cache<true>() #else set_l1_data_cache<false>()`); the CSR writes (`csrrc 0x7c0, 0x8` enables, `csrrs 0x7c0, 0x8` disables): `tt_metal/hw/inc/internal/tt-1xx/risc_common.h:150-175`. INFERRED: the same `configure_csr()` runs for the compute TRISCs.
- Same statement in the project's own guide: `tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md:67-80` ("The data cache is disabled by default but can be enabled globally through an env var `TT_METAL_ENABLE_L1_DATA_CACHE_RISCVS=<BR,NC,TR,ER>` ... can be enabled on kernel by kernel basis by calling `set_l1_data_cache<true>()`", must be paired with `set_l1_data_cache<false>()` before the kernel exits; `invalidate_l1_cache()` clears it; hardware periodic invalidation is off unless `TT_METAL_ENABLE_HW_CACHE_INVALIDATION`).
- `invalidate_l1_cache`: `tt_metal/hw/inc/internal/tt-1xx/cache.h:7-16`: "Blackhole L1 cache is a small write-through cache (4x16B L1 lines) ... Writing an address on one proc and reading it from another proc only requires the reader to invalidate. Need to invalidate any address written by noc ...", body `asm("fence")` under `ARCH_BLACKHOLE`.
- Semaphore wait on Blackhole (`tt_metal/hw/inc/api/dataflow/dataflow_api.h:2112-2125`):
  ```cpp
  void noc_semaphore_wait(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val) {
      ...
      do {
          invalidate_l1_cache();            // asm("fence") on Blackhole
      } while ((*sem_addr) != val);
      ...
  }
  ```
  `noc_semaphore_wait_min` (`>=`) is identical in shape: `dataflow_api.h:2141-2155`. CB reserve/wait loops use the same `invalidate_l1_cache()` per poll: `dataflow_api.h:410-425`.
- Other ordering traps on BH: NoC commands issued without flushes caused hangs/mismatches "because data and semaphore signals were getting updated faster than NoC has a chance to service the command": `tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md:95`; "Disabling the L1 cache can be helpful to identify missed cache invalidations": `:99`; a store followed by a Tensix instruction push may be processed out of order (use `STALLWAIT`, Auto TTSync, or read-back): `ISA:BlackholeA0/TensixTile/BabyRISCV/MemoryOrdering.md:102-105`. Relaxed memory ordering (loads may bypass stores to other addresses) is the default, can be turned off with `DISABLE_RELAXED_MEMORY_ORDERING`: `firmware_common.h:336-345`.

**Confidence.** High on the code and ISA text. Medium on "cache off on all five cores in our launch path" (read from firmware source only).

**Needs hardware measurement.** Whether a plain (non-`fence`) poll on a NoC-written flag ever returns a stale value with the cache enabled (only relevant if we enable it), the per-iteration cost of `fence` in a poll loop, and confirmation that the compute TRISCs also run with the cache disabled.

---

## V8. Can the SFPU gather per-lane addresses from L1?

**Answer.** No. The SFPU's only memory interface is `Dst` to `LReg` (`SFPLOAD`/`SFPSTORE`), addressed by one immediate plus RWC/offset registers that are the same for all 32 lanes; nothing in the SFPU instruction set takes an L1 address, and L1 is only reachable through the unpackers/packers, NoC and RISC-V. The only per-lane address-like freedoms are (a) a per-lane-group column parity override (`DEST_RD_COL_EXCHANGE` / `DEST_WR_COL_EXCHANGE`), (b) lane predication, and (c) `SFPMAD`'s indirect modes, which pick an LReg index per lane from `LReg[7]` (a register-file select, not memory).

**Evidence.**
- Instruction list and data-movement group: `ISA:BlackholeA0/TensixTile/TensixCoprocessor/VectorUnit.md:78-90` (load/store only `Dst` <-> `LReg`); no SFP instruction page mentions L1 (grep over `SFP*.md` finds only "L1" as an LReg name in `SFPSHFT2.md`).
- Address is uniform: `SFPLOAD.md:85-111` (`Addr = Imm10 + DEST_TARGET_REG_CFG_MATH_Offset + RWC.Dst + ...`; only `Lane` selects row/column offset), `SFPSTORE.md:86-106`.
- Column exchange bits: `ISA:WormholeB0/TensixTile/TensixCoprocessor/SFPCONFIG.md:124-125` (`LaneConfig[Lane & 7]` ... always load/store odd columns).
- Indirect register select: `ISA:BlackholeA0/TensixTile/TensixCoprocessor/SFPMAD.md:26-41` (`va = LReg[7].u32 & 15`, `vd = LReg[7].u32 & 15`).
- Moving data between lanes is limited to `SFPTRANSP`, `SFPSHFT2`: `VectorUnit.md:86-90`.
- Unpacker/packer/RISC-V are the L1 paths: `ISA:BlackholeA0/TensixTile/TensixCoprocessor/Dst.md:55-59`.

**Confidence.** High.

**Needs hardware measurement.** None.

---

## V9. NoC alignment for DRAM/L1 and interleaved page constraints (Blackhole)

**Answer.** From the installed HAL/NoC parameters: L1 reads and writes need 16 B alignment; DRAM reads need 64 B and DRAM writes 16 B, so `DRAM_ALIGNMENT` = 64 and `L1_ALIGNMENT` = 16 (PCIe read 64, write 16; NoC register access 4). The L1-side address must also be congruent to the NoC-side address modulo that alignment (watcher sanitizer rule). Interleaved buffers must have `size % page_size == 0`, and each page is padded up to the buffer type's alignment (64 B in DRAM, 16 B in L1), which is automatically satisfied for tile pages (2048 B bf16, 4096 B FP32).

**Evidence.**
- Constants: `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:374-394` (`NOC_L1_READ_ALIGNMENT_BYTES 16`, `NOC_L1_WRITE_ALIGNMENT_BYTES 16`, `NOC_PCIE_READ_ALIGNMENT_BYTES 64`, `NOC_PCIE_WRITE_ALIGNMENT_BYTES 16`, `NOC_DRAM_READ_ALIGNMENT_BYTES 64`, `NOC_DRAM_WRITE_ALIGNMENT_BYTES 16`, `NOC_REG_ALIGNMENT_BYTES 4`; `L1_ALIGNMENT`, `PCIE_ALIGNMENT`, `DRAM_ALIGNMENT` = max of read/write). Log2 forms: `tt_metal/hw/inc/internal/tt-1xx/blackhole/core_config.h:42-43` (`LOG_BASE_2_OF_DRAM_ALIGNMENT 6`, `LOG_BASE_2_OF_L1_ALIGNMENT 4`).
- HAL tables: `tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal.cpp:378-397` (`mem_alignments_`, `mem_read_alignments_`, `mem_write_alignments_` for L1/DRAM/HOST).
- Documentation table: `tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md:26-62` (Blackhole row: DRAM read 64 B / write 16 B, PCIe read 64 B / write 16 B, L1 read 16 B / write 16 B; Wormhole DRAM read 32 B).
- Congruence rule enforced by the watcher: `tt_metal/hw/inc/internal/debug/sanitize.h:598-625` (mask = alignment - 1, chosen by target kind and direction) and `:697` (`(worker_addr & alignment_mask) != (noc_addr & alignment_mask)` is an error).
- Interleaved buffers: `tt_metal/impl/buffers/buffer.cpp:286-291` ("For interleaved-buffers, buffer size should be divisible by the page size"), `:842-845` (`alignment()` from the allocator per buffer type, `aligned_page_size = align(page_size, alignment)`), allocator pads pages: `tech_reports/memory/allocator.md:45-49`, `tt_metal/impl/allocator/allocator.cpp:518-523, 764-772`. Semaphore slots are `L1_ALIGNMENT` apart: `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1690`. Tile sizes: 4096 B for FP32 and 2048 B for bf16 per 32x32 tile: `docs/source/tt-metalium/tt_metal/advanced_topics/tiles.rst:10`.
- ISA: the Blackhole `NoC/Alignment.md` is a stub ("similar, but not identical" to Wormhole). The Wormhole table (L1 to L1 / DRAM to L1: addresses must be congruent mod 16 or mod 32, `ISA:WormholeB0/NoC/Alignment.md:14-27`) is therefore only a hint; the HAL constants above are the authority for Blackhole.

**Confidence.** High for constants and the interleaved-buffer rules; medium for length constraints (none found in the installed code; only address rules).

**Needs hardware measurement.** None required for correct use. Optional: behavior of a DRAM read whose length is not a multiple of 64 from a 64-aligned address (not covered by the sources read).

---

## V10. SFPU math vs IEEE FP32

**Answer.** `SFPMAD` (and `SFPADD`/`SFPMUL`, which are `SFPMAD` aliases) computes `a*b+c` with one rounding step, round-to-nearest-even, but only "partially fused": the product keeps a few extra bits (3 low, 1 high) and is not exact, so results differ from both an IEEE fused FMA and two separate roundings (and an overflow/underflow of `a*b` alone is honored). Denormal inputs are read as zero and denormal results are flushed to signed zero; NaN outputs are the canonical `0x7fc00000`. Compares on Blackhole use a sign-magnitude total order (`-NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN`, so `-0 != +0` for equality, NaN compares are not IEEE-false), and float-to-int conversion is not a rounding-mode-free truncation: `SFPSTOCHRND` rounds to nearest with ties away from zero (default sfpi mode is STOCHASTIC!) or toward zero (with a bug), clamped to +-127/+-32767/0..255/0..65535; there is no direct float to 32-bit int instruction on Blackhole, tt-metal builds one from exponent/mantissa extraction and a shift, which truncates.

**Evidence.**
- SFPMAD: `ISA:BlackholeA0/TensixTile/TensixCoprocessor/SFPMAD.md:56-66` ("Denormal inputs are treated as if they were zero", "the multiply and the add are _partially_ fused, but not _completely_ fused ... A single rounding step is performed, with the rounding mode always round to nearest with ties to even", "If the output (after rounding) is denormal, it'll be flushed to sign-preserved zero", canonical NaN `0x7fc00000`). Bit-exact model and the list of differences from IEEE (extra 4 product bits, product overflow/underflow handled on its own): `ISA:Miscellaneous/FMA/README.md:15-27` (function `fma_model_bh` in `fma.c`). `SFPADD`/`SFPMUL` "Identical to SFPMAD": `ISA:BlackholeA0/TensixTile/TensixCoprocessor/SFPADD.md:3`, `SFPMUL.md:3`.
- sfpi expressions `a*b + c` are fused into a `SFPMAD` by the compiler ("MAD generation (from MUL/ADD)"): `docs/source/tt-metalium/tt_metal/apis/kernel_apis/sfpu/llk.rst:898`. So bit-exact CPU references must use the `fma_model_bh` semantics.
- Float comparisons on Blackhole: `docs/source/tt-metalium/tt_metal/apis/kernel_apis/sfpu/llk.rst:467-482` (equality compares bit patterns; "On Blackhole and Quasar, a sign-magnitude comparison is used ... +NaN > +Inf > ... > +0.0 > -0.0 > ... > -NaN"; "any comparison involving a NaN is false is not supported"); `ISA:BlackholeA0/TensixTile/TensixCoprocessor/SFPGT.md:3`, `SFPLE.md:3`. `SFPSETCC` against zero compares raw bits as a signed integer, so "software should ensure that ... FP32 have had negative zero flushed to positive zero": `ISA:WormholeB0/TensixTile/TensixCoprocessor/SFPSETCC.md:3, 30-36`. General list (-0 vs +0, NaN compares, ties round to larger magnitude): `llk.rst:136-149`.
- FP32 bit patterns: SFPU treats FP32 denormals as 0 and never emits denormals; its NaN has the lowest mantissa bit set: `ISA:WormholeB0/TensixTile/TensixCoprocessor/FloatBitPatterns.md:9-25`.
- Float to int: `SFPSTOCHRND` (FP32 to bounded sign-magnitude int): `ISA:BlackholeA0/TensixTile/TensixCoprocessor/SFPSTOCHRND_FloatInt.md:3-18, 54-86` (nearest = `PRNGBits = 0x400000`, i.e. ties away from zero; zero mode `0x7fffff`; hardware bugs: `|x| < 0.5` always rounds to zero even in stochastic mode, three magnitudes round away from zero in round-to-zero mode, `:18, 23`). sfpi `convert<>` default is `RoundMode::NearestStochastic` and `RoundMode::Nearest` means ties away on WH/BH: `runtime/sfpi/include/sfpi_lib.h:883-896, 939-990`, `runtime/sfpi/include/sfpi_constants.h:244`; conversion support matrix (no vFloat to vInt/vSMag on Blackhole): `llk.rst:402-440`. tt-metal's float to int32 helper uses `exexp`, `exman`, `shft` (truncating shift of the mantissa, positive inputs only): `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_conversions.h:7-30`.
- Int to float (`SFPCAST`): exact for `|x| <= 2^24`, otherwise RNE (mode 0) or stochastic (mode 1): `ISA:BlackholeA0/TensixTile/TensixCoprocessor/SFPCAST_IntFloat.md:3, 39-45`.
- Storing to a 16-bit Dest truncates: `SFPSTORE` FP32 to BF16 "Flush denormals to signed zero, then truncate toward zero", to FP16 truncates and saturates: `ISA:BlackholeA0/TensixTile/TensixCoprocessor/SFPSTORE.md:157-184`; FP32 Dest only flushes denormals (`:186-193`). tt-metal compensates where it matters: "bf16 dest stores truncate fp32->bf16 by default" (`ckernel_sfpu_binop_with_unary.h:47-51`, `float32_to_bf16_rne`).

**Confidence.** High for rounding/denormal/compare semantics (ISA + bit-exact model); medium for conversions (several documented hardware bugs).

**Needs hardware measurement.** Bit-exact check of `a*b+c` against `fma_model_bh` on edge inputs (including near-overflow products); confirm `-0`/NaN compare behavior of the sfpi `v_if` forms we plan to use; check float-to-int rounding of the path we choose for pixel coordinates (floor vs round vs truncate) on negative and large values.

---

## X1. Idiom for a compute kernel applying a custom SFPI function to Dest tiles (v0.81.1)

**Answer.** Write the SFPI function in namespace `ckernel::sfpu` in a header (loop of 8 iterations per face, `sfpi::dst_reg`), include it only under `#ifdef TRISC_MATH`, and call it from the kernel through `MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, fn, dst_idx, VectorMode::RC, args...))` (or `SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(..., in0, in1, out, VectorMode::RC, ...)`). Wrap it in a tiny inline function, between `tile_regs_acquire()` / `copy_tile()` and `tile_regs_commit()`. There is no CMake step for the device kernel: the host program calls `CreateKernel(program, path, core, ComputeConfig{...})` and the JIT compiles the same source three times (unpack/math/pack) with `-DTRISC_UNPACK/MATH/PACK`, so `MATH(...)` code only exists in the math binary.

**Evidence.**
- Minimal include set and structure (custom_sfpi_add): `tt_metal/programming_examples/custom_sfpi_add/kernels/compute/tiles_add.cpp:5-14` (`<cstdint>`, `api/compute/common.h`, `api/compute/tile_move_copy.h`, `api/compute/eltwise_binary_sfpu.h`, `api/compute/eltwise_unary/eltwise_unary.h`, `api/compute/compute_kernel_api.h`, then `#ifdef TRISC_MATH #include "experimental/llk_sfpu/ckernel_sfpu_custom_add.h" #endif`); wrapper `:38-41`; main loop `:43-83` (`compute_kernel_hw_startup(cb_in0, cb_out0); copy_init(cb_in0);` then per iteration `cb_wait_front`, `tile_regs_acquire`, `copy_tile`, custom op, `tile_regs_commit`, `tile_regs_wait`, `cb_reserve_back`, `pack_tile`, `cb_push_back`, `cb_pop_front`, `tile_regs_release`). Unary version with float params: `tt_metal/programming_examples/custom_sfpi_smoothstep/kernels/compute/tiles_smoothstep.cpp:5-33, 35-75`.
- The SFPU function bodies (pure sfpi): `tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_custom_add.h:5-28` (`#include "sfpi.h"`, `namespace ckernel::sfpu`), `.../ckernel_sfpu_smoothstep.h:5-24`.
- Macros: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h:43-51` (`SFPU_UNARY_CALL`, `SFPU_UNARY_CALL_NO_TEMPLATE_ARGS`: bound-check `dst_index` then `_llk_math_eltwise_unary_sfpu_params_(::ckernel::sfpu::FN, DST_IDX, VECTOR_MODE, args...)`), `llk_math_eltwise_binary_sfpu_macros.h:53-62`. `MATH(...)` is `__VA_ARGS__` only when `TRISC_MATH`: `tt_metal/hw/inc/api/compute/common_globals.h:14-34`. The macro headers are pulled in by `api/compute/compute_kernel_api.h:19-27` under `TRISC_MATH`.
- Host side: `tt_metal/programming_examples/custom_sfpi_add/custom_sfpi_add.cpp:124-155` (`CreateKernel(... OVERRIDE_KERNEL_PREFIX "custom_sfpi_add/kernels/compute/tiles_add.cpp", core, ComputeConfig{.fp32_dest_acc_en = false})`, `SetRuntimeArgs`). The CMake for the example only builds the host executable (`tt_metal/programming_examples/custom_sfpi_add/CMakeLists.txt`) and defines the kernel root: `tt_metal/programming_examples/CMakeLists.txt:8` (`add_compile_definitions(OVERRIDE_KERNEL_PREFIX="tt_metal/programming_examples/")`).
- JIT: one source compiled per TRISC with a role define (`chlkc_unpack.cpp`, `chlkc_math.cpp`, `chlkc_pack.cpp`, each `#include "<kernel path>"`): `tt_metal/jit_build/genfiles.cpp:808-835`. Include directories for Blackhole compute kernels (`llk_api`, `llk_api/llk_sfpu`, `tt-llk/tt_llk_blackhole/common/inc`, `.../llk_lib`, firmware dirs): `tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal.cpp:137-160`; common `-I` list: `tt_metal/jit_build/build.cpp:518-530`. Extra include dirs for a header outside the tree: `ComputeConfig::compiler_include_paths`, `tt_metal/api/tt-metalium/kernel_types.hpp:135` (INFERRED: a header next to the kernel `.cpp` also resolves with normal quote-include rules). `sfpi.h` comes from the SFPI toolchain at `runtime/sfpi/include/`.
- Ready-made patterns for ops with init state (programmable constants, LaneConfig) use `SFPU_UNARY_INIT*` macros: `llk_math_eltwise_unary_sfpu_macros.h:68-83`.
- For FP32 inputs use `ComputeConfig{.fp32_dest_acc_en = true, .unpack_to_dest_mode = {...UnpackToDestFp32...}}` and `copy_init(cb)` per format (docs: `docs/source/tt-metalium/tt_metal/advanced_topics/fp32_accuracy.rst`; `base_types.hpp:37-50`).

**Confidence.** High.

**Needs hardware measurement.** Not for the idiom itself; first on-device run should confirm the custom op works with `fp32_dest_acc_en = true` + unpack-to-dest FP32 (the shipped examples only show BF16).

---

## X2. Host FP32 tilize/untilize utilities and face order

**Answer.** Host utilities are in `tt_metal/api/tt-metalium/tilize_utils.hpp`: `tilize_nfaces<T>(input, m, n)` and `untilize_nfaces<T>(input, m, n)` (instantiated for `float`, `uint32_t`, `bfloat16`, `uint16_t`), the general `convert_layout<T>(inp, shape, inL, outL, tile_shape, face_shape, transpose_within_face, transpose_of_faces)` with `TensorLayoutType::TILED_NFACES`, and `tilize_swizzled`/`untilize_swizzled` for the other layout. `TILED_NFACES` order: tiles in row-major order of tiles (tile row outer, tile column inner); within a tile 4 faces in the order face0 = rows 0-15 x cols 0-15 (TL), face1 = rows 0-15 x cols 16-31 (TR), face2 = rows 16-31 x cols 0-15 (BL), face3 = rows 16-31 x cols 16-31 (BR); each face is a contiguous row-major 16x16 block (1024 B for FP32). The optional `transpose_of_faces` flag changes the face order to TL, BL, TR, BR; `transpose_within_face` transposes each face.

**Evidence.**
- Declarations: `tt_metal/api/tt-metalium/tilize_utils.hpp:18-23` (enum: "TILED_NFACES: row-major of tiles, each tile is N (N = 1, 2, or 4) faces, each face is row-major, faces are swizzled"), `:58-78` (`convert_layout`), `:88-92` (`tilize_nfaces`, `untilize_nfaces`).
- Implementation and face order: `tt_metal/impl/data_format/tilize_utils.cpp:286-326` (`write_face` copies `face_H x face_W` row by row; loops `face_h_index` outer, `face_w_index` inner for `!transpose_face_order`, reversed for `transpose_face_order`; tile loop `row_tile` outer, `col_tile` inner), `:519-534` (`tilize_nfaces` = `convert_layout(LIN_ROW_MAJOR -> TILED_NFACES)`, `untilize_nfaces` the reverse), float instantiations `:577, 581`.
- Same description in prose with C and NumPy code, plus tile byte sizes (FP32 tile = 4096 B, bf16 = 2048 B, faces at +0x200 steps for bf16): `docs/source/tt-metalium/tt_metal/advanced_topics/tiles.rst:10, 26-114, 116-152`.
- On-device equivalents (not host): `ttnn::tilize`/`ttnn::untilize` per `tiles.rst:154`, compute-API tilize in `tt_metal/hw/inc/api/compute/tilize.h`.

**Confidence.** High.

**Needs hardware measurement.** One small round trip of an FP32 buffer through `tilize_nfaces<float>` -> device copy -> `untilize_nfaces<float>` to confirm nothing reorders FP32 bits (INFERRED OK, no FP32-specific path found).

---

## Discrepancies and gotchas found while reading

1. `docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst:309` says `dst_reg[0:3]` is the first face and `[4:7]` the second. The installed code (8 iterations per face, 32 sfpi vectors per tile, `+16` rows between faces) and the ISA say one face is 8 `dst_reg` steps (`dst_reg[0:7]`). Trust the code.
2. `tt_metal/hw/inc/api/compute/reg_api.h:21, 43` says Dest is "16 tiles"; that is the full-sync 16-bit figure, usable counts are 8/4 in half-sync (see V2). The comment at `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_datacopy.h:47` ("each half of DEST can take 16 tiles" for 32-bit) is stale; the code below it uses 4 tiles per bank for FP32 (`tiles_per_bank = clear_fp32 ? 4 : 8`).
3. `docs/.../fp32_accuracy.rst` suggests `fp32_dest_acc_en` + `copy_init` suffices for FP32 accuracy; copying an FP32 CB through SrcA still loses precision unless `UnpackToDestFp32` is set (`base_types.hpp:37-50`; `tt_metal/hw/inc/api/compute/bcast.h:46, 86`, "SrcB is only 19bits wide").
4. sfpi `convert<>`/`SFPSTOCHRND` default rounding is stochastic, not nearest (`sfpi_lib.h:939`); always pass `RoundMode::Nearest` (ties away) or `Zero` explicitly.
5. The ISA `NoC/Alignment.md` for Blackhole is a stub pointing at Wormhole; use the HAL constants (V9).
6. In FP32 half-sync mode the second bank's base offset is the 16-bit-row value 512 even though only 256 Dst32b rows per bank exist; INFERRED from `Adj32` in `ISA:BlackholeA0/TensixTile/TensixCoprocessor/Dst.md:32-40` that row 512+x aliases row 256+x, which is why the LLK can use one constant. Not an issue unless SFPU code builds its own absolute Dest addresses past the half.
