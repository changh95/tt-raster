# FINDINGS

Evidence for every [VERIFY] item and every decision, as the plan requires. Machine-identifying details (hostname, board and ASIC ids, PCIe bus ids, IPs) are deliberately left out because this repository is public.

## Environment (Phase 0.1, recorded 2026-11-10)

| Item | Value | Evidence |
| --- | --- | --- |
| Machine | Tenstorrent Galaxy, Blackhole, 32 chips in 4 trays of 8 | `tt-smi -s`: 32 devices, board type `tt-galaxy-bh` |
| Assigned tray | The plans call it "tray 0"; `tt-smi` numbers it tray 1. Device nodes `/dev/tenstorrent/0` to `7` | User decision, 2026-11-10; `tt-smi -glx_list_tray_to_device` |
| Tray to node map | Tray 1: nodes 0-7, tray 2: 8-15, tray 3: 24-31, tray 4: 16-23 | `tt-smi -glx_list_tray_to_device` (read-only) |
| Firmware bundle | 19.13.1.0 (CM 0.35.1.0, ETH 1.12.0) | `tt-smi -s` |
| Host driver stack | TT-KMD 2.10.0, tt-smi 4.1.2, tt-umd 0.9.3 | `tt-smi -s` |
| AI clock | 1350 MHz maximum (`asic_fmax`); reads 800 MHz while idle | `tt-smi -s` before and after stopping the previous workload |
| Tensix columns | 13 of 14 enabled on every chip, a different column per chip | `ENABLED_TENSIX_COL` masks in `tt-smi -s` |
| Worker grid (device API) | Logical grid 13x10; `compute_with_storage_grid_size()` 12x10 = 120 worker cores. Identical on all 8 chips of the tray despite different harvested columns | `tools/probe_device` on chip 0, chips 0-3 (4 processes) and chips 0-7 (one mesh) |
| Per-chip memory | DRAM: 8 channels x 4080 MiB. L1: 1536 KiB per worker core | `tools/probe_device` |
| Tray mesh shape | `MeshShape([2, 4])` for nodes 0-7 opened as one mesh | `tools/probe_device mesh`, `TT_VISIBLE_DEVICES=0,...,7` |
| Host path to chips | Every chip has its own PCIe device node, so all 8 are local (MMIO) chips of the host: "local chip ids/PCIe ids {0..7}, remote chip ids {}". Each link is PCIe Gen5 (32 GT/s). Exactly one chip per tray is wired x8 and the other seven x1: on this tray node 5 is x8 and nodes 0-4, 6, 7 are x1 (link width in sysfs; `tools/scaleout/kmd_triage/README.md` states the same rule) | UMD log on open, sysfs, `spikes/e_tray` |
| Stock examples | `metal_example_add_2_integers_in_riscv` and `metal_example_eltwise_sfpu` pass on one chip | run on node 0 with `TT_VISIBLE_DEVICES=0` |
| tt-metal | Tag `v0.81.1`, commit `d80d9482a418cf04ddcbeecf6adb67010d3091e6`, shallow clone with submodules | `git describe`, `git rev-parse HEAD` |
| tt-metal build | Release, clang-20, programming examples on, Python bindings off; 5 min 54 s on a 64-core host | build log |
| Mesa, CTS and tools | Not installed yet. LLVM 14, 17 and 20 dev packages are present; meson, glslang, vulkaninfo and glxinfo are not | `dpkg -l`, `command -v` |

## [VERIFY] items and their status

| ID | Status | Notes |
| --- | --- | --- |
| V1 lane to (row, col) map | **Measured** (FP32 Dest mode) | See "Lane map" below. Evidence: `spikes/a_lane_map` on one chip, all 1024 positions consistent. The ISA documentation (`SFPLOAD.md`) states the same map independently: lane `L` of the vector at address `Addr` goes to Dest row `(Addr & ~3) + L/8` and column `2*(L & 7) + ((Addr >> 1) & 1)`; `dst_reg[i]` encodes `Addr = 2*i`; `vConstTileId` holds `2*lane`. See `docs/READING_NOTES.md` |
| V3 one SFPU function, several Dest tiles | **Confirmed** | `spike_a_checker_face(idx_x, idx_y, idx_out)` reads Dest tiles 0 and 1 and writes tile 2 in one pass, addressing tile `t` as `dst_reg[t * 32 + i]`; a unary call on tile 3 in the same acquire. Dest tile indices 0 to 3 work in FP32 mode with `fp32_dest_acc_en` |
| V4 tiles resident across a long loop | **Confirmed** | One `tile_regs_acquire` to `tile_regs_commit` held 4 FP32 tiles (dx, dy, accumulator, coefficient) across a 68 ms loop of 131,072 triangle-tile updates with a correct result (Spike B calibration run, K = 2048). Dest works as on-chip tile memory for as long as the kernel holds it. Rules from the reading notes: Dest contents do not survive release (it is zeroed), commit and release are mandatory even if nothing is packed, and an `SFPLOAD` of a Dest row never written is undefined, so the kernel must store initial values (zeros) before the first load, as Spikes B and C do |
| V10 SFPU math vs IEEE FP32 | Partly measured | Exact for the checkerboard and gradient (multiples of 0.25 and integer parity): 0 of 1024 differ. `x*0.1f + y*0.3f` on the grid: 927/1024 bit-equal to the two-rounding CPU result, 828/1024 to a fused one, 955/1024 to either, every difference at most 1 ulp. So SFPU multiply-add is neither of those two CPU evaluations; coverage must stay exact-integer and interpolated values need a tolerance. Integer exactness is full 24-bit: in Spike B, `a*dx + b*dy + c >= 0` and `acc += d` matched an exact int64 reference at all 1024 pixels for every variant, with `a*dx` up to 16.2 million and partial sums up to about 12 million, so integer-valued edge functions are exact on the SFPU while every intermediate stays below 2^24. Source note: SFPI conversions default to stochastic rounding (`RoundMode::NearestStochastic`), so always pass `RoundMode::Nearest`. From the ISA documentation (source only, not yet measured; `docs/READING_NOTES.md`): `SFPMAD` rounds once to nearest-even but is only partially fused (the product keeps a few extra bits), so a bit-exact CPU model must use the ISA's `fma_model_bh`; denormals are read as zero and flushed; float compares are sign-magnitude, so `-0.0` is not equal to `+0.0` and `e >= 0` is false for `-0.0` (keep edge-function constants as `+0.0`); there is no float to 32-bit int conversion on Blackhole, only 16-bit types, and `RoundMode::Nearest` rounds ties away from zero |
| V5 runtime scalars into the SFPU | **Measured**, all three paths work | Numbers in `docs/PERF.md` (Spike B). (A) The math thread reads floats from a circular-buffer tile in L1 and issues them as runtime immediates: about 9 cycles per scalar. The L1 address comes from `get_tile_address(cb, tile)`, which the unpack thread distributes to the math and pack threads through a mailbox, so the data can simply be a raw float array streamed by the reader kernel. (B) A coefficient tile with each scalar replicated over the lanes, unpacked into a Dest slot: about 4 cycles per scalar, but one extra Dest slot and 512 bytes of tile data per triangle instead of 16. (C) The three programmable constant registers: about 16 cycles per write, so the most expensive and limited to three scalars. Vector-major order with approach A is 3.5x slower, because the scalars are re-issued for each of the 32 vectors |
| V2 Dest capacity | **Measured** (FP32) | Source (`get_dest_max_tiles_rt` in `ckernel.h`): 1024 Dest rows, 64 rows per tile in 16-bit mode and 128 in FP32 mode, half-sync gets half. So 16-bit: 16 tiles full-sync, 8 half-sync; FP32 (`fp32_dest_acc_en`): 8 tiles full-sync (`dst_full_sync_en`), 4 half-sync (default). Device test (`spikes/c_dest`): FP32 half-sync holds 4 tiles intact, and a 5th also survived a single acquire/commit cycle because the other half is idle, but that region belongs to the packer's double buffer, so do not use it. FP32 full-sync holds 8 tiles intact; a 9th tile index silently overwrote tile 4 (1024 elements corrupted), so going past capacity corrupts data without any error |
| V6 SFPU issue rate | **Measured** | A multiply-add issues at 1 per cycle (1.03 cycles with 4 independent chains) and a dependent chain takes 2 cycles per multiply-add, so the latency is 2 cycles. The compiler refuses code whose live SFPU values do not fit the register file (8 chains plus 2 constants failed with "too few lregs"), so about 8 live vectors is the working limit. The masked update step (3 loads, 2 multiply-adds, compare, predicated add, store) costs 18.4 cycles per vector with the 32-vector loop unrolled and 22.1 with a runtime vector count (17% slower): that is more than its instruction count, which suggests the dependent 2-cycle latencies are exposed; interleaving two or more vectors per loop step is the first thing to try in Phase 2.5. One wall-clock tick is one AI clock cycle (calibrated at 1349 MHz) |
| Math-thread timing method | Works | `ckernel::read_wall_clock()` before and after the loop with `ckernel::tensix_sync()` before the second read; the cycle count rides out of the kernel inside a Dest tile |
| Runtime vector count costs | Noted | A loop bound that is not a compile-time constant is not unrolled and costs about 17% more per vector. Specialize the common cases (full tile, one face) at compile time |
| V7 L1 caching on the RISC-V cores | **Source-verified**, idiom works on hardware | The ISA describes a non-coherent 4x16 B L0 data cache, but tt-metal v0.81.1 firmware disables it by default (`tt_metal/hw/inc/internal/firmware_common.h`, opt-in `TT_METAL_ENABLE_L1_DATA_CACHE_RISCVS`). Regardless, polling a word written by the NoC or another core must call `invalidate_l1_cache()` (a `fence`, `tt_metal/hw/inc/internal/tt-1xx/cache.h`) each iteration, which is what `noc_semaphore_wait` does. Spike D used exactly that handshake with 119 receivers without a stale read |
| V8 SFPU gather from L1 | **Confirmed no** (ISA) | The SFPU's only memory path is Dest to LReg with one address shared by all lanes; L1 is reachable only through unpacker, packer, NoC and the RISC-V. Per-lane freedoms are column parity, predication and `SFPMAD`'s register select, none of which is a gather. So texturing cannot be done in one SFPU gather |
| V9 NoC alignment | **Source-verified** | L1 reads and writes 16 B; DRAM reads 64 B and DRAM writes 16 B (`NOC_*_ALIGNMENT_BYTES` in `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h`, so `DRAM_ALIGNMENT` is 64 and `L1_ALIGNMENT` 16). The L1 address and the NoC address must be congruent modulo that alignment (watcher rule). Interleaved buffer size must be a multiple of the page size and pages are padded to the alignment; tile pages (4096 B in FP32) satisfy it. Triangle records stored in DRAM should be padded to 64 B multiples |
| V6 follow-up | Hypotheses for Phase 2.5 | The ISA says the SFPU accepts one instruction per cycle except through `SFPLOADMACRO` (up to 5 per cycle), `SFPREPLAY` can replay unrolled sequences of under 32 instructions, and an in-tree tuned op reaches 3 cycles per input row. Our 18.4 cycles per vector is far above the instruction count, so interleaving vectors, `SFPREPLAY` and `SFPLOADMACRO` are the optimizations to try |
| Separate processes holding separate chips | **Confirmed** | Four processes, each with `TT_VISIBLE_DEVICES` set to a different node (0 to 3), opened and held their chip for 15 s at the same time, started simultaneously; all exited 0 and `lsof` showed one holder per node. The design reason: UMD guards "multiple users initializing the same chip" with a per-chip `CHIP_IN_USE` mutex (`tt_metal/third_party/umd/tt-umd/api/tt-umd/utils/lock_manager.hpp`). Same-chip sharing was deliberately not tested (plan rule: never two device programs on one chip). Consequence for the full-coverage plan: a device server is only needed to share one chip, not to share the tray |
| Device-node numbering vs tt-smi index | Noted | Nodes 0 to 7 are the same chips in both views, but trays 3 and 4 are swapped relative to node order (nodes 16-23 are tray 4, nodes 24-31 are tray 3), so a tt-smi list index is not always the node id. Always select chips by device node id |
| Chip selection | **Confirmed** | `TT_VISIBLE_DEVICES` takes `/dev/tenstorrent/<id>` numbers (`tech_reports/Programming_Mesh_of_Devices`). With `0` UMD opens only node 0; with `0,...,7` it opens exactly those eight. Inside the process the visible chips are renumbered from 0, so code always opens device 0 of its visible set. A "board has N chips, expected 32" UMD warning is expected and harmless when fewer chips are visible |

## Lane map (V1, measured by Spike A)

Setup: FP32 tiles in Dest (`fp32_dest_acc_en`, `UnpackToDestFp32`), an SFPU function called per face through the `VectorMode::RC` wrapper, each face being 8 SFPU vectors of 32 lanes, the host tilizing with `tilize_nfaces<float>`. Each lane wrote `(face*8 + vector)*64 + vConstTileId`, and the untilized output shows which one landed at each tile position.

- Faces are 16x16 and ordered row-major inside the tile: top-left, top-right, bottom-left, bottom-right. The SFPU's own face counter equals `(row/16)*2 + (col/16)` at all 1024 positions.
- Inside a face, for local `rr = row%16`, `cc = col%16`:
  - vector index `v = 2*(rr/4) + (cc & 1)`
  - lane `l = 8*(rr % 4) + cc/2`
  - `vConstTileId = 2*l = 16*(rr % 4) + (cc & 0xE)`; it takes 32 distinct even values, 0 to 62
- Inverse, which lets an SFPU kernel derive pixel coordinates with no coordinate tiles: `col = 16*(face&1) + (vConstTileId & 0xE) + (v & 1)`, `row = 16*(face>>1) + 4*(v>>1) + (vConstTileId >> 4)`.
- Consequence: the "derive coordinates from lane ids" fallback in the plan's risk table is available. Spike B generates the dx and dy tiles once per tile from lane ids and every result was exact. Deriving the coordinates again inside the triangle loop, instead of keeping the two tiles, was not measured; it would free two Dest slots at the cost of a few integer SFPU instructions per vector per triangle.
- Not measured: the same map in 16-bit Dest mode (an in-tree kernel comment suggests it differs for BF16). Phase 2 uses FP32.

## Multicast (Spike D, V-multicast)

Setup: one sender core multicasts 4 KB to a rectangle of worker cores, semaphore handshake (ready, flag, ack), every receiver poisons its buffer first and then verifies the pattern and writes a record to DRAM. `spikes/d_multicast`, one chip. Eleven rectangles including single core, row, column, interior block, right and bottom edges, all rows below the sender, sender at the far corner, sender mid-grid, and the full grid with the sender inside (loopback variant of the multicast calls). Every receiver of every rectangle verified (up to 119 of 119).

- NoC coordinates: the 12x10 worker grid sits at NoC x = 1..7 and 10..14, NoC y = 2..11. Worker columns are not contiguous in NoC space: x = 8 and 9 hold non-Tensix tiles. A rectangle spanning the gap (the full-grid multicast) still worked, so the gap needs no special handling for multicast. Always convert logical to NoC coordinates with the device API (`worker_core_from_logical_core`), never by arithmetic.
- Latency and bandwidth are in `docs/PERF.md`.
- Not tested: multicast to a rectangle that contains harvested columns on a different chip. Every chip presents the same logical grid; the logical to NoC map may differ per chip, so tray-wide code must use each device's own map.

## Single chip and full tray (Spike E)

- One binary, two modes (`spike_e unit` and `spike_e mesh`, chips chosen by `TT_VISIBLE_DEVICES`): the trivial program ran and verified on 1 of 1 device in unit mode and on 8 of 8 devices in mesh mode, on the 2x4 mesh of nodes 0-7.
- Mesh coordinates are topology-based, not node-ordered: mesh row 0 holds nodes 4, 5, 6, 7 and row 1 holds nodes 0, 1, 2, 3 (coordinates `[0,0]` to `[0,3]` and `[1,0]` to `[1,3]`). Assign screen regions by mesh coordinate, and look up the device with `mesh->get_device(coord)`.
- Host to chip bandwidth is link-limited and uneven: about 3.2 GB/s each way on an x1 chip (82% of the 3.94 GB/s Gen5 x1 limit), 19.8 GB/s write and 13 to 16 GB/s read on the x8 chip (numbers in `docs/PERF.md`). Reading all eight chips back at once reached only 11.3 GB/s in total with non-blocking per-shard reads, against about 37 GB/s if every link ran in parallel, so the read-back of multi-chip frames must be designed and measured, not assumed.
- Consequence for frame read-back: a 1920x1088 RGBA8 frame is 8.4 MB, which is 2.6 ms through one x1 link, so splitting the frame over eight chips and reading the pieces back concurrently is worth measuring in Phase 2.6; the x8 chip can carry a larger share of the screen.

## Tray topology and chip-to-chip (Spike E)

- Fabric control plane for nodes 0-7: one mesh of shape 2x4 whose X and Y axes are both a line (no wraparound inside the tray). Every chip has 12 of 14 Ethernet cores enabled (`ENABLED_ETH` in `tt-smi -s`); the per-pair link counts were not enumerated.
- Node, fabric chip id and mesh coordinate: fabric chip id is the row-major index in the 2x4 mesh, so fabric chips 0-3 are mesh row 0 = nodes 4-7, and fabric chips 4-7 are row 1 = nodes 0-3. Node `k` sits at tray position (`asic_location`) `k + 1` of tray 1.
- Fabric bring-up (`FABRIC_2D`, 8 devices) takes about 3.2 s, and it ran and shut down cleanly: afterwards all 32 chips still answered `tt-smi`, DRAM status was fine, and the Ethernet status values had the same distribution as before.
- Chip-to-chip bandwidth measured with the in-tree `bench_unicast` (one sender core to one receiver core, 4 KiB pages, through the fabric; built by reconfiguring the tt-metal build with `-DTT_METAL_BUILD_TESTS=ON` and building only that target): 10.5 GB/s between horizontal neighbours and also across three hops along a row, 14.8 GB/s to the vertical neighbour and also to the diagonal chip. So extra hops cost no bandwidth, and the vertical direction is faster than the horizontal one for a single stream. These are single-stream numbers; several cores or several links in parallel were not measured. Small transfers are latency-bound: 4 KiB takes 3.5 us and 64 KiB 9.2 us.
- Consequence: chip-to-chip traffic inside the tray is about 3x the x1 host link per stream and comparable to the x8 host link, so passing intermediate results between chips costs less than a trip through the host.

## Dest layout for Phase 2 (proposal, to be confirmed at the Phase 0 STOP)

Budget: FP32 half-sync gives 4 tiles and keeps pack overlapped with the next tile's math; FP32 full-sync gives 8 tiles but serializes pack and math. Phase 2 stores color as one packed RGBA8 word per pixel, so color needs one tile instead of four FP32 planes. Per-triangle plane equations for depth and color are evaluated in SFPU registers, not stored.

| Slot | Half-sync layout (4 tiles) | Full-sync layout (8 tiles) |
| --- | --- | --- |
| 0 | depth (FP32) | depth |
| 1 | color, RGBA8 packed as a 32-bit integer | color RGBA8 |
| 2 | dx (from lane ids, generated once per tile) | dx |
| 3 | dy | dy |
| 4 to 7 | not available | coefficient tile (scalar path B), triangle id or alpha plane, two spare |

Scalars use path A (L1 runtime immediates) in both layouts, so no slot is reserved for coefficients in the half-sync layout. Recommended scalar path for Phase 2: A, triangle-major, with compile-time-unrolled bodies for a full tile and a runtime-bounded loop for partial coverage. Path B has the lowest delivery cost (about 16 versus 37 cycles per triangle) but takes a Dest slot and 32x more data per triangle, and A is within 7% of the no-scalar baseline for a full-tile triangle; revisit B if the profile shows scalar delivery above roughly 10%. Do not use C (programmable constants) or vector-major order with A. Deriving dx and dy from `vConstTileId` inside the triangle loop instead of keeping them as tiles would free slots 2 and 3, but costs a few integer SFPU instructions per vector per triangle, against two loads: keep the tiles unless slots are needed.

## Build notes

- Kernels can be compiled from absolute paths outside the tt-metal tree (`CreateKernel` with the absolute path of `spikes/.../kernels/*.cpp`), and a header next to the kernel is found because the kernel's own directory is on the JIT include path.
- Kernel JIT flags include `-ffast-math -fno-finite-math-only -fsigned-zeros -fno-associative-math`.
- Run device programs through `scripts/run.sh <nodes> <binary>`: it sets `TT_VISIBLE_DEVICES`, `TT_METAL_RUNTIME_ROOT`, a timeout, and a scratch working directory (tt-metal writes `generated/` into the current directory), and accepts only nodes 0 to 7.

- tt-raster builds out of tree against the tt-metal build directory: `TT_METAL_HOME=... scripts/build.sh`. The build must use the same toolchain file tt-metal was built with (read from its `CMakeCache.txt`).
- Use the package config in `build_Release/lib/cmake/tt-metalium` (set `tt-metalium_DIR`). The `tt-metalium-config.cmake` at the build directory root is not self-contained. The exported target is `TT::tt_stl`, not `TT::STL`, and `TT::Metalium` already links it.
- Running from outside the tt-metal tree needs `TT_METAL_RUNTIME_ROOT` pointing at the checkout, or the process aborts before opening any device.

## Decisions

- 2026-11-10: tt-metal is a fresh `v0.81.1` checkout next to this repository. The other tt-metal checkouts on the machine belong to other projects and are not touched.
- 2026-11-10: Track G (softpipe fork) is superseded by the llvmpipe and lavapipe fork in `docs/FULL_COVERAGE_PLAN.md`. Confirmation pending.
