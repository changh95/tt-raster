# PERF

One row per measurement: what was measured, configuration, result with units, commit.

| What | Configuration | Result | Commit |
| --- | --- | --- | --- |
| tt-metal v0.81.1 Release build, programming examples, no Python bindings | 64-core host, clang-20 | 5 min 54 s wall | n/a (tt-metal checkout) |
| Device open time, `create_unit_mesh` | 1 chip, process starts with an idle chip | 0.12 to 0.13 s (4 processes opening 4 chips at once: 0.117, 0.123, 0.130, 0.130 s) | tools/probe_device.cpp |
| Device open time, 8-chip mesh | `MeshDevice::create`, shape 2x4, nodes 0-7 | 0.28 s | tools/probe_device.cpp |
| `metal_example_add_2_integers_in_riscv` wall time | chip 0, cold JIT cache (0/7 hits) | 3.4 s | n/a (tt-metal example) |
| `metal_example_eltwise_sfpu` wall time | chip 0, warm JIT cache (6/11 hits) | 1.2 s | n/a (tt-metal example) |
| AI clock under test | `TT_VISIBLE_DEVICES=0..7` mesh open | 1337 MHz settled (requested 1350), 800 MHz idle | n/a |
| Math-thread wall-clock tick rate | `read_wall_clock()` against host time, two runs of different length (launch overhead cancels) | 1348.8 and 1349.1 MHz in two runs, so one tick is one AI clock cycle (1350 MHz) | spikes/b_scalars |
| First-run time of a new kernel variant | JIT compile plus run, cold cache, one compute+two data-movement kernels | about 0.6 s | spikes/b_scalars |

### Spike B: runtime scalars into the SFPU

Workload: one 32x32 FP32 tile in Dest, per triangle `if (a*dx + b*dy + c >= 0) acc += d` over the touched SFPU vectors, 64 triangles per batch, K = 64 batches (4096 triangle-tile updates), one chip, cycles read on the math thread after `tensix_sync()`. Every variant matches an exact int64 reference bit for bit, including triangles whose `a*dx` reaches 16.2 million (just under 2^24).

Cycles per triangle-tile update, all 32 vectors touched, loop bound known at compile time or runtime as noted:

| Variant | Scalars come from | Order | Triangles per SFPU call | Cycles per triangle | Extra over baseline |
| --- | --- | --- | --- | --- | --- |
| const (baseline) | baked into the code | triangle-major | 64 | 588.3 (unrolled, first run) / 721.3 (runtime vector count) | n/a |
| A | L1 floats read by the math thread, issued as runtime immediates | triangle-major | 64 | 628.3 / 758.3 | +40.0 / +37.0 |
| A | same | triangle-major | 8 | 630.6 | +2.3 versus 64 per call, about 18 cycles per SFPU call |
| B | coefficient tile (each scalar replicated over the lanes) unpacked into Dest | triangle-major | 8 | 607.1 / 737.1 | +18.8 / +15.8 |
| C | programmable SFPU constant registers for a, b, c; d as immediate | triangle-major | 64 | 647.3 / 777.3 | +59.0 / +56.0 |
| A | L1 immediates | vector-major | 64 | 2059.8 / 2061.8 | 3.5x slower |
| B | coefficient tile | vector-major | 8 | 701.1 / 717.4 | about the same as the baseline |

Vector-count sweep, runtime loop bound, cycles per triangle-tile update, with a least-squares fit `cycles = fixed + per_vector * vectors`:

| Variant | 1 vector | 2 | 4 | 8 | 16 | 32 | Fixed per triangle | Per vector |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| const | 31.3 | 61.3 | 105.3 | 193.3 | 369.3 | 721.3 | 14.8 | 22.1 |
| A | 67.3 | 98.3 | 142.3 | 230.3 | 406.3 | 758.3 | 51.5 | 22.1 |
| B | 47.1 | 77.1 | 121.1 | 209.1 | 385.1 | 737.1 | 30.7 | 22.1 |
| C | 86.3 | 117.3 | 161.3 | 249.3 | 425.3 | 777.3 | 70.5 | 22.1 |

Derived scalar-delivery cost per triangle (4 scalars): A about 37 cycles (9 per scalar), B about 16 (4 per scalar, including the amortized unpack of one coefficient tile per 8 triangles), C about 56 (about 16 per programmable-constant write). Per-vector cost of the update step: 22.1 cycles with a runtime vector count, 18.4 cycles when the loop over all 32 vectors is unrolled at compile time.

### SFPU multiply-add issue rate (V6)

`x = x*a + b` on one vector, 8 steps per loop iteration, cycles per multiply-add from the slope between 1000 and 3000 iterations (loop and start-up overhead cancel). One chip, `spikes/b_scalars/mad_bench`.

| Independent chains | Cycles per multiply-add | Reading |
| --- | --- | --- |
| 1 (fully dependent) | 2.00 | multiply-add latency is 2 cycles |
| 2 | 1.06 | latency mostly hidden |
| 4 | 1.03 | issue throughput is one multiply-add per cycle |
| 8 | does not compile | "too few lregs to hold live values": 8 chains plus `a` and `b` exceed the SFPU register file |

### NoC multicast (Spike D)

One chip, sender core multicasts a 4 KB buffer. Latency = multicast of data plus flag, until every receiver's ack has arrived at the sender (round trip, includes ack serialization). Injection bandwidth = 256 back-to-back 4 KB multicasts completed at the sender (`noc_async_write_barrier`), 4 KB x 256 / time. Delivered bandwidth = injection x destinations. AI clock 1.35 GHz. `spikes/d_multicast`.

| Rectangle (sender outside unless noted) | Receivers | Latency (cycles) | Latency (ns) | Injection (GB/s) | Delivered (GB/s) |
| --- | --- | --- | --- | --- | --- |
| single receiver | 1 | 306 | 227 | 64.0 | 64.0 |
| top row | 11 | 927 | 687 | 18.6 | 204 |
| left column | 9 | 697 | 516 | 25.5 | 230 |
| interior block 6x4 | 24 | 1172 | 868 | 14.6 | 350 |
| right edge column | 10 | 1364 | 1010 | 12.5 | 125 |
| bottom row | 12 | 1357 | 1005 | 12.5 | 150 |
| all rows below the sender | 108 | 2298 | 1702 | 12.5 | 1350 |
| sender at far corner | 99 | 2397 | 1776 | 10.9 | 1078 |
| sender mid-grid, rectangle outside | 96 | 2276 | 1686 | 11.7 | 1120 |
| full grid, sender inside (loopback) | 119 | 2429 | 1799 | 12.5 | 1499 |
| full grid, sender mid-grid (loopback) | 119 | 2606 | 1930 | 10.9 | 1310 |

Reading: the sender's injection rate falls from 64 GB/s (one receiver) to 11 to 12 GB/s once a rectangle has about ten or more receivers, so one sender core delivers about 1.3 to 1.5 TB/s of aggregate multicast traffic to a full 12x10 grid. Latency grows with the receiver count (about 19 cycles per receiver in this test, which includes the acks being serialized at the sender), from 0.2 us to 1.9 us.

### Host to chip bandwidth (Spike E)

256 MiB DRAM buffer, 64 KiB pages, median of 3 repetitions, buffers verified bit for bit. `spikes/e_tray`, nodes 0-7 as a 2x4 mesh (and single chips for comparison). Write = host to chip DRAM, read = chip DRAM to host.

| Device (mesh coordinate, node) | Link | Write GB/s | Read GB/s |
| --- | --- | --- | --- |
| node 0, unit mesh | x1 | 3.22 | 3.20 |
| node 5, unit mesh | x8 | 19.82 | 13.37 |
| `[0,0]` node 4 (in the 8-chip mesh) | x1 | 3.22 | 3.20 |
| `[0,1]` node 5 | x8 | 13.46 | 15.92 |
| `[0,2]` node 6 | x1 | 3.23 | 3.20 |
| `[0,3]` node 7 | x1 | 3.23 | 3.20 |
| `[1,0]` node 0 | x1 | 3.23 | 3.20 |
| `[1,1]` node 1 | x1 | 3.23 | 3.20 |
| `[1,2]` node 2 | x1 | 3.23 | 3.20 |
| `[1,3]` node 3 | x1 | 3.23 | 3.20 |
| all 8 at once, aggregate | n/a | 25.99 (one replicated write) | 11.34 (non-blocking per-shard reads) |

### Chip to chip (Spike E, `bench_unicast`)

Fabric `FABRIC_2D` over nodes 0-7 (mesh 2x4). One sender core to one receiver core, DRAM source, DRAM or L1 destination as the tool does it, 4 KiB pages, 10 timed iterations of a captured trace of 64 enqueues; median per transfer. Fabric chip id `m:c` is the row-major mesh index (chip 0 = node 4 at `[0,0]`, chip 4 = node 0 at `[1,0]`).

| Pair | What it is | Size | p50 time | p50 GB/s |
| --- | --- | --- | --- | --- |
| 0 to 1 | horizontal neighbour | 4 KiB | 3.5 us | 1.16 |
| 0 to 1 | horizontal neighbour | 64 KiB | 9.2 us | 7.11 |
| 0 to 1 | horizontal neighbour | 1 MiB | 103 us | 10.20 |
| 0 to 1 | horizontal neighbour | 4 MiB | 401 us | 10.45 |
| 0 to 1 | horizontal neighbour | 16 MiB | 1594 us | 10.53 |
| 1 to 2 | horizontal neighbour | 4 MiB | 401 us | 10.46 |
| 0 to 3 | three hops along a row | 4 MiB | 401 us | 10.46 |
| 0 to 4 | vertical neighbour | 4 MiB | 284 us | 14.77 |
| 0 to 7 | diagonal: one vertical plus three horizontal hops | 4 MiB | 284 us | 14.78 |

## CPU reference (Phase 1)

One thread of the host CPU, Release build, 1920 x 1088, FP32 tile-relative evaluation as in docs/FINDINGS.md. The reference evaluates every pixel of every (tile, triangle) pair in a bin except Full tiles, so time follows the pair count and is not a performance target; it exists to be correct. `ttr_render all` prints these.

| Scene | Pieces | Tile pairs | Setup | Binning | Render |
| --- | --- | --- | --- | --- | --- |
| one_triangle | 1 | 526 | 0.0 ms | 0.0 ms | 34 ms |
| fan | 96 | 3,532 | 0.0 ms | 0.2 ms | 108 ms |
| grid | 4,080 | 15,135 | 0.3 ms | 0.4 ms | 247 ms |
| interpenetrating | 27 | 2,335 | 0.0 ms | 0.1 ms | 72 ms |
| fullscreen | 2 | 2,111 | 0.0 ms | 0.1 ms | 84 ms |
| tiny_10k | 8,176 | 8,681 | 0.7 ms | 0.3 ms | 80 ms |
| random_10k | 8,912 | 80,668 | 1.0 ms | 2.0 ms | 1,203 ms |

Host-side setup plus binning for 10,000 triangles is about 3 ms, so it will not be the bottleneck of the first Tensix frame. (The binning column was measured with the first implementation; see "Binning" below for what it takes now.)

## Tensix rasterizer (Phase 2)

All on node 0 unless stated, 1920 x 1088, all 120 worker cores, AI clock 1.35 GHz. "Device" is the host-measured time of enqueue, run and finish of one frame with the JIT cache warm (it includes the launch, about 0.3 ms for one core and 0.75 ms for 120). Minimum of 15 runs.

### Device time, v1 against now (three output planes, so both read back depth and id)

| Scene | v1 (commit 5d4fb63) | Now | Colour plane only |
| --- | --- | --- | --- |
| one_triangle | 0.93 ms | 0.93 ms | 0.72 ms |
| fan | 1.95 ms | 1.23 ms | 1.01 ms |
| grid | 1.88 ms | 1.08 ms | 0.92 ms |
| interpenetrating | 1.39 ms | 0.91 ms | 0.97 ms |
| fullscreen | 1.00 ms | 0.87 ms | 0.83 ms |
| tiny_10k | 1.52 ms | 0.93 ms | 0.76 ms |
| random_10k | 6.59 ms | 3.07 ms | 2.93 ms |

The floor (every tile cleared, quantized and packed, no triangles drawn, `RASTER_EXP_SKIP_TRIANGLES`) is 0.88 ms for 120 cores x 17 tiles, so the one-triangle row is almost all floor.

### Experiments on `random_10k` (busiest core draws 827 of 80,668 records; v1 6.70 ms at the start)

| Step | Device ms | Result |
| --- | --- | --- |
| v1: runtime vector index in the loop | 6.70 | baseline |
| Constant instruction words (`dst_reg++`, counter-relative addressing) | 5.18 | -23% |
| Flat-colour path, per-vector class skip (reject) and no edge tests for full vectors | 3.16 | -39% more, -53% in total |
| One SFPU call per page of records instead of per record | 3.03 | no gain; not kept |

Ablations of the final kernel (colour plane only, same scene):

| Variant | Device ms |
| --- | --- |
| final | 2.95 |
| no triangles drawn (floor) | 0.88 |
| no colour writes | 1.82 |
| every non-rejected vector treated as partial (all edge tests) | 3.49 |
| every non-rejected vector treated as full (no edge tests) | 2.38 |

From these: about 135 cycles per fully covered vector, about 220 per partial one, 54% of the triangle time in colour writes, and 3 edge tests are worth 83 cycles per vector. Vectors per record on this scene: 7.4 partial and 10.7 full.

### Colour layout (Phase 2.4): FP32 planes against packed RGBA8 per pixel (colour plane only, ms)

| Scene | Planes (default) | Packed |
| --- | --- | --- |
| one_triangle | 0.60 | 0.83 |
| fan | 0.88 | 0.86 |
| grid | 0.77 | 0.69 |
| interpenetrating | 0.72 | 0.66 |
| fullscreen | 0.62 | 0.58 |
| tiny_10k | 0.63 | 0.55 |
| random_10k | 2.66 | 2.78 |

### Frame time breakdown, one chip (colour plane only, median of 10 warm frames, ms)

Host setup and binning (CPU, one thread) are in the Phase 1 table (at most 3 ms). The rest, from the 1-chip rows of `ttr_multichip` (persistent host buffers, record encoding on 64 threads):

| Scene | Record build | Upload | Device | Read-back (8.4 MB over one x1 PCIe link) |
| --- | --- | --- | --- | --- |
| one_triangle | 4.4 | 2.6 | 0.78 | 3.7 |
| fan | 4.5 | 2.6 | 0.71 | 3.8 |
| grid | 4.4 | 2.6 | 0.83 | 3.9 |
| interpenetrating | 4.4 | 2.6 | 0.76 | 3.5 |
| fullscreen | 4.5 | 2.6 | 0.93 | 4.1 |
| tiny_10k | 4.3 | 2.7 | 0.79 | 3.9 |
| random_10k | 4.9 | 4.9 | 2.47 | 4.0 |

The device is the smallest part of a frame; before persistent buffers the same read-back took 7 ms and the record build 8 to 12 ms.

### Multi-chip (Phase 2.6): frame time and identity

Nodes 0 to 7 of one tray, chips opened together as independent unit meshes, colour plane only, median of 10 frames. "Frame" is the wall time of the whole frame: host partition and encode, upload, device, read-back, merge. Every cell was checked: the merged image is bit-identical to the 1-chip image and equals the reference except on the depth-ambiguous pixels (17 on `random_10k`, which agree with the single chip anyway).

| Scene | 1 chip | 2 chips | 4 chips | 8 chips |
| --- | --- | --- | --- | --- |
| one_triangle | 12.2 ms | 8.0 ms | 6.6 ms | 6.3 ms |
| fan | 12.2 ms | 8.0 ms | 7.1 ms | 6.2 ms |
| grid | 12.5 ms | 8.0 ms | 6.3 ms | 6.2 ms |
| interpenetrating | 12.3 ms | 8.0 ms | 6.8 ms | 6.5 ms |
| fullscreen | 12.4 ms | 7.9 ms | 6.4 ms | 7.5 ms |
| tiny_10k | 12.3 ms | 8.0 ms | 6.3 ms | 6.1 ms |
| random_10k | 16.5 ms | 10.2 ms | 7.6 ms | 7.3 ms |

Slowest chip's phases on `random_10k` (build, upload, device, read-back, ms): 1 chip 4.9 / 4.9 / 2.5 / 4.0; 2 chips 3.9 / 2.5 / 1.6 / 2.1; 4 chips 3.8 / 1.3 / 1.1 / 1.1; 8 chips 3.6 / 1.6 / 1.1 / 1.2. Records per chip with 8 chips: 10,059 to 10,107 (balanced within 0.5%), tiles 240 to 278.

Reading: device and read-back shrink with the chip count (read-back runs on eight independent PCIe links), but record encoding and the fixed device launch put a floor of about 6 ms on a frame, so beyond two or three chips only heavy scenes gain. Before persistent buffers and a shared framebuffer the 8-chip frame took 34 ms (page-fault contention, see FINDINGS).

## Tiled-camera rendering for Isaac Lab (Track I)

`validation/bench_scene.py`, 1,024 environments of 96 x 96, 316 triangles per environment (pole capsule 4 x 16), poses change every frame, median of 9 frames after a warm-up, outputs written into preallocated host arrays. Chips: one x8 chip (node 5), one x1 chip (node 0) or all eight of the tray. Seven of the eight chips have x1 PCIe links, so which chip is used matters (see below).

| Outputs | Passes | 1 chip, x8: frame | 1 chip, x8: images/s | 1 chip, x1: frame | 8 chips: frame |
| --- | --- | --- | --- | --- | --- |
| depth | 1 | 32.8 ms | 31,200 | 63.3 ms | 32.8 ms |
| rgb + depth | 1 | 39.2 ms | 26,100 | 78.7 ms | 38.3 ms |
| depth, shape, rgb, normals | 2 | 77.4 ms | 13,200 | 165.7 ms | 75.7 ms |

Breakdown of the depth frame (ms), slowest chip where several: vertex stage 3.3, setup 4.6, binning 1.5, record encoding 2.5, upload 3.0 (x1 chip: 14.0), device kernels 1.4, read-back 8.1 (transfer 4.9, untiling 3.2; x1 chip: 27.6 with 24.2 transfer), frame construction 2.6, output assembly 2.7. With eight chips: upload 2.4, device 0.9, read-back 8.1 (transfer 6.9), so the transfer of an eighth of the data per chip takes as long as the whole frame's transfer on the x8 chip. CPU reference backend, same scene, depth: 827 ms for 1,024 environments (256: 219 ms). History of the 1-chip (x8) depth frame: 126 ms (serial setup 25 ms, assembly 27 ms), 57 ms after parallelizing the host stages, 36.5 ms after reusing the frame's triangle storage (frame construction 20.6 ms to 2.4 ms), 32.8 ms after reusing setup and bin storage (setup 6.1 to 4.6 ms, binning 1.6 to 1.5 ms) and writing into preallocated output arrays (assembly 5.2 to 2.7 ms). Quality against Newton Warp is in FINDINGS.md.

### Pipelined frames and the host thread pool

Two changes after the table above. (1) `parallel_for` now runs on one persistent worker pool (`src/ttraster/parallel.h`) instead of creating and joining up to 64 `std::thread`s in every stage; the stages shrank (frame construction 2.5 to 0.8 ms, setup 4.6 to 2.0 ms) and a synchronous depth frame went from about 33 to 25 to 29 ms on the x8 chip. (2) `ttr_submit` / `ttr_collect` (`Scene.submit` / `Scene.collect` in Python) keep two frames in flight: poses are snapshotted at submit, each frame has its own storage slot (triangles, bins, record buffer, read-back buffer, framebuffer), and the chips serialize only their device section (buffers, upload, run, transfer) with a mutex, so the host stages of one frame overlap the device section of the other. Output is bit-identical to synchronous frames (`validation/test_pipeline.py`, CPU and Tensix backends); the price is one frame of latency.

1,024 environments, 96 x 96, depth, steady state over 80 frames, three runs each (`validation/bench_scene.py --pipeline`; the host carried other users' load during these runs, load average 35, so expect +-15 %):

| Chips | Synchronous | Pipelined (2 in flight) | Images/s pipelined | Latency submit to collect |
| --- | --- | --- | --- | --- |
| 1 (x8 link, node 5) | 25 to 29 ms | 19.6 to 20.1 ms | about 51,000 | 34 to 35 ms |
| 8 | 31 to 36 ms | 19.6 to 21.3 ms | about 50,000 | 33 to 39 ms |
| 1 (x1 link, node 0) | 55.7 ms | 43.8 ms | 23,400 | 81 ms |

Other outputs, one run: rgb + depth 35.2 to 25.2 ms (x8 chip), four outputs 62.5 to 51.9 ms; the multi-pass frames gain less because every pass already keeps the chip busy.

Where the pipelined frame goes (`TTR_TRACE=1` prints a timeline of stage boundaries per slot): one frame is about 38 ms long, a new one starts every 18 to 20 ms. The device section (create buffers, upload, run, transfer) is 10 to 12 ms of it and is mutually exclusive, so a single chip cannot go below that; the rest is host work (vertex 3.5, construction 3, setup 2, binning 3 before the change described under "Binning", 1.3 after, encode 2, untile 2, assemble 2 ms) that now runs concurrently with the other frame. With eight chips the device sections of different chips run in parallel, but the frame still takes as long as with one x8 chip, because the transfer problem described above (eight chips do not read back eight times faster) and the host stages remain.

Binning (`bin_triangles`, `src/ttraster/tile.cpp`). The loop kept its counters in the caller's `TileBins` and pushed into vectors reachable through the same reference, so the compiler reloaded and stored them for every (tile, triangle) pair; with local counters the single-thread loop on `random_10k` at 1920 x 1088 takes 0.35 to 0.41 ms instead of 1.35 ms (host-only, back to back on the same machine). Tile rows are independent (a bin belongs to exactly one row), so above 2,048 triangles and 16,384 candidate pairs the rows are binned as one band per row on the worker pool, each band walking the triangles in API order; a compact array of (first row, last row) per triangle keeps a band from touching the 330-byte `SetupTriangle` of triangles outside its row. The result is identical to the sequential loop (`tests/test_reuse.cpp` compares it with a plain loop on 17,000- and 13,000-triangle frames). Two things cost time on the way: per-band counters in one array shared cache lines between threads (2.4 ms instead of 1.2 ms for 36,000 triangles), and `getenv` in the inner loop of an experiment. On the Track I frame (1,024 x 96 x 96 depth, 323,584 triangles, 94,632 records, x8 chip, loaded machine) binning went from 2.7 to 3.7 ms with the sequential loop to 1.2 to 1.4 ms, and the synchronous frame from 27.0 to 24.3 to 25.8 ms. The pipelined frame did not get faster (21.4 to 21.7 ms per frame in two runs): the device section and the other host stages decide it, and the host stages share one worker pool. The gain does not grow with triangle count the way the core count suggests: at 143,000 `random_10k` triangles the banded and sequential loops both take about 5 ms, probably because the bands re-read the same large triangle structures from other cores' caches.

Empty tiles are no longer sent to the chip (`MultiChipRenderer`): a tile with no record shows only the clear values, so the host writes those (only when the tile does not hold them already) instead of encoding, drawing and reading back a blank tile. About a third of the tiles of this scene are sky. Measured afterwards, same scene (one run each, host load average about 20): x1 chip, depth, transfer 24.2 to 15.8 ms, synchronous 55.7 to 42.6 ms, pipelined 43.8 to 30.5 ms; x8 chip 24.4 ms synchronous and 19.1 ms pipelined; eight chips 23.4 ms synchronous and 17.7 ms pipelined (58,000 images/s); rgb + depth on the x8 chip 27.6 and 22.8 ms. Pixels are unchanged: the device acceptance run (every scene on 1, 2, 4 and 8 chips against the CPU reference) and the Newton comparisons pass.

A process-wide CPU census explains why the first version of the pipeline gained little: a synchronous frame burned 443 ms of CPU time (12 cores busy on average, 65 ms of it system time) before the pool, mostly in creating and joining threads; with the pool the frame is shorter but uses more cores while it runs (750 ms of CPU, 23 cores), so the host stages are now CPU-bound when two frames overlap.

### Per-pixel lighting of `rgb` (`ttr_set_lighting`)

1,024 environments, 96 x 96, 356,352 triangles (coarse tessellation), node 5, median of 10 frames after 4 warm-up frames, synchronous `Scene.render`:

| Output | Lighting | Frame | Device section | Encode | Upload |
| --- | --- | --- | --- | --- | --- |
| rgb | per-vertex (default) | 29.1 ms | 3.9 ms | 4.1 ms | 3.7 ms |
| rgb | per-pixel (40-instruction shader) | 84.6 ms | 47.3 ms | 10.4 ms | 6.2 ms |
| depth + rgb | per-vertex | 33.9 ms | 4.0 ms | 4.1 ms | 3.7 ms |
| depth + rgb | per-pixel | 90.9 ms | 47.2 ms | 10.4 ms | 5.8 ms |

The shader runs for every covered vector of every triangle (about 40 instructions at 25 to 45 ns, "Interpreter cost" below), and these triangles are a few pixels each, so the cost scales with triangle count and not with pixels. The image gain is small (`docs/FINDINGS.md`: interior pixels 49.4 to 50.8 dB against Newton, whole image 38.05 to 38.13 dB), so the option is off by default.

### Isaac Lab renderer: asynchronous rendering (`async_rendering`)

`TtRasterRendererCfg.async_rendering` uses the pipeline above the way Isaac Lab's delayed captures do (`OVRTXRendererCfg.async_rendering`, `BaseRenderer.prepare_capture`): `render()` submits the frame of the current step, collects the frame of the previous step and publishes it into the camera's buffers, so the chip renders while the physics step and the policy run, at the price of one step of latency in the camera observations (the first capture and the first after a reset wait for a fresh image; the pose, calibration and frame indices of the delivered image come with it in `camera.data.info[output]["capture"]`). `validation/bench_isaaclab_shim.py` (stand-ins for Isaac Lab, a real Newton model of 1,024 environments with coarse shape tessellation: 356,352 triangles per frame, 96 x 96, `distance_to_camera`, node 5 = x8 chip, median of 30 steps, a sleep stands in for physics and policy):

| Other work per step | Synchronous step | `render()` blocks | Asynchronous step | `render()` blocks |
| --- | --- | --- | --- | --- |
| 0 ms | 41.4 ms | 41.4 ms | 20.4 ms | 20.4 ms |
| 20 ms | 67.0 ms | 46.9 ms | 31.3 ms | 11.2 ms |
| 40 ms | 87.4 ms | 47.3 ms | 47.2 ms | 7.1 ms |

Without other work the step is the pipelined frame time (two frames in flight between `submit` and `collect`); with other work it is the larger of the work and one frame's latency (31 ms here), plus what `render()` still does on the calling thread (pose update, collect, copying the image into the output buffer). The synchronous shim costs 41 ms where the C API alone takes 25 to 30 ms in `docs/PERF.md` above because the Newton bridge recomputes instance poses and the output buffers are copied once more. With the default tessellation of the same model (3.2 million triangles per frame) a synchronous frame takes 135 ms and an asynchronous step with 20 ms of work 90 ms (one run).

## OpenGL: glxgears through Mesa (Track G)

`scripts/run_gl.sh 5 [--backend off] glxgears -geometry WxH`, Xvfb, Mesa 26.3.0 softpipe on this host's CPU (single-threaded, the stock driver) against the same Mesa with the Tensix fast path on the x8 chip (node 5), three 5-second windows each, host load average about 6:

| Window | Stock softpipe | Tensix fast path | Ratio |
| --- | --- | --- | --- |
| 300 x 300 | 571 FPS | 333 FPS | 0.58 |
| 400 x 400 | 384 FPS | 303 FPS | 0.79 |
| 500 x 500 | 273 FPS | 265 FPS | 0.97 |
| 700 x 700 | 153 FPS | 239 FPS | 1.6 |
| 1024 x 1024 | 79 FPS | 217 FPS | 2.8 |
| 1920 x 1080 | 31 FPS | 160 FPS | 5.1 |

(Median of the 5-second windows of a 16-second run each; `TTR_GL_MIN_PIXELS=0` so that small windows use the chip too. Before the copy changes described below the Tensix column read 310 / 173 / 126 FPS at 300 x 300, 1024 x 1024 and 1920 x 1080.)

Per-frame breakdown of the fast path at 1920 x 1080 (`TTR_GL_DEBUG=1`): 3.9 ms in the rasterizer call (setup 0.2, binning 0.2, record encoding 0.3, upload 0.6, device 0.8, read-back 0.8 of which 0.4 is the transfer, and 0.4 for the copy into Mesa's colour surface); the remaining 2.4 ms of the 6.3 ms frame are Mesa (vertex processing in the draw module) and the swap into the X server. At 300 x 300 the rasterizer takes 2.0 ms for 750 records of which about 1.7 ms is chip latency, which is why small windows do not gain. What helped: writing the colour surface directly as words instead of through softpipe's float tile cache (6 to 10 ms per 1080p frame to 0.75 ms), deferring the depth write (2.7 ms at 1080p) until something reads the depth buffer, and then (frame 8.6 to 6.3 ms, 116 to 160 FPS) three changes to the copies: (1) the frame is ended without its depth plane, so the chip's depth read-back and the host copy of 8 MB disappear from every frame; if something does read the depth buffer later (a depth `glReadPixels`, a draw after a swap without a clear, a partial clear), `ttr_frame_depth` renders the frame again for its depth plane, and a draw that falls back to softpipe in the middle of a frame still asks for depth with the frame, so it renders once; a program that reads the depth buffer every frame would render every frame twice, so after the first re-render frames end with their depth plane until four frames in a row did not use it (a random-frame run that reads the depth buffer of all 16 frames rendered 1 depth plane afterwards); (2) the rasterizer writes straight into Mesa's colour surface with its row pitch, converting R/B order and alpha on the way (`ttr_frame_end_ex`), which removes the scratch RGBA image and a second 8 MB pass; (3) the row copies run on the host thread pool (1.1 ms single-threaded for colour and depth, 0.4 ms for colour alone). Blending and per-draw state cost more on the device: a record with its own depth/blend state takes the kernel's general path (about 11.5 us per record on the `state` scene against 3.5 us for default-state records on `random_10k`; the scenes differ, so this is an upper bound for the extra work). Not done: overlapping frames with Mesa's work. It was considered and rejected for the default: the window system copies the colour surface to the display inside `glXSwapBuffers`, so a rendering still in flight at the swap can only be shown one frame late, and a program that draws one last frame and then waits would never see it. Zero-copy read-back is done for the Mesa path in the sense that the only host copy left is the one into Mesa's surface.

## Fragment shaders (G.6)

### Interpreter cost (`tools/ttr_shader_bench`, node 5, 1920 x 1088, one full-screen quad, 120 cores, median of 20 warm runs)

The program is a chain of K instructions of one kind on one register; the device time rises linearly with K. Time per instruction and per pixel vector (32 lanes) on the busiest core (17 tiles of 2 triangles x 32 vectors):

| instruction | K = 32 | K = 128 | K = 128 before the change below | device ms at K = 32 / 128 |
| --- | --- | --- | --- | --- |
| Mad (3 operands) | 43.4 ns | 44.6 ns | 39.0 ns | 2.84 / 7.53 |
| Mul (2) | 35.1 | 34.1 | 40.1 | 2.80 / 6.32 |
| Slt (2) | 32.9 | 30.2 | 37.6 | 2.85 / 5.90 |
| Rcp (1) | 28.3 | 26.3 | 40.6 | 2.63 / 5.31 |
| Ex2 (1) | 37.1 | 34.6 | 48.1 | 2.90 / 6.43 |
| Sin (1) | 24.5 | 37.7 | 53.2 | 2.47 / 6.88 |

The first version fetched all three operands of every instruction (an unused operand reads register 0). Every case of the interpreter now fetches only the operands it uses: a fetch is a runtime-indexed load that the RISC-V core assembles and issues, so an instruction with one source saves two of them. One-source instructions got 25 to 35 % faster, two-source ones 10 to 15 %, Mad (three sources) is unchanged within the noise of the machine (+-10 % between runs; the Mad row above is from a run that was 10 % slower than another one of the same build, 40.8 / 41.9 ns).

About 25 to 45 ns (30 to 55 cycles) for an instruction, depending on its operand count: fetching and decoding the instruction word and its three operands on the RISC-V core, and issuing the SFPU operations through runtime-indexed `dst_reg`, cost far more than the arithmetic (the whole `sin` is only 14 ns more than a `mad`). A 1080p frame with a 32-instruction shader over the whole screen costs 2.5 ms on the device, a 128-instruction one 6 to 7 ms. The 1.0 ms at K = 0 is the fixed cost of a frame (clear, packed colour, read-back path). Decoding is the place to optimize (pre-decoded operand fields, superinstructions such as Mad with constant operands, dot products as one instruction).

Where the 40 ns go (Mad chain, K = 128, same benchmark with timing-experiment defines): reading the instruction words and looping without issuing anything costs 8.9 ns; dropping the result store saves only 3 ns (37.5 ns); a chain of independent Mads (no result is read back through Dest) costs the same 40 ns, so it is issue cost, not Dest read-after-write latency; with register operands only (no constant, which needs two `sfploadi` with a runtime immediate) it is 28 to 34 ns. The disassembly shows why: every operand access is a `dst_reg[...]` with a runtime index, which the compiler turns into building the SFPU instruction word on the RISC-V core (load the template from the stack, add the index, store it to the instruction buffer), about 6 to 8 RISC-V instructions and a store per access, and a constant is two of those plus a load. An interpreter on this core is therefore bound by pushing instructions, not by the SFPU; only compiling the shader to SFPI source (compile-time registers and addresses, values kept in LREGs across instructions) would remove it, at the price of a JIT build per shader set (tens of seconds, cached).

Load balancing by program length did not help: with a cost model for shader records (per pixel vector `a + b * instructions`, tried with a from 0 to 1.2 us and b 0.04 and 0.08 us, partial tiles 12 and 20 vectors) the `shader` scene at 1920 x 1088 took 2.55 to 3.4 ms on the device against 2.53 ms with the plain per-record weights of `plan_balanced` (and 4.76 ms with equal tile counts, `--mode grid`); the model was reverted. `ttr_device --mode balanced --plan-stats` shows the spread (records per core 32 / 54 / 92 minimum / median / maximum).

### Scene `shader` and OpenGL

The `shader` scene (background from `gl_FragCoord`, perspective checkerboard, Phong sphere, discard stripes, translucent layer; 6,524 records at 1920 x 1088): device 4.6 ms minimum, 5.1 ms median, whole `render()` 12.7 ms; the busiest core draws 257 records at 18 us each (a fixed-function record is 3.5 us).

The default test scene of a few large triangles with the per-pixel Phong shader (GLSL fragment shader of 44 translated instructions, `gl_scene --fs phong`) at 1024 x 1024 through Mesa: stock softpipe 73 ms per frame, one chip 9.7 ms per frame (differences of two runs of different length, so start-up cancels): 4.6 ms in the rasterizer call (device 2.2 ms, read-back 0.7 ms, 763 records), 0.3 ms writing the result into softpipe's buffers, the rest is Mesa's vertex stage and the `glReadPixels` of the test program.
