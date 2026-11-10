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

Host-side setup plus binning for 10,000 triangles is about 3 ms, so it will not be the bottleneck of the first Tensix frame.

