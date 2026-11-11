// SPDX-License-Identifier: Apache-2.0
//
// Data layout shared by the host renderer and the Tensix kernels (docs/FINDINGS.md, "Phase 2").
// No tt-metal includes: this header is compiled by the host and by the device JIT.
//
// Work for one core is a sequence of tile blocks in DRAM. A block is a whole number of pages;
// a page is kRecordsPerPage records of kRecordBytes. Record slot 0 of a block's first page is the
// block header, the triangle records follow, the last page is zero padded.
//
//   header:  word 0 = number of triangle records in the block
//            word 1 = tile x, word 2 = tile y (informational)
//   record:  32 words (128 B, a multiple of the 64 B DRAM read alignment, V9), see below.
//            Words 26 and 27 hold a 2-bit class per SFPU vector (core/src/ttraster/tile.h, classify_vectors).
//
// Output: for every block the core produces kOutPlanes tiles (4 KB each, FP32 tile layout, faces
// in the order TL, TR, BL, BR, each 16x16 row-major): packed RGBA8, depth, triangle id.

#pragma once

#include <cstdint>

namespace ttr::layout {

constexpr uint32_t kRecordWords = 32;
constexpr uint32_t kRecordBytes = kRecordWords * 4;
constexpr uint32_t kRecordsPerPage = 32;
constexpr uint32_t kPageBytes = kRecordBytes * kRecordsPerPage;  // 4 KiB: one FP32 tile, so CB page == tile

// Record word offsets (all IEEE floats except where noted).
constexpr uint32_t kEdge = 0;       // 3 edges x (a, b, c)         words 0..8
constexpr uint32_t kDepth = 9;      // zc, zgx, zgy                words 9..11
constexpr uint32_t kColor = 12;     // 4 channels x (c, gx, gy)    words 12..23, order R, G, B, A
constexpr uint32_t kId = 24;        // 1 + source triangle index, as a float (exact below 2^24)
constexpr uint32_t kFlags = 25;     // uint32 bit flags
constexpr uint32_t kFlagFlat = 1u;  // colour gradients are zero (flat shaded)
constexpr uint32_t kFlagFull = 2u;  // every pixel of the tile is inside (informational: the classes say it too)
constexpr uint32_t kFlagStateful = 8u;    // the triangle has its own depth/blend state: bits below
constexpr uint32_t kDepthFuncShift = 4;   // 2 bits: 0 LESS, 1 LESS_EQUAL, 2 ALWAYS (matches ttr::DepthFunc)
constexpr uint32_t kFlagDepthWrite = 1u << 6;
constexpr uint32_t kBlendShift = 7;       // 3 bits: 0 none, 1 alpha, 2 add, 3 premultiplied, 4 multiply, 5 alpha-add (ttr::BlendMode)
constexpr uint32_t kFlagPersp = 4u;  // colour planes hold colour * rhw; the plane in kWPlane is rhw (colour = plane / rhw)
constexpr uint32_t kWPlane = 29;    // rhw plane (c, gx, gy), words 29..31, perspective triangles only
constexpr uint32_t kFlatPacked = 28;  // flat colour as packed RGBA8 (host-computed), for the packed-colour variant
constexpr uint32_t kVecClass = 26;  // 2 words, 2 bits per SFPU vector: 0 reject (skip), 1 partial (edge tests), 2 full

// Fragment shaders (frames with shader triangles only; docs/FINDINGS.md, "Fragment shaders"). Every record of such a frame
// carries its depth and blend state in the stateful bits. A shader record has kFlagShader and
//   * words 12..23: the planes (c, gx, gy) of varying components 0..3 (where fixed-function records have their colour),
//   * word 28: the program index (kProgramId), words 29..31: the rhw plane (c, gx, gy),
//   * kExtSlots(flags) extension slots after it (32 words each, 3 words per further varying component, component 4 first),
//     so a record takes 1 + ext slots in the page; a record never straddles a page.
// Fixed-function records of such a frame have no kFlagShader: their four colour planes are the inputs of a built-in
// program that returns them. The first unused slot of a page that the next record does not fit in holds kPageEnd in its
// flags word.
constexpr uint32_t kFlagShader = 1u << 10;
constexpr uint32_t kExtShift = 11;           // 2 bits: extension slots (0..3)
constexpr uint32_t kProgramId = 28;
constexpr uint32_t kPageEnd = 0xFFFFFFFFu;
constexpr uint32_t kInputsInRecord = 4;
constexpr uint32_t ext_slots(uint32_t n_inputs) {
    return n_inputs > kInputsInRecord ? (3 * (n_inputs - kInputsInRecord) + kRecordWords - 1) / kRecordWords : 0;
}
// Program table (read once into L1): word 0 = number of programs, then 8-word directory entries from word 1:
//   code offset (words from the table start), instruction count, constants offset, inputs, fragcoord register + 1 (0: none),
//   output register, uses kill, reserved. Code and constants follow the directory.
constexpr uint32_t kMaxTablePages = 16;   // 64 KiB of programs and constants per frame
constexpr uint32_t kTabDirectory = 1;
constexpr uint32_t kTabEntryWords = 8;

// Header word offsets (inside record slot 0 of the block's first page).
// Blend of one channel (src s, src alpha sa, dst d), the same chain on host and device:
//   A = sa * ka_sa + ka_c;  B = s * kb_s + (sa * kb_sa + kb_c);  r = clamp01(s * A + d * B)
// Rows by BlendMode (None unused): Alpha (sa, 1 - sa), Add (1, 1), Premultiplied (1, 1 - sa), Multiply (0, s), AlphaAdd (sa, 1).
// Columns: ka_c, ka_sa, kb_c, kb_sa, kb_s.
constexpr float kBlendCoef[6][5] = {
    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 1.0f, -1.0f, 0.0f},
    {1.0f, 0.0f, 1.0f, 0.0f, 0.0f},
    {1.0f, 0.0f, 1.0f, -1.0f, 0.0f},
    {0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
    {0.0f, 1.0f, 1.0f, 0.0f, 0.0f},
};

constexpr uint32_t kHdrRecords = 0;
constexpr uint32_t kHdrTileX = 1;
constexpr uint32_t kHdrTileY = 2;
constexpr uint32_t kHdrPages = 3;   // pages of the block (in shader frames the pages are not derived from the record count)
constexpr uint32_t kHdrCx = 4;      // shader frames: centre of the tile in window pixels, as float bits (tx * 32 + 16)
constexpr uint32_t kHdrCy = 5;

// Output planes per tile block.
constexpr uint32_t kPlaneColor = 0;
constexpr uint32_t kPlaneDepth = 1;
constexpr uint32_t kPlaneId = 2;
constexpr uint32_t kOutPlanes = 3;

constexpr uint32_t kTileWords = 1024;
constexpr uint32_t kTileBytes = kTileWords * 4;

// Pages in a block holding n records (header takes one record slot).
constexpr uint32_t pages_for_records(uint32_t n) { return (n + 1 + kRecordsPerPage - 1) / kRecordsPerPage; }

}  // namespace ttr::layout
