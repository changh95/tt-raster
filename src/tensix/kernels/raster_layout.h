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
//            Words 26 and 27 hold a 2-bit class per SFPU vector (src/ttraster/tile.h, classify_vectors).
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
constexpr uint32_t kFlagPersp = 4u;  // colour planes hold colour * rhw; the plane in kWPlane is rhw (colour = plane / rhw)
constexpr uint32_t kWPlane = 29;    // rhw plane (c, gx, gy), words 29..31, perspective triangles only
constexpr uint32_t kFlatPacked = 28;  // flat colour as packed RGBA8 (host-computed), for the packed-colour variant
constexpr uint32_t kVecClass = 26;  // 2 words, 2 bits per SFPU vector: 0 reject (skip), 1 partial (edge tests), 2 full

// Header word offsets (inside record slot 0 of the block's first page).
constexpr uint32_t kHdrRecords = 0;
constexpr uint32_t kHdrTileX = 1;
constexpr uint32_t kHdrTileY = 2;

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
