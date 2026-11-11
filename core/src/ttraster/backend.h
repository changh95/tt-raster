// SPDX-License-Identifier: Apache-2.0
//
// The seam between the frame description (frame.h) and whatever rasterizes it: the CPU reference, the Tensix renderer on
// one or several chips. Everything above the core (the C frame API, the Mesa and Vulkan drivers, the plugins) talks to a
// Backend and never to a device.

#pragma once

#include <cstdint>
#include <string>

#include "ttraster/frame.h"
#include "ttraster/reference.h"

namespace ttr {

// Planes of the framebuffer a caller needs back from the rasterizer.
enum RenderPlane : uint32_t {
    kRenderColor = 1,     // packed RGBA8 colour (or, with kRenderFloat, the unquantized R, G, B channels)
    kRenderDepth = 2,
    kRenderId = 4,       // 1 + index of the visible triangle, 0 = background
    kRenderFloat = 8,    // colour channels as unquantized floats in Framebuffer::plane[0..2]
};

// Something that rasterizes a Frame. The CPU reference is one; the Tensix renderer is another.
// `planes`: which planes of the returned framebuffer must be valid (others may be left empty).
class Backend {
public:
    virtual ~Backend() = default;
    // `slot` (0 .. kRenderSlots-1) selects the backend's storage for this frame: two frames may be rendered
    // at the same time on different threads if they use different slots. The returned framebuffer stays
    // valid until the next call with the same slot.
    virtual const Framebuffer& render(const Frame& frame, uint32_t planes, int slot = 0) = 0;
    // One line of phase timings of the most recent render() call of that slot, for logs and benchmarks.
    virtual std::string report(int slot = 0) const { (void)slot; return {}; }
};

class ReferenceBackend : public Backend {
public:
    const Framebuffer& render(const Frame& frame, uint32_t planes, int slot = 0) override;

private:
    Framebuffer fb_[kRenderSlots];
};

}  // namespace ttr
