// SPDX-License-Identifier: Apache-2.0

#include "ttraster/backend.h"

namespace ttr {

const Framebuffer& ReferenceBackend::render(const Frame& frame, uint32_t planes, int slot) {
    ReferenceOptions opt;
    opt.float_planes = (planes & kRenderFloat) != 0;
    fb_[slot] = render_reference(frame, opt);
    return fb_[slot];
}

}  // namespace ttr
