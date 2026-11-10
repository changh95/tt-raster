// SPDX-License-Identifier: Apache-2.0
//
// Procedural, deterministic test scenes (docs/PLAN.md, Phase 1.4). Every scene is a pure
// function of its name and the viewport size: the same inputs give bit-identical frames on any
// machine (own RNG, no standard-library distributions).

#pragma once

#include <string>
#include <vector>

#include "ttraster/frame.h"

namespace ttr {

struct SceneInfo {
    const char* name;
    const char* description;
    bool watertight_mesh;  // inside the mesh every pixel must be covered exactly once
};

const std::vector<SceneInfo>& scene_list();

// Build scene `name` for a width x height viewport. Returns false for an unknown name.
bool make_scene(const std::string& name, uint32_t width, uint32_t height, Frame& out);

}  // namespace ttr
