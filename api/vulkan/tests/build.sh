#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the Vulkan tests into $1 (default: a temporary directory): compile the shaders to SPIR-V headers, then the C programs.
#   needs glslangValidator, a C compiler and the Vulkan headers and loader (libvulkan-dev)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${1:-$(mktemp -d)}"
mkdir -p "$out"
for s in tri.vert tri.frag alu.vert alu.frag; do
    glslangValidator -V "$here/$s" --vn "${s%%.*}_${s#*.}" -o "$out/${s%%.*}_${s#*.}.h" >/dev/null
done
gcc -O2 -I"$out" -o "$out/vk_offscreen" "$here/vk_offscreen.c" -lvulkan
echo "$out"
