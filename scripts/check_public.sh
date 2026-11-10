#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# This repository is public. Block commits that would publish machine-identifying
# data: home-directory paths, PCIe bus ids, device or board ids, this host's name,
# and runtime output directories.
#
# Install once per clone:  ln -sf ../../scripts/check_public.sh .git/hooks/pre-commit
# Override for a reviewed false positive: SKIP_PUBLIC_CHECK=1 git commit ...
set -euo pipefail

[[ "${SKIP_PUBLIC_CHECK:-0}" == "1" ]] && exit 0

fail=0

# 1. Paths that must never be committed (runtime output from tt-metal, logs, build trees).
bad_paths="$(git diff --cached --name-only --diff-filter=AM | grep -E '^(generated|build|build_Release)/|\.log$|\.ndjson$' || true)"
if [[ -n "$bad_paths" ]]; then
    echo "check_public: refusing to commit runtime or build output:" >&2
    echo "$bad_paths" | sed 's/^/  /' >&2
    fail=1
fi

# 2. Machine-identifying strings in added lines.
added="$(git diff --cached -U0 --no-color | grep -E '^\+[^+]' || true)"
pat='/home/[A-Za-z0-9_-]+|0000:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]|(asic|board|unique)_?id|blackhole-[0-9A-F]{16}'
if grep -Eiq "$pat" <<<"$added"; then
    echo "check_public: added lines match a machine-identifying pattern:" >&2
    grep -Ein "$pat" <<<"$added" | head -10 | sed 's/^/  /' >&2
    fail=1
fi
host="$(hostname)"
if [[ -n "$host" ]] && grep -Fiq -- "$host" <<<"$added"; then
    echo "check_public: added lines contain this machine's hostname" >&2
    fail=1
fi

exit $fail
