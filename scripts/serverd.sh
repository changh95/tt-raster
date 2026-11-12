#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Supervise a device server (core/tools/ttr_serverd) on one chip of the assigned tray: start it, and when its watchdog ends it with status 3
# (a frame ran past --frame-timeout: the chip is hung) reset that chip and start it again. Clients that set TTR_SERVER=<socket> render
# the frames of those minutes on the CPU and find the new server by themselves.
#
#   scripts/serverd.sh <node 0-7> [--socket PATH] [--frame-timeout SECONDS] [--verbose]
#   TTR_SERVER=/tmp/ttraster.sock python3 my_renderer.py          # any number of clients
#
# The reset is `tt-smi -r <node>` (the narrowest scope: this one chip, never a bare `tt-smi -r`), and only if no other process holds the
# device; TT_RASTER_RESET_CMD replaces it (a test uses `true`). At most five restarts in ten minutes, then the script gives up with status 1.
# Exit status 0: the server was stopped on purpose (SIGINT or SIGTERM).
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
node="${1:?usage: serverd.sh <node 0-7> [ttr_serverd arguments]}"
shift
[[ "$node" =~ ^[0-7]$ ]] || { echo "serverd.sh: node '$node' is outside the assigned tray (nodes 0-7)" >&2; exit 2; }
build="${TT_RASTER_BUILD_DIR:-$root/build}"
server="${TT_RASTER_SERVERD:-$build/core/tools/ttr_serverd}"
backend_args=()
[[ "${TT_RASTER_SERVERD_BACKEND:-tensix}" == tensix ]] || backend_args=(--backend "${TT_RASTER_SERVERD_BACKEND}")

child=""
stop() { [[ -n "$child" ]] && kill -TERM "$child" 2>/dev/null; wait "$child" 2>/dev/null; exit 0; }
trap stop INT TERM

restarts=()
while true; do
    if [[ "${TT_RASTER_SERVERD_BACKEND:-tensix}" == tensix ]]; then
        TT_RASTER_TIMEOUT=0 "$root/scripts/run.sh" "$node" "$server" "${backend_args[@]}" "$@" &
    else
        "$server" "${backend_args[@]}" "$@" &
    fi
    child=$!
    wait "$child"
    rc=$?
    child=""
    [[ $rc -eq 0 ]] && exit 0
    now=$(date +%s)
    restarts=("${restarts[@]}" "$now")
    recent=()
    for t in "${restarts[@]}"; do (( now - t < 600 )) && recent+=("$t"); done
    restarts=("${recent[@]}")
    if (( ${#restarts[@]} > 5 )); then
        echo "serverd.sh: the server ended abnormally more than five times in ten minutes; giving up" >&2
        exit 1
    fi
    if [[ $rc -eq 3 ]]; then
        echo "serverd.sh: the server reported a hung device (node $node)" >&2
        if [[ -n "${TT_RASTER_RESET_CMD:-}" ]]; then
            ${TT_RASTER_RESET_CMD} || true
        elif lsof "/dev/tenstorrent/$node" >/dev/null 2>&1; then
            echo "serverd.sh: another process holds /dev/tenstorrent/$node; not resetting it" >&2
            sleep 10
        else
            tt-smi -r "$node" || echo "serverd.sh: tt-smi -r $node failed" >&2
            sleep 5
        fi
    else
        echo "serverd.sh: the server ended with status $rc; starting it again" >&2
        sleep 2
    fi
done
