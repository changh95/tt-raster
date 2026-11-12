#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Run parts of the Khronos Vulkan CTS (dEQP-VK) on lavapipe, with or without the Tensix offload, on several workers at once: one worker per
chip of the assigned tray (offload on a chip) or per group of CPU threads (offload on its CPU backend, or lavapipe alone). Every case's
status is collected into results.json, and the failures are compared with a list of expected ones (the ones lavapipe has on its own and the
known differences of the hybrid driver), so that a run says "N new failures" instead of listing everything.

  scripts/cts_farm.py --backend tensix --nodes 0,1,2,3,4,5,6,7            # the default groups, split into sub-groups, one chip per worker
  scripts/cts_farm.py --backend off --workers 8 --out /tmp/cts-lavapipe   # lavapipe alone, 8 workers
  scripts/cts_farm.py --groups 'dEQP-VK.rasterization.*' --no-split --backend reference

The case patterns come from scripts/cts_groups.txt unless --groups is given. Results go to --out (default: a time-stamped directory under
$TMPDIR/tt-raster-cts, outside the repository). Exit status: 0 if every case passed or is an expected failure, 1 if there are new failures,
or cases that never ran (a crashed or timed-out job); 2 for usage errors. Expected failures that pass are reported, not counted. A job that times
out on a chip marks that node as suspect (its worker stops; the job is run again on another worker once) and the run says so: a hung
chip needs `tt-smi -r <its node>` before it is used again.
"""
import argparse
import collections
import concurrent.futures
import datetime
import json
import os
import pathlib
import queue
import re
import signal
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
OK_STATUSES = {"Pass", "NotSupported", "QualityWarning", "CompatibilityWarning", "Waiver"}


def cts_dir() -> pathlib.Path:
    base = pathlib.Path(os.environ.get("TT_RASTER_CTS_DIR", pathlib.Path.home() / "experiments/refs/vk-gl-cts"))
    return base / "build/external/vulkancts/modules/vulkan"


def read_patterns(path: pathlib.Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def read_expected(path: pathlib.Path) -> dict[str, str]:
    """name (or name prefix ending in '*') -> reason, one per line: `case.name  # reason`."""
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            name, _, reason = line.partition("#")
            name = name.strip()
            if name:
                out[name] = reason.strip()
    return out


def is_expected(name: str, expected: dict[str, str]) -> bool:
    for pat in expected:
        if pat == name or (pat.endswith("*") and name.startswith(pat[:-1])):
            return True
    return False


def mustpass_cases() -> list[str]:
    """Every case name of the CTS's default must-pass lists (the CTS case list file names groups only)."""
    base = pathlib.Path(os.environ.get("TT_RASTER_CTS_DIR", pathlib.Path.home() / "experiments/refs/vk-gl-cts"))
    names = []
    for f in sorted((base / "external/vulkancts/mustpass/main/vk-default").glob("*.txt")):
        names += [l.strip() for l in f.read_text().splitlines() if l.startswith("dEQP-VK.")]
    return names


def split_pattern(pattern: str, cases: list[str], max_cases: int) -> list[tuple[str, int]]:
    """`dEQP-VK.a.b.*` -> (pattern, case count) per child group of dEQP-VK.a.b, a child with more than max_cases cases split again.
    Cases of the pattern that are not in the must-pass lists are not in any count but are still run as part of their group."""
    if not pattern.endswith(".*"):
        return [(pattern, 0)]
    prefix = pattern[:-2] + "."
    sub = [c for c in cases if c.startswith(prefix)]
    if not sub:
        return [(pattern, 0)]
    depth = prefix.count(".")
    children = collections.Counter(".".join(c.split(".")[:depth + 1]) for c in sub if c.count(".") > depth)
    leaves = [c for c in sub if c.count(".") == depth]
    out = []
    for name, n in children.items():
        if n > max_cases and any(c.startswith(name + ".") and c.count(".") > depth + 1 for c in sub):
            out += split_pattern(name + ".*", cases, max_cases)
        else:
            out.append((name + ".*", n))
    out += [(c, 1) for c in leaves]
    return out


def parse_out(path: pathlib.Path) -> dict[str, str]:
    """Case name -> status from the CTS's console output (`Test case 'name'..` then `  Pass (..)`)."""
    results = {}
    cur = None
    if not path.exists():
        return results
    for line in path.open(errors="replace"):
        m = re.match(r"Test case '([^']+)'", line)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r"\s+(\w+) \(.*\)\s*$", line)
        if m and cur:
            results[cur] = m.group(1)
            cur = None
    return results


class Farm:
    def __init__(self, args, patterns):
        self.args = args
        self.out = pathlib.Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.jobs: "queue.Queue[tuple[int, str, int]]" = queue.Queue()
        for i, p in enumerate(patterns):
            self.jobs.put((i, p, 0))
        self.lock = threading.Lock()
        self.records = []
        self.suspect_nodes = []
        self.total = len(patterns)
        self.done = 0

    def run_job(self, idx: int, pattern: str, node):
        a = self.args
        log = self.out / f"job{idx:04d}.qpa"
        cmd = [str(ROOT / "scripts/run_cts.sh"), "--backend", a.backend, "--threads", str(a.threads), "--log", str(log)]
        env = dict(os.environ)
        env["TT_RASTER_TIMEOUT"] = str(a.timeout + 60)
        if "TT_METAL_HOME" not in env and (pathlib.Path.home() / "experiments/tt-metal").is_dir():
            env["TT_METAL_HOME"] = str(pathlib.Path.home() / "experiments/tt-metal")
        if node is not None:
            cmd += ["--node", str(node)]
        cmd.append(pattern)
        t0 = time.time()
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, start_new_session=True)
        timed_out = False
        try:
            rc = proc.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)   # the job's own session, never this process
            proc.wait()
            rc = -9
        results = parse_out(pathlib.Path(str(log) + ".out"))
        return {"index": idx, "pattern": pattern, "node": node, "seconds": round(time.time() - t0, 1), "exit_status": rc, "timed_out": timed_out,
                "results": results}

    def worker(self, node):
        while True:
            try:
                idx, pattern, attempt = self.jobs.get_nowait()
            except queue.Empty:
                return
            rec = self.run_job(idx, pattern, node)
            counts = collections.Counter(rec["results"].values())
            tag = "TIMEOUT" if rec["timed_out"] else f"rc={rec['exit_status']}"
            retire = rec["timed_out"] and node is not None   # a node that hung does not take more jobs
            with self.lock:
                print(f"[{self.done + 1}/{self.total}] {pattern}  node={node}  {rec['seconds']:.0f}s  {tag}  "
                      + " ".join(f"{k}={v}" for k, v in sorted(counts.items())), flush=True)
                if retire:
                    self.suspect_nodes.append(node)
                if retire and attempt == 0:
                    self.jobs.put((idx, pattern, 1))   # once more on another worker; this record is dropped
                else:
                    rec["attempt"] = attempt
                    self.records.append(rec)
                    self.done += 1
            if retire:
                return

    def run(self):
        workers = list(self.args.nodes) if self.args.backend == "tensix" else [None] * self.args.workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(workers)) as pool:
            for f in [pool.submit(self.worker, n) for n in workers]:
                f.result()
        # jobs left over because every worker retired
        left = []
        while not self.jobs.empty():
            left.append(self.jobs.get_nowait()[1])
        return left


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["off", "reference", "tensix"], default="tensix")
    ap.add_argument("--nodes", default="0,1,2,3,4,5,6,7", help="tensix: device nodes of the assigned tray, one worker each")
    ap.add_argument("--workers", type=int, default=8, help="off/reference: number of parallel jobs")
    ap.add_argument("--threads", type=int, default=0, help="LP_NUM_THREADS per job (default: the CPU count divided by the workers, at least 4)")
    ap.add_argument("--groups", action="append", help="case pattern (repeatable); default: scripts/cts_groups.txt")
    ap.add_argument("--groups-file", default=str(ROOT / "scripts/cts_groups.txt"))
    ap.add_argument("--no-split", action="store_true", help="one job per pattern instead of one per child group")
    ap.add_argument("--max-cases", type=int, default=3000, help="a group with more cases than this is split into its children")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per job")
    ap.add_argument("--out", default="")
    ap.add_argument("--expected", default=str(ROOT / "scripts/cts_expected_failures.txt"))
    ap.add_argument("--baseline", default="", help="results.json of an earlier run: report what changed against it as well")
    args = ap.parse_args()

    if args.backend == "tensix":
        try:
            args.nodes = [int(x) for x in args.nodes.split(",") if x]
        except ValueError:
            print("cts_farm: --nodes takes a comma-separated list of node numbers", file=sys.stderr)
            return 2
        if not args.nodes or any(n not in range(8) for n in args.nodes):
            print("cts_farm: nodes must be in 0-7 (the assigned tray)", file=sys.stderr)
            return 2
        n_workers = len(args.nodes)
    else:
        n_workers = args.workers
    if args.threads <= 0:
        args.threads = max(4, (os.cpu_count() or 8) // max(1, n_workers))
    if not args.out:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
        args.out = str(pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / "tt-raster-cts" / f"{stamp}-{args.backend}")
    if not (cts_dir() / "deqp-vk").exists():
        print(f"cts_farm: no deqp-vk in {cts_dir()} (see scripts/run_cts.sh for how to build it)", file=sys.stderr)
        return 2

    groups = args.groups or read_patterns(pathlib.Path(args.groups_file))
    jobs = []
    cases = [] if args.no_split else mustpass_cases()
    for g in groups:
        jobs += [(g, 0)] if args.no_split else split_pattern(g, cases, args.max_cases)
    jobs.sort(key=lambda j: -j[1])   # the biggest first, so that the last jobs are small
    patterns = [p for p, _ in jobs]
    print(f"cts_farm: {len(patterns)} jobs on {n_workers} workers ({args.backend}, {args.threads} CPU threads each), results in {args.out}", flush=True)

    farm = Farm(args, patterns)
    t0 = time.time()
    left = farm.run()
    wall = time.time() - t0

    expected = read_expected(pathlib.Path(args.expected))
    status_counts = collections.Counter()
    failed = {}
    for rec in farm.records:
        for name, st in rec["results"].items():
            status_counts[st] += 1
            if st not in OK_STATUSES:
                failed[name] = st
    unexpected = {n: s for n, s in failed.items() if not is_expected(n, expected)}
    # an entry whose reason starts with "offload" fails only with the offload on: not a surprise when it passes on lavapipe alone
    fixed = sorted(n for n, why in expected.items() if not n.endswith("*") and n not in failed
                   and not (args.backend == "off" and why.startswith("offload"))
                   and any(n in r["results"] for r in farm.records))
    problems = [r for r in farm.records if r["timed_out"] or (r["exit_status"] not in (0, 1) and not r["results"])]
    no_result = [r["pattern"] for r in farm.records if not r["results"] and not r["timed_out"] and r["exit_status"] in (0, 1)]

    changes = {}
    if args.baseline:
        base = json.loads(pathlib.Path(args.baseline).read_text())
        was = {n: s for n, s in base.get("failed", {}).items()}
        changes = {"newly_failing": sorted(set(failed) - set(was)), "newly_passing": sorted(set(was) - set(failed))}

    summary = {
        "backend": args.backend, "nodes": args.nodes if args.backend == "tensix" else None, "wall_seconds": round(wall),
        "cases": sum(status_counts.values()), "status": dict(status_counts), "failed": failed, "unexpected_failures": sorted(unexpected),
        "expected_failures_that_passed": fixed, "suspect_nodes": sorted(set(farm.suspect_nodes)), "jobs_not_run": left,
        "jobs": [{k: v for k, v in r.items() if k != "results"} | {"cases": len(r["results"])} for r in sorted(farm.records, key=lambda r: r["index"])],
        "changes_vs_baseline": changes,
    }
    pathlib.Path(args.out, "results.json").write_text(json.dumps(summary, indent=1) + "\n")

    print(f"\n{summary['cases']} cases in {wall / 60:.1f} min: " + "  ".join(f"{k}: {v}" for k, v in sorted(status_counts.items())))
    print(f"failing: {len(failed)}, of which expected: {len(failed) - len(unexpected)}, new: {len(unexpected)}")
    for n in sorted(unexpected)[:40]:
        print(f"  NEW {unexpected[n]}  {n}")
    if len(unexpected) > 40:
        print(f"  ... and {len(unexpected) - 40} more (results.json)")
    if fixed:
        print(f"{len(fixed)} expected failures passed (remove them from {args.expected}):")
        for n in fixed[:20]:
            print(f"  FIXED  {n}")
    if changes:
        print(f"against the baseline: {len(changes['newly_failing'])} newly failing, {len(changes['newly_passing'])} newly passing")
    for r in problems:
        print(f"  JOB {r['pattern']}: {'timed out' if r['timed_out'] else 'exit status %d' % r['exit_status']}, node {r['node']}, "
              f"{len(r['results'])} cases recorded")
    for r in farm.records:
        if not r["results"] and not r["timed_out"]:
            first = ""
            try:
                first = next(iter(pathlib.Path(args.out, f"job{r['index']:04d}.qpa.out").read_text(errors="replace").splitlines()), "")
            except OSError:
                pass
            print(f"  JOB {r['pattern']}: no results (pattern matches nothing, or the run died before its first case): {first[:150]}")
    if farm.suspect_nodes:
        print("nodes whose job timed out (reset each with `tt-smi -r <node>` before the next run): "
              + ", ".join(str(n) for n in sorted(set(farm.suspect_nodes))))
    if left:
        print(f"{len(left)} jobs were never run (every worker retired): " + ", ".join(left[:10]))
    bad = bool(unexpected or problems or left or no_result)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
