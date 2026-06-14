#!/usr/bin/env python3
# Copyright 2026 Free Software Foundation, Inc.
#
# This file is part of VOLK
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run the assembled kernel-correctness harness (epic #85) across the full
kernel catalog and emit one combined per-kernel x per-impl triage CSV.

Each (kernel, mode) pair runs in its OWN process via volk_test_correctness's
argv[1] filter, so a kernel that aborts (e.g. at tiny vlens) is recorded as an
`abort` finding row instead of killing the snapshot. This is the one-shot
snapshot consumed by the per-kernel fix campaign (mtibbits/volk#92).

NOTE on reproducibility: qa test data is re-randomized every run
(load_random_data seeds from std::random_device), so near-tolerance findings
-- reduction/accumulator kernels diverging marginally past tol -- fluctuate
across runs in their exact failing vlens and can appear/disappear at the
margin. The stable signal of a snapshot is the failing-kernel set and the
gross findings, not the exact small-vlen lists.

Usage:
  python3 tools/run_harness_triage.py --build-dir build --out triage.csv \
      [--modes sweep,canary,immutable,misaligned] [--jobs N] [--timeout 900] \
      [--only volk_32f_tanh_32f,...]
"""
import argparse
import concurrent.futures as cf
import csv
import glob
import os
import signal
import subprocess
import sys
import zlib

MODES = {
    "sweep": {},  # default correctness sweep (ref|impl per registration)
    "canary": {"HARNESS_CANARY": "1"},
    "immutable": {"HARNESS_IMMUTABLE": "1"},
    "misaligned": {"HARNESS_MISALIGNED": "1"},
}
HEADER = ["kernel", "impl", "mode", "result", "failed_vlens", "max_err"]


def kernel_names(repo_root):
    # Deliberately a local glob: gen/volk_kernel_defs.py is the canonical
    # enumerator but parses every kernel header at import time; the harness
    # binary's argv filter makes a stray non-kernel name harmless anyway.
    pat = os.path.join(repo_root, "kernels", "volk", "volk_*.h")
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(pat))


def signal_name(signum):
    try:
        return signal.Signals(signum).name
    except ValueError:  # e.g. real-time signals have no enum entry
        return str(signum)


def read_report_rows(report):
    rows = []
    if os.path.exists(report):
        with open(report, newline="") as f:
            for row in csv.reader(f):
                # Skip the version-marker/comment line (#135: '# volk-harness-report
                # v2'), the header, and any truncated trailing line (a crash can
                # cut the final stdio flush mid-row).
                if (
                    row
                    and not row[0].startswith("#")
                    and row[0] != "kernel"
                    and len(row) == len(HEADER)
                ):
                    rows.append(row)
    return rows


def run_one(binary, libdir, kernel, mode, timeout, tmpdir, base_seed=None):
    report = os.path.join(tmpdir, f"{kernel}.{mode}.csv")
    # A process killed before fopen would otherwise merge a PRIOR invocation's
    # rows for this (kernel, mode).
    try:
        os.unlink(report)
    except FileNotFoundError:
        pass
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = libdir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env["HARNESS_REPORT"] = report
    # #134: derive a distinct deterministic seed per (kernel, mode) child;
    # zlib.crc32 is stable across runs/platforms (Python's hash() is salted).
    if base_seed is not None:
        child = (base_seed * 1000003 ^ zlib.crc32(f"{kernel}.{mode}".encode())) & 0x7FFFFFFF
        env["HARNESS_SEED"] = str(child)
    else:
        env.pop("HARNESS_SEED", None)  # ambient export must not half-seed a run
    env.update(MODES[mode])
    if mode == "misaligned":
        # the misaligned mode's own SIGSEGV handler must win over ASan's
        env.setdefault(
            "ASAN_OPTIONS",
            "handle_segv=0:handle_sigbus=0:handle_sigill=0:allow_user_segv_handler=1",
        )
    try:
        proc = subprocess.run(
            [binary, kernel],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        # Keep whatever was written before the kill, consistent with rc<0.
        rows = read_report_rows(report)
        rows.append([kernel, "-", mode, "abort", "timeout", ""])
        return rows
    rows = read_report_rows(report)
    # rc<0 => died on a signal; rows written before the crash are kept and the
    # abort itself becomes a finding row (tiny-vlen aborts are findings).
    if rc < 0:
        rows.append([kernel, "-", mode, "abort", f"signal={signal_name(-rc)}", ""])
    elif rc not in (0, 1):  # 0=clean, 1=FAILs recorded in rows; else abnormal
        rows.append([kernel, "-", mode, "abort", f"exit={rc}", ""])
    elif rc == 1 and not any(r[3] in ("FAIL", "partial") for r in rows):
        # Fail closed: the binary reported failure but no finding row was
        # captured (e.g. its HARNESS_REPORT fopen failed and it degraded to
        # human output) -- record the inconsistency rather than skipping.
        rows.append([kernel, "-", mode, "abort", "exit=1-without-finding-rows", ""])
    if not rows:
        rows.append([kernel, "-", mode, "skip", "no-qa-entry", ""])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--out", default="triage.csv")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--only", default="")
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="pin qa data deterministically; per-(kernel,mode) child seeds are "
        "derived from this value (falls back to HARNESS_SEED in the env)",
    )
    args = ap.parse_args()
    base_seed = args.seed
    if base_seed is None and os.environ.get("HARNESS_SEED"):
        try:
            base_seed = int(os.environ["HARNESS_SEED"])
        except ValueError:
            sys.exit(f"HARNESS_SEED must be an integer, got: {os.environ['HARNESS_SEED']!r}")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    libdir = os.path.join(os.path.abspath(args.build_dir), "lib")
    binary = os.path.join(libdir, "volk_test_correctness")
    if not os.path.exists(binary):
        sys.exit(f"not found: {binary} (build with -DENABLE_TESTING=ON first)")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        sys.exit(f"unknown mode(s): {unknown}; valid: {list(MODES)}")
    kernels = kernel_names(repo_root)
    if args.only:
        keep = {k.strip() for k in args.only.split(",")}
        kernels = [k for k in kernels if k in keep]

    tmpdir = os.path.join(os.path.abspath(args.build_dir), ".harness-triage")
    os.makedirs(tmpdir, exist_ok=True)
    jobs = [(k, m) for k in kernels for m in modes]
    print(
        f"# {len(kernels)} kernels x {modes} = {len(jobs)} runs, "
        f"{args.jobs} workers",
        file=sys.stderr,
    )
    all_rows = []
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [
            ex.submit(run_one, binary, libdir, k, m, args.timeout, tmpdir, base_seed)
            for (k, m) in jobs
        ]
        done = 0
        for fut, (k, m) in zip(futs, jobs):
            # One job's unexpected exception must not kill the snapshot —
            # that is the whole point of per-process isolation.
            try:
                all_rows.extend(fut.result())
            except Exception as e:  # noqa: BLE001 -- synthesize a finding row
                all_rows.append([k, "-", m, "abort", f"runner-error: {e}", ""])
            done += 1
            if done % 50 == 0:
                print(f"# {done}/{len(jobs)}", file=sys.stderr)
    all_rows.sort(key=lambda r: (r[0], r[2], r[1]))
    with open(args.out, "w", newline="") as f:
        # #135: self-describing version marker first (matches the in-binary child
        # reports; read_report_rows skips '#'-prefixed lines), then the header.
        f.write("# volk-harness-report v2\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(HEADER)
        w.writerows(all_rows)
    n_find = sum(1 for r in all_rows if r[3] in ("FAIL", "abort"))
    seed_desc = base_seed if base_seed is not None else "none (re-randomized)"
    print(f"# seed={seed_desc}", file=sys.stderr)
    print(
        f"# wrote {args.out}: {len(all_rows)} rows, "
        f"{n_find} FAIL/abort finding rows",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
