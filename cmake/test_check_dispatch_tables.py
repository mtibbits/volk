#!/usr/bin/env python3
# Copyright 2026 Free Software Foundation, Inc.
#
# This file is part of VOLK
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#
"""
Informal test suite for cmake/check_dispatch_tables.py (mtibbits/volk#132, #166).

Not wired into ctest -- runnable standalone:
    python3 cmake/test_check_dispatch_tables.py

Drives the real script via subprocess against synthetic source trees
(a fake gen/volk_kernel_defs.py) and synthetic machine .c fixtures.
"""

import contextlib
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

THIS_DIR = Path(__file__).parent
SCRIPT = THIS_DIR / "check_dispatch_tables.py"

KERNEL_DEFS = textwrap.dedent("""\
    class _Impl:
        def __init__(self, name, deps):
            self.name, self.deps = name, deps

    class _Kernel:
        def __init__(self, name, impls):
            self.name, self._impls = name, impls

    kernels = [
        _Kernel("volk_test_add", [_Impl("generic", ["generic"]),
                                  _Impl("a_avx", ["avx"])]),
        _Kernel("volk_test_mul", [_Impl("generic", ["generic"])]),
    ]
""")


def make_source_tree(tmp):
    """Write a fake source root with gen/volk_kernel_defs.py; return it."""
    src = tmp / "src"
    (src / "gen").mkdir(parents=True)
    (src / "gen" / "volk_kernel_defs.py").write_text(KERNEL_DEFS)
    return src


def machine_c(defines, blocks):
    """Render a synthetic machine .c body the script's regexes parse.

    defines: LV_HAVE_* names to '#define <name> 1'
    blocks:  {kernel_name: [impl names]} -> '"<kernel>",\\n{"i0", "i1"},'
    """
    lines = [f"#define {d} 1" for d in defines]
    for kern, impls in blocks.items():
        impl_strs = ", ".join(f'"{i}"' for i in impls)
        lines.append(f'"{kern}",')
        lines.append("{%s}," % impl_strs)
    return "\n".join(lines) + "\n"


def run_check(source_dir, build_lib_dir, machines=None):
    cmd = [sys.executable, str(SCRIPT),
           "--source-dir", str(source_dir),
           "--build-lib-dir", str(build_lib_dir)]
    if machines is not None:
        cmd += ["--machines", machines]
    return subprocess.run(cmd, capture_output=True, text=True)


ALL_GOOD = {
    "volk_test_add": ["generic", "a_avx"],
    "volk_test_mul": ["generic"],
}

# a_avx gated on defined LV_HAVE_AVX but absent from the dispatch array --
# the canonical violation fixture.
MISSING_AVX = dict(ALL_GOOD, volk_test_add=["generic"])


@contextlib.contextmanager
def env():
    """Fixture tree: fake source root + empty build lib dir."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        src = make_source_tree(tmp)
        lib = tmp / "lib"
        lib.mkdir()
        yield src, lib


def test_clean_pass():
    """Both kernels' gated impls present -> exit 0, ok line counts machines."""
    with env() as (src, lib):
        (lib / "volk_machine_avx_64.c").write_text(
            machine_c(["LV_HAVE_GENERIC", "LV_HAVE_AVX"], ALL_GOOD))
        r = run_check(src, lib, machines="avx_64")
        assert r.returncode == 0, r.stderr
        assert "ok (1 machines checked)" in r.stdout, r.stdout


def test_violation_detected():
    """a_avx gated on defined LV_HAVE_AVX but absent from array -> exit 1."""
    with env() as (src, lib):
        (lib / "volk_machine_avx_64.c").write_text(
            machine_c(["LV_HAVE_GENERIC", "LV_HAVE_AVX"], MISSING_AVX))
        r = run_check(src, lib, machines="avx_64")
        assert r.returncode == 1, (r.returncode, r.stderr)
        assert "volk_test_add.a_avx" in r.stderr, r.stderr


def test_drift_defines_fails_closed():
    """#define format drift -> regex parses no macros -> must exit 2 (#132).

    Born-red verified (2026-07-30): unfixed script exited 0 here -- the
    fail-open defect this test pins (analysis/2026-07-30-fail-open-demo.txt).
    """
    with env() as (src, lib):
        # Same info, drifted shape: '#define LV_HAVE_X (1)' -- the regex
        # (which requires a bare '1') parses NO macros from this.
        body = ("#define LV_HAVE_GENERIC (1)\n#define LV_HAVE_AVX (1)\n"
                + machine_c([], ALL_GOOD))
        (lib / "volk_machine_avx_64.c").write_text(body)
        r = run_check(src, lib, machines="avx_64")
        assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
        assert "no '#define LV_HAVE_GENERIC 1' parsed" in r.stderr, r.stderr


def test_drift_blocks_fails_closed():
    """Impl-array format drift -> zero kernel blocks parse -> must exit 2 (#132).

    Born-red verified (2026-07-30): unfixed script exited 0 ("ok") here while
    checking nothing (analysis/2026-07-30-fail-open-demo.txt).
    """
    with env() as (src, lib):
        # Same info, drifted shape: impl arrays use ( ) not { } braces.
        body = machine_c(["LV_HAVE_GENERIC", "LV_HAVE_AVX"], {})
        body += '"volk_test_add",\n("generic", "a_avx"),\n'
        body += '"volk_test_mul",\n("generic"),\n'
        (lib / "volk_machine_avx_64.c").write_text(body)
        r = run_check(src, lib, machines="avx_64")
        assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
        assert "parsed kernel blocks (" in r.stderr, r.stderr


def test_kernel_set_mismatch_fails_closed():
    """File parses but lists only a subset of kernel-defs kernels -> exit 2.

    Guard form is set-equality per the U2 probe (all 9 real machine files
    parse to exactly the 154-kernel defs set --
    analysis/2026-07-30-u2-kernel-set-probe.txt: VERDICT equality HOLDS).
    """
    with env() as (src, lib):
        only_add = {"volk_test_add": ["generic", "a_avx"]}  # volk_test_mul absent
        (lib / "volk_machine_avx_64.c").write_text(
            machine_c(["LV_HAVE_GENERIC", "LV_HAVE_AVX"], only_add))
        r = run_check(src, lib, machines="avx_64")
        assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
        assert "parsed kernel blocks (" in r.stderr, r.stderr
        assert "volk_test_mul" in r.stderr, r.stderr


def test_duplicate_kernel_block_fails_closed():
    """Duplicate kernel block -> exit 2. Set-equality does NOT catch this
    (duplicates leave the key set equal), so the duplicate guard in
    parse_machine_c -- converted from a bare `assert` (strippable under
    python -O, i.e. a live fail-open) to an exit-2 print -- is the only
    cover; this test proves it fires."""
    with env() as (src, lib):
        body = machine_c(["LV_HAVE_GENERIC", "LV_HAVE_AVX"], ALL_GOOD)
        body += machine_c([], {"volk_test_add": ["generic"]})  # 2nd block
        (lib / "volk_machine_avx_64.c").write_text(body)
        r = run_check(src, lib, machines="avx_64")
        assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
        assert "has multiple impl-name arrays" in r.stderr, r.stderr


def test_orphan_ignored_and_count_reflects_active():
    """A stale machine file with a REAL violation, absent from --machines,
    is ignored (exit 0) and the ok-count reflects the active set (#166 AC1+AC2)."""
    with env() as (src, lib):
        (lib / "volk_machine_avx_64.c").write_text(
            machine_c(["LV_HAVE_GENERIC", "LV_HAVE_AVX"], ALL_GOOD))
        # Orphan from a "previous configure": violation-bearing (a_avx gated
        # satisfied, missing from array) -- the exact 2026-06-24 symptom.
        (lib / "volk_machine_stale_64.c").write_text(
            machine_c(["LV_HAVE_GENERIC", "LV_HAVE_AVX"], MISSING_AVX))
        r = run_check(src, lib, machines="avx_64")
        assert r.returncode == 0, (r.returncode, r.stderr)
        assert "ok (1 machines checked)" in r.stdout, r.stdout
        assert "volk_machine_stale_64.c" in r.stderr, r.stderr  # the note names it


def test_empty_machines_list_noop():
    """--machines "" (static-dispatch / no-machine lane) -> warning + exit 0."""
    with env() as (src, lib):
        # Even with a stale violation-bearing file on disk:
        (lib / "volk_machine_stale_64.c").write_text(
            machine_c(["LV_HAVE_GENERIC", "LV_HAVE_AVX"], MISSING_AVX))
        r = run_check(src, lib, machines="")
        assert r.returncode == 0, (r.returncode, r.stderr)
        assert "no machines in the active configure" in r.stderr, r.stderr
        # The empty-list lane must NOT be a silent fail-open: the warning
        # names the on-disk files it is not checking.
        assert "UNCHECKED on disk:" in r.stderr, r.stderr
        assert "volk_machine_stale_64.c" in r.stderr, r.stderr


def test_all_expected_missing_fails_closed():
    """Active list names machines but NO listed file exists -> exit 2.
    (Unified with partial-missing: with cmake supplying the list, zero
    files present means a wiped/wrong build/lib -- the same root cause.
    #166 AC4's exit-0-with-warning lane is the EMPTY-list lane above;
    disposition operator-ratified 2026-07-30.)"""
    with env() as (src, lib):
        r = run_check(src, lib, machines="avx_64;sse_64")
        assert r.returncode == 2, (r.returncode, r.stderr)
        assert "missing from" in r.stderr, r.stderr
        assert "avx_64" in r.stderr and "sse_64" in r.stderr, r.stderr


def test_partial_missing_fails_closed():
    """SOME listed machine files missing -> exit 2 (inconsistent build dir)."""
    with env() as (src, lib):
        (lib / "volk_machine_avx_64.c").write_text(
            machine_c(["LV_HAVE_GENERIC", "LV_HAVE_AVX"], ALL_GOOD))
        r = run_check(src, lib, machines="avx_64;sse_64")
        assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
        assert "sse_64" in r.stderr and "missing from" in r.stderr, r.stderr


def test_multi_machine_clean_pass():
    """Two active machines, both clean -> exit 0, count == 2 (#166 AC3)."""
    with env() as (src, lib):
        (lib / "volk_machine_avx_64.c").write_text(
            machine_c(["LV_HAVE_GENERIC", "LV_HAVE_AVX"], ALL_GOOD))
        (lib / "volk_machine_sse_64.c").write_text(
            machine_c(["LV_HAVE_GENERIC"], ALL_GOOD))
        r = run_check(src, lib, machines="avx_64;sse_64")
        assert r.returncode == 0, (r.returncode, r.stderr)
        assert "ok (2 machines checked)" in r.stdout, r.stdout


def test_active_violation_still_detected():
    """Scoping must not swallow a violation in an ACTIVE machine (the #58 core)."""
    with env() as (src, lib):
        (lib / "volk_machine_avx_64.c").write_text(
            machine_c(["LV_HAVE_GENERIC", "LV_HAVE_AVX"], MISSING_AVX))
        (lib / "volk_machine_stale_64.c").write_text(
            machine_c(["LV_HAVE_GENERIC", "LV_HAVE_AVX"], MISSING_AVX))
        r = run_check(src, lib, machines="avx_64")
        assert r.returncode == 1, (r.returncode, r.stderr)
        assert "avx_64: volk_test_add.a_avx" in r.stderr, r.stderr
        # Discriminating: the orphan's VIOLATION LINE must be absent even
        # though its filename appears in the ignore note.
        assert "stale_64: volk_test_add" not in r.stderr, r.stderr


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
