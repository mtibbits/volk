#!/usr/bin/env python3
"""
Per-machine impl dispatch-table integrity check.

Reads each generated volk_machine_<name>.c file. For each kernel block,
verifies that every implementation whose LV_HAVE_* source-gates are all
defined by that machine's #define section is present in the machine's
per-kernel _impl_names[] dispatch array.

Catches a class of bug VOLK's own ctest cannot detect: a
codegen-pipeline change that silently filters impls out of production
dispatch tables. The qa_<kernel> tests construct their own
function-pointer table via lib/qa_utils.cc and never exercise the
per-machine arrays that production callers use, so ctest 254/254 can
pass even with the dispatch broken.

Invoked by lib/CMakeLists.txt as a custom target made an explicit
dependency of volk_obj.

Exit codes:
    0  all active-configure dispatch tables consistent with their
       machine's gates, or nothing to check (empty --machines: static
       dispatch / no-machine lane)
    1  one or more impls are missing from a machine's dispatch
       (prints each violation with machine name, kernel, impl, gates)
    2  internal error (parse failure/drift, missing inputs, inconsistent
       build dir)

Blind spots (what this check cannot decide): it audits only the machines
named by --machines (the active configure) against today's kernel defs.
Stale machine .c files from a prior configure are deliberately ignored
(named in a stderr note); under an empty --machines nothing is checked
even when machine files sit on disk (warned, UNCHECKED-named); and the
correctness of the --machines list itself is cmake's contract (the
codegen loop's _generated_machines accumulator, derived from
available_machines), not this script's.

See mtibbits/volk#58 for context; #132 (fail-closed parse guards) and
#166 (active-machine scoping) for the hardening.
"""

import argparse
import re
import sys
from pathlib import Path


def parse_machine_c(path: Path):
    """Return (defined_macros: set[str], dispatch: dict[kernel_name, set[impl_name]])."""
    text = path.read_text(encoding="utf-8")
    defined = set(re.findall(r'^#define (LV_HAVE_\w+)\s+1', text, re.M))

    # Each kernel block in the generated machine .c has this shape:
    #
    #     "volk_<kernel>",
    #     {"impl0", "impl1", ...},
    #     {<deps masks>},
    #     ...
    #
    # The match anchor is a kernel-name string immediately followed by a
    # {"..."} brace group. The deps mask block uses {(1 << LV_X), ...}
    # which has no double-quotes, so a string-only brace group can only
    # be the impl-name array.
    #
    # One block per kernel is invariant per the generator's structure;
    # the duplicate guard below fails closed if that ever changes.
    #
    # The inner string-group quantifier is * (not +) so a hypothetical
    # zero-impl kernel still enters the dispatch dict with an empty set,
    # rather than silently dropping out and masking a real regression.
    block_re = re.compile(
        r'"(volk_\w+)"\s*,\s*\{\s*((?:"[^"]+"\s*,?\s*)*)\}',
        re.M,
    )
    dispatch = {}
    for kernel_name, names_block in block_re.findall(text):
        # Fail-closed, not a bare assert (mtibbits/volk#132): a duplicate
        # block leaves the kernel-name SET unchanged, so the set-equality
        # guard in check_machine cannot see it -- this print+exit is the
        # only duplicate cover. (A bare assert would also vanish under an
        # ambient PYTHONOPTIMIZE; cmake invokes with -B, never -O, so that
        # was latent rather than live.)
        if kernel_name in dispatch:
            print(f"error: {path.name}: kernel {kernel_name!r} has multiple "
                  f"impl-name arrays; generator structure changed -- update "
                  f"this parser.", file=sys.stderr)
            sys.exit(2)
        names = set(re.findall(r'"([^"]+)"', names_block))
        dispatch[kernel_name] = names
    return defined, dispatch


def check_machine(path: Path, all_kernels, expected_kernels):
    """Check one machine file; exits 2 on parse drift, else returns a list
    of (machine, kernel, impl, gates) violations.

    expected_kernels is the build-wide {kernel.name} set, computed once in
    main() -- the set-equality guard compares every machine file against
    the same constant.
    """
    defined, dispatch = parse_machine_c(path)
    machine_name = path.stem.replace("volk_machine_", "")

    # Parse-drift guards (mtibbits/volk#132): both regexes fail OPEN -- a
    # template-format change that matches nothing makes every kernel skip
    # and the script print ok while checking nothing. Fail closed instead.
    # Every machine defines LV_HAVE_GENERIC (tmpl/volk_machine_xxx.tmpl.c
    # emits one '#define LV_HAVE_<ARCH> 1' per arch and every machine
    # includes 'generic'), and the generator emits one kernel block per
    # gen/volk_kernel_defs kernel (verified empirically on all 9 machine
    # files of a dev/all-prs configure, Issue-Fork-132 U2 probe).
    if "LV_HAVE_GENERIC" not in defined:
        print(f"error: {path.name}: no '#define LV_HAVE_GENERIC 1' parsed -- "
              f"the #define regex no longer matches the generated format; "
              f"update this parser.", file=sys.stderr)
        sys.exit(2)
    got_kernels = set(dispatch)
    if got_kernels != expected_kernels:
        missing = sorted(expected_kernels - got_kernels)
        extra = sorted(got_kernels - expected_kernels)
        print(f"error: {path.name}: parsed kernel blocks ({len(got_kernels)}) "
              f"!= gen/volk_kernel_defs kernels ({len(expected_kernels)}); "
              f"missing={missing[:5]} extra={extra[:5]} -- either build/lib "
              f"is stale (re-run cmake: kernels/volk/*.h is a "
              f"non-CONFIGURE_DEPENDS glob, so adding/removing a kernel "
              f"header does not retrigger codegen) or the impl-array regex "
              f"no longer matches the generated format; update this parser.",
              file=sys.stderr)
        sys.exit(2)

    violations = []
    for kernel in all_kernels:
        # Direct indexing is safe: the set-equality guard above exits
        # unless every kernel-defs kernel has exactly one parsed block.
        actual_impls = dispatch[kernel.name]
        # Reads kernel._impls (the unfiltered list) rather than calling
        # the public kernel.get_impls(archs) because the latter applies
        # the same intersection-with-archs filter we are auditing here;
        # we want every declared impl, then apply our own gate test.
        for impl in kernel._impls:
            required = {f"LV_HAVE_{dep.upper()}" for dep in impl.deps}
            if required.issubset(defined) and impl.name not in actual_impls:
                violations.append((machine_name, kernel.name, impl.name,
                                   sorted(required)))
    return violations


def main():
    ap = argparse.ArgumentParser(
        description="Per-machine impl dispatch-table integrity check",
        epilog="See mtibbits/volk#58 for context; #132/#166 for the "
               "fail-closed parse guards and active-machine scoping.",
    )
    ap.add_argument("--source-dir", required=True, type=Path,
                    help="Volk source root (the dir with gen/, kernels/, ...)")
    ap.add_argument("--build-lib-dir", required=True, type=Path,
                    help="Build output dir containing volk_machine_*.c files")
    ap.add_argument("--machines", required=True,
                    help="Semicolon-separated machine names whose .c the "
                         "active configure generated (cmake's "
                         "_generated_machines, accumulated by the codegen "
                         "loop). Empty string = nothing to check "
                         "(static-dispatch or no-machine lane).")
    args = ap.parse_args()

    # Cheap wiring sanity (a single stat) kept ahead of the no-op lane so a
    # typo'd --source-dir is caught even where nothing gets checked; the
    # expensive kernel-defs import stays below the early exit.
    gen_dir = args.source_dir / "gen"
    if not gen_dir.is_dir():
        print(f"error: gen dir not found at {gen_dir}", file=sys.stderr)
        sys.exit(2)

    # Scope the check to the ACTIVE configure's machine set (mtibbits/volk#166):
    # cmake passes the codegen loop's machine list instead of this script
    # globbing the build dir, so machine .c files orphaned by a PRIOR configure
    # are ignored (named on stderr) rather than checked against today's
    # kernel/arch definitions. The empty-list no-op comes FIRST -- before the
    # (measured ~150 ms) volk_kernel_defs import it would never use.
    active = [m for m in args.machines.split(";") if m]
    on_disk = {p.name for p in args.build_lib_dir.glob("volk_machine_*.c")}
    if not active:
        extra = (f"; UNCHECKED on disk: {', '.join(sorted(on_disk))}"
                 if on_disk else "")
        print(f"warning: no machines in the active configure (static "
              f"dispatch or no-machine lane); nothing to check{extra}",
              file=sys.stderr)
        sys.exit(0)
    expected_names = {m: f"volk_machine_{m}.c" for m in active}
    missing = sorted(m for m, n in expected_names.items() if n not in on_disk)
    if missing:
        # None-exist and partial-missing are the SAME failure: cmake supplied
        # a non-empty active list, so every listed file must exist -- zero
        # present means a wiped or wrong build/lib, not a lane to wave
        # through.
        print(f"error: {len(missing)} of {len(active)} active machine "
              f"files missing from {args.build_lib_dir}: "
              f"{', '.join(missing)} (codegen incomplete or wrong "
              f"--build-lib-dir)", file=sys.stderr)
        sys.exit(2)
    machine_files = sorted(args.build_lib_dir / n
                           for n in expected_names.values())
    orphans = sorted(on_disk - set(expected_names.values()))
    if orphans:
        print(f"note: ignoring {len(orphans)} volk_machine_*.c not in the "
              f"active configure (stale from a prior configure?): "
              f"{', '.join(orphans)}", file=sys.stderr)

    sys.path.insert(0, str(gen_dir))
    try:
        import volk_kernel_defs as K  # noqa: E402
    except Exception as e:
        print(f"error: failed to import volk_kernel_defs: {e}", file=sys.stderr)
        sys.exit(2)

    # Defensive assertions: this script depends on gen/volk_kernel_defs.py
    # internals (kernels list, kernel._impls, impl.deps, impl.name). If any
    # of those go away, fail fast with a clear message rather than later
    # with a cryptic AttributeError mid-iteration.
    if not hasattr(K, "kernels") or not K.kernels:
        print("error: gen/volk_kernel_defs exposes no `kernels` attribute "
              "(or it is empty)", file=sys.stderr)
        sys.exit(2)
    _k = K.kernels[0]
    for attr in ("_impls", "name"):
        if not hasattr(_k, attr):
            print(f"error: gen/volk_kernel_defs kernel objects lack `{attr}` "
                  f"-- generator API changed; update this check.",
                  file=sys.stderr)
            sys.exit(2)
    if _k._impls and not all(hasattr(i, "deps") and hasattr(i, "name")
                             for i in _k._impls):
        print("error: gen/volk_kernel_defs impl objects lack `deps` or `name` "
              "-- generator API changed; update this check.", file=sys.stderr)
        sys.exit(2)

    expected_kernels = {k.name for k in K.kernels}
    all_violations = []
    for mc in machine_files:
        all_violations.extend(check_machine(mc, K.kernels, expected_kernels))

    if all_violations:
        print("DISPATCH TABLE INTEGRITY CHECK FAILED", file=sys.stderr)
        print("", file=sys.stderr)
        print("Impls compiled by the machine's #define LV_HAVE_* set are", file=sys.stderr)
        print("missing from the machine's per-kernel _impl_names[] dispatch", file=sys.stderr)
        print("array. The kernel will silently fall back to a lower impl at", file=sys.stderr)
        print("runtime even though the better impl is present in the .so.", file=sys.stderr)
        print("", file=sys.stderr)
        print("To diagnose, look at:", file=sys.stderr)
        print("  - gen/volk_kernel_defs.py    (impl-deps extraction)", file=sys.stderr)
        print("  - gen/volk_machine_defs.py   (per-machine arch set)", file=sys.stderr)
        print("  - tmpl/volk_machine_xxx.tmpl.c (dispatch-array codegen)", file=sys.stderr)
        print("  - gen/archs.xml, gen/machines.xml (arch/machine grammar)", file=sys.stderr)
        print("", file=sys.stderr)
        print("See https://github.com/mtibbits/volk/issues/58 for context.", file=sys.stderr)
        print("", file=sys.stderr)
        print("Violations:", file=sys.stderr)
        for machine, kernel, impl, gates in all_violations:
            gates_str = " && ".join(gates)
            print(f"  {machine}: {kernel}.{impl} (gated on {gates_str})",
                  file=sys.stderr)
        sys.exit(1)

    print(f"dispatch-table integrity: ok ({len(machine_files)} machines checked)")


if __name__ == "__main__":
    main()
