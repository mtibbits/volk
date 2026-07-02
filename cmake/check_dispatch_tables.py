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
    0  all dispatch tables consistent with their machine's gates,
       or no machine .c files to check (cross-compile lane)
    1  one or more impls are missing from a machine's dispatch
       (prints each violation with machine name, kernel, impl, gates)
    2  internal error (parse failure, missing inputs)

See mtibbits/volk#58 for context.
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
    # `dispatch[kernel_name] = names` assigns into a dict: if a kernel
    # ever had multiple matching blocks (it doesn't today), only the
    # last would survive. One block per kernel is invariant per the
    # generator's structure -- but if that ever changes, the assertion
    # below catches the regression.
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
        assert kernel_name not in dispatch, (
            f"{path.name}: kernel {kernel_name!r} has multiple impl-name "
            f"arrays; generator structure changed -- update this parser."
        )
        names = set(re.findall(r'"([^"]+)"', names_block))
        dispatch[kernel_name] = names
    return defined, dispatch


def check_machine(path: Path, all_kernels):
    """Return a list of (machine, kernel, impl, gates) violations for this file."""
    defined, dispatch = parse_machine_c(path)
    violations = []
    machine_name = path.stem.replace("volk_machine_", "")

    for kernel in all_kernels:
        actual_impls = dispatch.get(kernel.name)
        if actual_impls is None:
            # Kernel not in this machine .c at all -- should never happen
            # (every machine includes every kernel header per the template).
            continue
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
        epilog="See mtibbits/volk#58 for context.",
    )
    ap.add_argument("--source-dir", required=True, type=Path,
                    help="Volk source root (the dir with gen/, kernels/, ...)")
    ap.add_argument("--build-lib-dir", required=True, type=Path,
                    help="Build output dir containing volk_machine_*.c files")
    args = ap.parse_args()

    gen_dir = args.source_dir / "gen"
    if not gen_dir.is_dir():
        print(f"error: gen dir not found at {gen_dir}", file=sys.stderr)
        sys.exit(2)

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

    machine_files = sorted(args.build_lib_dir.glob("volk_machine_*.c"))
    if not machine_files:
        print(f"warning: no volk_machine_*.c files in {args.build_lib_dir} "
              f"(nothing to check; likely a cross-compile lane that doesn't "
              f"produce x86 machine .c)", file=sys.stderr)
        sys.exit(0)

    all_violations = []
    for mc in machine_files:
        all_violations.extend(check_machine(mc, K.kernels))

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
