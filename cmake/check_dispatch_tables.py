#!/usr/bin/env python3
"""
Per-machine impl dispatch-table integrity check.

Reads each generated volk_machine_<name>.c file. For each kernel block,
verifies that every implementation whose LV_HAVE_* source-gates are all
defined by that machine's #define section is present in the machine's
per-kernel _impl_names[] dispatch array.

Catches the bug class introduced (and later fixed) by fork PR
mtibbits/volk#57 cd2d50d: a codegen-pipeline change that silently
filtered impls out of production dispatch tables. ctest does not
catch this -- VOLK's qa_<kernel> tests construct their own
function-pointer table via lib/qa_utils.cc and bypass the per-machine
arrays that production callers use.

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
    text = path.read_text()
    defined = set(re.findall(r'^#define (LV_HAVE_\w+)\s+1', text, re.M))

    # Each kernel block in the generated machine .c has this shape:
    #
    #     "volk_<kernel>",
    #     {"impl0", "impl1", ...},
    #     {<deps masks>},
    #     ...
    #
    # We pin to the *first* {"..."} brace group after the kernel-name
    # string. The deps mask block uses {(1 << LV_X), ...} which has no
    # double-quotes, so the next-string-group anchor is unambiguous.
    block_re = re.compile(
        r'"(volk_\w+)"\s*,\s*\{\s*((?:"[^"]+"\s*,?\s*)+)\}',
        re.M,
    )
    dispatch = {}
    for kernel_name, names_block in block_re.findall(text):
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
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
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
