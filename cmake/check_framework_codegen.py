#!/usr/bin/env python3
# Copyright 2026 Free Software Foundation, Inc.
#
# This file is part of VOLK
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#
"""
Codegen-equivalence test harness for volk fusion-framework impls.

Reads a JSON manifest of (kernel, ISA, alignment, impl_a, impl_b, criterion)
tuples. For each tuple, disassembles the named symbol in each impl's compiled
.o file, extracts that function's whole instruction body (from its `<symbol>:`
header to the next symbol header), and applies the declared equivalence
criterion:

    byte_identical  -- raw instruction-byte sequences must match
    within_noise    -- mnemonic sequences match and operand classes match;
                       register reassignment and immediate-value differences
                       are tolerated, instruction order is NOT.

Failure halts the build with the offending tuple, the disassembly diff, and
the criterion evaluated.

Invoked by lib/CMakeLists.txt as a dependency of the volk shared-lib target
so it runs after volk_obj compiles but before linking.

Exit codes:
    0  all declared tuples pass (or manifest is empty)
    1  one or more tuples fail (per-tuple diff printed to stderr)
    2  internal error (missing manifest, missing .o, symbol not found, parse)

See gnuradio/volk issue mtibbits/volk#78 for context. Mirrors
the dispatch-table integrity check (mtibbits/volk#58) for shape and reporting.

DESIGN NOTE (whole-function comparison): an earlier design bracketed just the
inner loop with source-level asm labels. That was abandoned: inserting a label
symbol mid-function makes its address a basic-block boundary the optimizer must
respect, which perturbs the surrounding codegen (an extra test, register-
allocation churn) -- i.e. instrumenting the kernel for the harness would change
the kernel that ships. Instead the harness compares the WHOLE function body
between two impls. This needs no source markers, leaves production codegen
untouched, and is the stronger contract for the framework use case: a framework
instantiation that reproduces a hand-written impl produces the same *function*,
not merely the same loop. The impl must be emitted standalone (its address taken
for the dispatch table), which volk's kernels always are.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Manifest parsing
# ---------------------------------------------------------------------------

def parse_manifest(path: Path) -> list:
    if not path.is_file():
        print(f"error: manifest not found at {path}", file=sys.stderr)
        sys.exit(2)
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"error: manifest is not valid JSON: {e}", file=sys.stderr)
        sys.exit(2)
    tuples = data.get("tuples", [])
    for t in tuples:
        for required in ("kernel", "isa", "alignment", "impl_a", "impl_b",
                         "criterion"):
            if required not in t:
                print(f"error: manifest tuple missing required field "
                      f"{required!r}: {t}", file=sys.stderr)
                sys.exit(2)
        for side in ("impl_a", "impl_b"):
            for f in ("symbol", "machine_o"):
                if f not in t[side]:
                    print(f"error: manifest tuple {side} missing {f!r}: {t}",
                          file=sys.stderr)
                    sys.exit(2)
        if t["criterion"] not in ("byte_identical", "within_noise"):
            print(f"error: unknown criterion {t['criterion']!r} (must be "
                  f"byte_identical or within_noise)", file=sys.stderr)
            sys.exit(2)
    return tuples


def tuple_id(t: dict) -> str:
    return f"{t['kernel']}.{t['isa']}.{t['alignment']}"


# ---------------------------------------------------------------------------
# Disassembly extraction
# ---------------------------------------------------------------------------

# A label/function header line in llvm-objdump (and GNU objdump) disassembly:
#     0000000000000007 <fwk_codegen_my_start>:
# The address column is optional in some objdump variants, so it is tolerated.
_label_line = re.compile(r'^\s*(?:[0-9a-fA-F]+\s+)?<([^>+]+)>:\s*$')

# A disassembled instruction line. Two whitespace styles across objdump
# variants; both have: addr ':' raw-bytes <tab-or-spaces> mnemonic [operands].
#     18: c5 fc 28 0c 06   <tab> vmovaps <tab> (%rsi,%rax), %ymm1   (llvm)
#     6600:\tc5 fc 28 0c 06 \tvmovaps (%rsi,%rax,1),%ymm1           (gnu)
_disasm_line = re.compile(
    r'^\s*([0-9a-fA-F]+):\s+'                 # address
    r'((?:[0-9a-fA-F]{2}[ \t]+)+?)'           # raw bytes (non-greedy)
    r'[ \t]+'                                  # separator before mnemonic
    r'([^\s]+)'                                # mnemonic
    r'(?:[ \t]+(.*))?$'                        # optional operands
)

# A continuation line: GNU objdump wraps an instruction longer than 7 bytes
# onto a second line carrying only the trailing bytes, no mnemonic, e.g.
#     1e:	00 00
# Those bytes belong to the previous instruction and are stitched onto it.
_continuation_line = re.compile(
    r'^\s*[0-9a-fA-F]+:\s+((?:[0-9a-fA-F]{2}[ \t]*)+)$'
)


def resolve_object(build_lib_dir: Path, machine_o: str) -> Path:
    """Resolve a manifest machine_o entry to an actual object-file path.

    machine_o may be:
      - an absolute path (used as-is),
      - a path relative to build_lib_dir that exists (used directly), or
      - a bare filename, in which case the build tree is searched recursively
        (different OBJECT libraries land their .o files in different
        CMakeFiles/<target>.dir/ subdirs, so a recursive search avoids
        hardcoding fragile per-target paths).
    """
    p = Path(machine_o)
    if p.is_absolute() and p.is_file():
        return p
    direct = build_lib_dir / machine_o
    if direct.is_file():
        return direct
    matches = sorted(build_lib_dir.rglob(Path(machine_o).name))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise RuntimeError(
            f"object file {machine_o!r} not found under {build_lib_dir}")
    raise RuntimeError(
        f"object file {machine_o!r} is ambiguous under {build_lib_dir} "
        f"({len(matches)} matches): {[str(m) for m in matches]}")


def _disassemble(o_file: Path, objdump: str, symbol: str = None) -> str:
    if not o_file.is_file():
        raise RuntimeError(f"object file not found: {o_file}")
    # NOTE: deliberately NOT using --symbolize-operands. It makes llvm-objdump
    # synthesize <L0>/<L1> branch labels that collide with the _label_line
    # parser. Plain --disassemble shows symbol headers as clean <name>: lines.
    cmd = [objdump, "--disassemble"]
    # Efficiency: llvm-objdump can disassemble a single symbol, skipping the
    # hundreds of other kernels in a machine .o. GNU objdump has no equivalent,
    # so only use it for llvm-objdump and fall back to whole-object otherwise.
    if symbol and "llvm" in Path(objdump).name:
        cmd.append(f"--disassemble-symbols={symbol}")
    cmd.append(str(o_file))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{objdump} failed on {o_file}: {result.stderr}")
    return result.stdout


def extract_function_body(o_file: Path, symbol: str,
                          objdump: str = "llvm-objdump") -> list:
    """Return [{address, bytes, mnemonic, operands}, ...] for the whole body of
    `symbol` in the disassembly of o_file: every instruction line from the
    `<symbol>:` header up to (exclusive) the next symbol header or a blank line
    that ends the function's block. Raises RuntimeError if the symbol is absent.
    """
    text = _disassemble(o_file, objdump, symbol=symbol)
    in_body = False
    saw_symbol = False
    instrs = []

    for line in text.splitlines():
        m = _label_line.match(line)
        if m:
            label = m.group(1)
            if label == symbol:
                in_body = True
                saw_symbol = True
                continue
            # A different symbol header ends the body we were collecting.
            if in_body:
                in_body = False
            continue
        if in_body:
            if not line.strip():
                # Blank line terminates the function's disassembly block.
                in_body = False
                continue
            mi = _disasm_line.match(line)
            if not mi:
                ci = _continuation_line.match(line)
                if ci and instrs:
                    # GNU objdump wrapped a long instruction: append the
                    # trailing bytes to the instruction they belong to.
                    instrs[-1]["bytes"].extend(ci.group(1).split())
                    continue
                # Any other in-body line we cannot parse must be loud, not
                # silently dropped: a dropped instruction would weaken the
                # comparison without anyone noticing.
                raise RuntimeError(
                    f"unparsable disassembly line for {symbol!r} in {o_file}: "
                    f"{line!r}")
            addr_hex, bytes_str, mnemonic, operands = mi.groups()
            instrs.append({
                "address": addr_hex,
                "bytes": bytes_str.split(),
                "mnemonic": mnemonic,
                "operands": (operands or "").strip(),
            })

    if not saw_symbol:
        raise RuntimeError(
            f"symbol {symbol!r} not found in disassembly of {o_file}. "
            f"The impl must be emitted standalone (its address taken for the "
            f"dispatch table) for its symbol to appear in the object file.")
    # Strip trailing NOP-family instructions: padding emitted after the
    # function's final control-flow terminator to align the NEXT function. It
    # belongs to no function and varies with inter-function layout, so it is
    # not part of this function's codegen. Internal alignment nops (e.g. before
    # a hot loop) are mid-body and are preserved.
    while instrs and instrs[-1]["mnemonic"].startswith("nop"):
        instrs.pop()
    if not instrs:
        raise RuntimeError(
            f"symbol {symbol!r} found in {o_file} but its body is empty")
    return instrs


# ---------------------------------------------------------------------------
# Equivalence criteria
# ---------------------------------------------------------------------------

def _operand_class(op: str) -> str:
    """Normalize an operand string to its class for within_noise comparison:
    strip specific register identifiers, immediate values, and disassembler
    symbol annotations, keeping the operand's structural shape.

    Coverage is x86_64-validated (SIMD + all GPR width forms). It is NOT a
    complete model of every architecture's operand syntax; within_noise on
    other targets needs its register table extended here.
    """
    # Drop disassembler symbol annotations first: branch/RIP-relative operands
    # render as "0x18 <SYMBOL+0x18>", and SYMBOL is the function's OWN name,
    # which differs between the two impls being compared (e.g. _a_avx vs
    # _a_avx_ref). The numeric displacement already encodes the distance, so
    # the annotation is pure noise that would otherwise force a false failure.
    op = re.sub(r'<[^>]*>', '', op)
    # SIMD vector registers: %xmm0..31 / %ymm.. / %zmm..  -> %xmmN etc.
    op = re.sub(r'%(x|y|z)mm\d+', r'%\1mmN', op)
    # Extended GPRs r8-r15 with optional d/w/b width suffix -> %rN.
    op = re.sub(r'%r1[0-5][dwb]?\b', '%rN', op)
    op = re.sub(r'%r[89][dwb]?\b', '%rN', op)
    # 64-/32-bit ABI-named GPRs. (%rip is left intact: rip-relative addressing
    # is semantically distinct and should not collapse to a GPR class.)
    op = re.sub(r'%r(?:ax|bx|cx|dx|si|di|sp|bp)\b', '%rregN', op)
    op = re.sub(r'%e(?:ax|bx|cx|dx|si|di|sp|bp)\b', '%eregN', op)
    # 16-bit GPRs (%ax..%bp) and 8-bit GPRs (%al/%ah/%bl.., %sil/%dil/%spl/%bpl).
    op = re.sub(r'%(?:ax|bx|cx|dx|si|di|sp|bp)\b', '%wregN', op)
    op = re.sub(r'%(?:[abcd][lh]|sil|dil|spl|bpl)\b', '%bregN', op)
    # Immediates: hex, signed/unsigned decimal.
    op = re.sub(r'\$0x[0-9a-fA-F]+', '$imm', op)
    op = re.sub(r'\$-?\d+', '$imm', op)
    # Bare hex displacements in memory operands: 0x10(%reg) -> IMMx(%reg).
    op = re.sub(r'\b0x[0-9a-fA-F]+', 'IMMx', op)
    # GNU objdump renders branch targets as a bare hex address (no 0x); after
    # the <...> strip above, a leading bare-hex token may remain -> normalize.
    op = re.sub(r'^\s*[0-9a-fA-F]+\s*$', 'IMMx', op)
    # Canonicalize whitespace so e.g. a stripped annotation leaving a trailing
    # space cannot make two otherwise-equal operand classes compare unequal.
    return re.sub(r'\s+', ' ', op).strip()


def compare_byte_identical(a: list, b: list):
    if len(a) != len(b):
        return False, (f"instruction count differs: {len(a)} vs {len(b)}\n"
                       f"  a: {[i['mnemonic'] for i in a]}\n"
                       f"  b: {[i['mnemonic'] for i in b]}")
    diffs = []
    for i, (ia, ib) in enumerate(zip(a, b)):
        if ia["bytes"] != ib["bytes"]:
            diffs.append(
                f"  [{i}] bytes differ: "
                f"{' '.join(ia['bytes'])} ({ia['mnemonic']}) vs "
                f"{' '.join(ib['bytes'])} ({ib['mnemonic']})")
    if diffs:
        return False, "byte_identical comparison FAILED:\n" + "\n".join(diffs)
    return True, ""


def compare_within_noise(a: list, b: list):
    if len(a) != len(b):
        return False, (f"instruction count differs: {len(a)} vs {len(b)}\n"
                       f"  a: {[i['mnemonic'] for i in a]}\n"
                       f"  b: {[i['mnemonic'] for i in b]}")
    diffs = []
    for i, (ia, ib) in enumerate(zip(a, b)):
        if ia["mnemonic"] != ib["mnemonic"]:
            diffs.append(
                f"  [{i}] mnemonic differs: {ia['mnemonic']} vs {ib['mnemonic']}")
            continue
        ca, cb = _operand_class(ia["operands"]), _operand_class(ib["operands"])
        if ca != cb:
            diffs.append(
                f"  [{i}] {ia['mnemonic']} operand class differs: "
                f"{ca!r} vs {cb!r}")
    if diffs:
        return False, "within_noise comparison FAILED:\n" + "\n".join(diffs)
    return True, ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Codegen-equivalence test harness for volk fusion impls",
        epilog="See gnuradio/volk issue mtibbits/volk#78 for context.",
    )
    ap.add_argument("--manifest", required=True, type=Path,
                    help="JSON manifest of equivalence tuples")
    ap.add_argument("--build-lib-dir", type=Path,
                    help="Dir containing the compiled .o files "
                         "(required unless --list-only)")
    ap.add_argument("--objdump", default="llvm-objdump",
                    help="Disassembler binary (default: llvm-objdump; "
                         "GNU objdump also works)")
    ap.add_argument("--list-only", action="store_true",
                    help="List declared tuples and exit (no disassembly)")
    args = ap.parse_args()

    tuples = parse_manifest(args.manifest)

    if args.list_only:
        for t in tuples:
            print(tuple_id(t))
        sys.exit(0)

    if not args.build_lib_dir:
        print("error: --build-lib-dir required unless --list-only",
              file=sys.stderr)
        sys.exit(2)

    if not tuples:
        print("codegen-equivalence: ok (0 tuples declared)")
        sys.exit(0)

    failures = []
    errors = []
    for t in tuples:
        try:
            a_o = resolve_object(args.build_lib_dir, t["impl_a"]["machine_o"])
            b_o = resolve_object(args.build_lib_dir, t["impl_b"]["machine_o"])
            a_instrs = extract_function_body(
                a_o, t["impl_a"]["symbol"], objdump=args.objdump)
            b_instrs = extract_function_body(
                b_o, t["impl_b"]["symbol"], objdump=args.objdump)
        except RuntimeError as e:
            errors.append((tuple_id(t), str(e)))
            continue

        if t["criterion"] == "byte_identical":
            ok, diff = compare_byte_identical(a_instrs, b_instrs)
        else:
            ok, diff = compare_within_noise(a_instrs, b_instrs)
        if not ok:
            failures.append((tuple_id(t), diff))

    if errors:
        print("CODEGEN-EQUIVALENCE CHECK ERROR", file=sys.stderr)
        print("", file=sys.stderr)
        for tid, msg in errors:
            print(f"--- {tid} ---", file=sys.stderr)
            print(f"  {msg}", file=sys.stderr)
            print("", file=sys.stderr)
        sys.exit(2)

    if failures:
        print("CODEGEN-EQUIVALENCE CHECK FAILED", file=sys.stderr)
        print("", file=sys.stderr)
        print("One or more declared (kernel, ISA, alignment) tuples failed",
              file=sys.stderr)
        print("their codegen-equivalence criterion. A framework instantiation",
              file=sys.stderr)
        print("no longer emits the machine code its reference impl emits.",
              file=sys.stderr)
        print("", file=sys.stderr)
        print("See https://github.com/mtibbits/volk/issues/78 for context.",
              file=sys.stderr)
        print("", file=sys.stderr)
        for tid, diff in failures:
            print(f"--- {tid} ---", file=sys.stderr)
            print(diff, file=sys.stderr)
            print("", file=sys.stderr)
        sys.exit(1)

    print(f"codegen-equivalence: ok ({len(tuples)} tuples checked)")


if __name__ == "__main__":
    main()
