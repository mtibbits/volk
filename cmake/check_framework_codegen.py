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

Cross-compiler robustness (mtibbits/volk#145): benign codegen noise that differs
by compiler is normalized away -- trailing alignment NOPs of any encoding and
trailing `data16` padding are stripped (they pad the next function and belong to
no function), and a symbol the compiler inlined rather than emitting standalone
is skipped-with-warning since there is nothing to compare (symbol names are
first mapped to the object format's labels -- Mach-O prepends an underscore
to C symbols, mtibbits/volk#225) -- exit 0 only while real coverage remains:
a run where EVERY declared tuple skipped, or a kernel
whose declared tuples ALL skipped, is a zero-coverage failure (exit 1;
mtibbits/volk#165). A genuine post-normalization divergence still fails
(exit 1); a genuinely unparsable non-padding line still errors (exit 2).

Per-tuple opt-in `require_standalone` (bool, default off) flips that inlined-away
outcome: for a tuple whose dispatch relies on the impl existing as a real,
separately-dispatchable symbol, "inlined away" is a regression, so the checker
hard-fails it (exit 1) instead of skip-with-warning. Orthogonal to
`require_mnemonic`, which asserts which instructions a present symbol contains.

Exit codes:
    0  all declared tuples pass; skips warn on stderr while real coverage
       remains (an empty manifest -- zero tuples declared -- is also ok)
    1  one or more tuples fail their criterion (per-tuple diff on stderr),
       or zero coverage: every declared tuple skipped, or some kernel's
       declared tuples all skipped (mtibbits/volk#165)
    2  internal error: missing manifest/.o, ambiguous .o, unparsable
       non-padding line, or empty function body

See mtibbits/volk#78 for context. Mirrors
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
from collections import Counter
from pathlib import Path


class SymbolNotEmittedError(RuntimeError):
    """The compared label is absent from the object file's disassembly --
    the compiler inlined the impl rather than emitting it standalone, or the
    label genuinely is not there. There is nothing to compare, so main()
    skips the tuple with a warning rather than failing the build -- subject
    to the aggregate zero-coverage guard: a run (or a single kernel) whose
    declared tuples ALL skip fails loudly instead (mtibbits/volk#165).
    Distinct from the other RuntimeError cases (missing/ambiguous .o,
    unparsable line) which remain hard errors (mtibbits/volk#145).

    NOTE (mtibbits/volk#225): "not found" once had a second, non-inlining
    cause -- Mach-O's leading-underscore C mangling made the unprefixed
    lookup miss labels that WERE present (measured on a Linux cross-compile
    probe: an address-taken static-inline impl compiled for
    x86_64-apple-macos was emitted standalone as _volk_32f_x2_add_32f_a_avx).
    object_symbol_name() now maps C name -> object-file label per the file's
    magic, and the error text reports similar labels seen, so a naming
    mismatch is loud instead of masquerading as inlining.
    """


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
        # Optional opt-in field: a regex the compared function must match at
        # least one instruction mnemonic against (scalar-fallback guard).
        if "require_mnemonic" in t:
            if not isinstance(t["require_mnemonic"], str):
                print(f"error: require_mnemonic must be a string regex: {t}",
                      file=sys.stderr)
                sys.exit(2)
            try:
                re.compile(t["require_mnemonic"])
            except re.error as e:
                print(f"error: require_mnemonic is not a valid regex "
                      f"({t['require_mnemonic']!r}): {e}", file=sys.stderr)
                sys.exit(2)
        # Optional opt-in field (orthogonal to require_mnemonic): a boolean that,
        # when true, turns an inlined-away impl from skip-with-warning into a
        # hard failure -- the impl must exist as a standalone dispatchable symbol.
        if "require_standalone" in t:
            if not isinstance(t["require_standalone"], bool):
                print(f"error: require_standalone must be a boolean: {t}",
                      file=sys.stderr)
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

# A NOP/padding instruction line that _disasm_line above fails to capture across
# objdump variants. _disasm_line needs >=2 whitespace chars between the last byte
# and the mnemonic (its byte-group [ \t]+ plus a separate mandatory [ \t]+
# separator), but the long multi-byte alignment NOPs that clang-15+/gcc-11/14
# emit as trailing pad render with a single tab, e.g.
#     21656: 66 2e 0f 1f 84 00 00 00 00 00\tnopw\t%cs:(%rax,%rax)
# This fallback drops the separate separator and anchors the padding mnemonic
# directly after the byte column, so it matches the single-tab form. It is tried
# ONLY when _disasm_line fails, so the passing path is unchanged; it matches only
# the NOP family and the lone data16 prefix, both of which the trailing-strip
# below removes when they are inter-function padding (mtibbits/volk#145).
_padding_line = re.compile(
    r'^\s*([0-9a-fA-F]+):\s+'
    r'((?:[0-9a-fA-F]{2}[ \t]+)+?)'
    r'(nop[a-z]*|data16)\b'
    r'(?:[ \t]+(.*))?$'
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


def _require_object_file(o_file: Path):
    """Single owner of the missing-object diagnostic (exit-2 class), shared by
    every seam that touches the file so the message cannot drift between
    copies."""
    if not o_file.is_file():
        raise RuntimeError(f"object file not found: {o_file}")


# Mach-O object-file magics (as the first 4 bytes appear on disk): thin
# 64-/32-bit little-endian, and fat/universal (big-endian on disk) 32/64.
_MACHO_MAGICS = {
    b"\xcf\xfa\xed\xfe",  # MH_MAGIC_64
    b"\xce\xfa\xed\xfe",  # MH_MAGIC
    b"\xca\xfe\xba\xbe",  # FAT_MAGIC
    b"\xca\xfe\xba\xbf",  # FAT_MAGIC_64
}


def object_symbol_name(o_file: Path, symbol: str) -> str:
    """Map a C-level symbol name to the label it carries in this object file.

    Mach-O (Darwin ABI) prepends an underscore to every C symbol, so the
    disassembly label for volk_32f_x2_add_32f_a_avx is
    _volk_32f_x2_add_32f_a_avx; ELF and COFF-x64 use the C name unchanged.
    Keyed on the object file's magic bytes rather than the host platform so a
    cross-compiled Mach-O object resolves correctly anywhere
    (mtibbits/volk#225 -- on macOS the unprefixed lookup made the bootstrap
    tuple skip as 'symbol not found', zero coverage under the #165 guard).

    Blind-spot shape (what this helper CANNOT decide): detection is a
    positive Mach-O check on 4 magics (thin LE 32/64 + fat 32/64).
    Anything else -- ELF, COFF (which has no single 4-byte magic), or a
    format outside the support set -- keeps the C name unchanged, which is
    correct for every non-Mach-O lane CI builds today. Big-endian thin
    Mach-O (ppc-era) is deliberately not recognized: no such target exists
    in the CI matrix or support set. If a misclassification ever happens,
    SymbolNotEmittedError's similar-labels report is the loud signal (the
    underscored twin shows up there).
    """
    _require_object_file(o_file)
    with o_file.open("rb") as f:
        magic = f.read(4)
    if magic in _MACHO_MAGICS:
        return "_" + symbol
    return symbol


def _uses_symbol_filter(objdump: str) -> bool:
    """True when _disassemble will pass --disassemble-symbols for this tool
    (llvm-objdump only; GNU objdump has no equivalent). Shared predicate so
    extract_function_body can know its fast-path output was FILTERED -- a
    filtered disassembly of a wrong label contains zero labels, which would
    starve the similar-labels diagnostic (mtibbits/volk#225)."""
    return "llvm" in Path(objdump).name


def _disassemble(o_file: Path, objdump: str, symbol: str = None) -> str:
    _require_object_file(o_file)
    # NOTE: deliberately NOT using --symbolize-operands. It makes llvm-objdump
    # synthesize <L0>/<L1> branch labels that collide with the _label_line
    # parser. Plain --disassemble shows symbol headers as clean <name>: lines.
    cmd = [objdump, "--disassemble"]
    # Efficiency: llvm-objdump can disassemble a single symbol, skipping the
    # hundreds of other kernels in a machine .o. GNU objdump has no equivalent,
    # so only use it for llvm-objdump and fall back to whole-object otherwise.
    if symbol and _uses_symbol_filter(objdump):
        cmd.append(f"--disassemble-symbols={symbol}")
    cmd.append(str(o_file))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{objdump} failed on {o_file}: {result.stderr}")
    return result.stdout


def _is_trailing_padding(instr: dict) -> bool:
    """True if `instr`, sitting after the function's final control-flow
    terminator, is inter-function alignment padding (belongs to no function).
    Covers every x86 inter-function pad filler observed across the CI matrix:

      - the NOP family (`nop`, `nopw`, `nopl`, `nopq`, ...),
      - the lone `data16` prefix some disassemblers print for a multi-byte NOP
        (GNU objdump renders it `data16 cs nopw ...`),
      - `int3` (0xcc) gap fill,
      - the legacy `xchg %ax,%ax` (0x66 0x90) two-byte NOP -- but ONLY the
        self-exchange idiom (same src/dst); a real register `xchg` is never
        stripped.

    Mid-body alignment NOPs are NOT stripped -- only the trailing run.

    Invariant: this set MUST stay a superset of the mnemonics `_padding_line`
    accepts, so any pad the fallback absorbs is guaranteed to be stripped here
    and never reaches the comparison.
    """
    m = instr["mnemonic"]
    if m.startswith("nop") or m == "data16" or m == "int3":
        return True
    if m == "xchg":
        # Strip only `xchg %reg,%reg` (the NOP idiom), never a real exchange.
        ops = [o for o in instr["operands"].replace(" ", "").split(",") if o]
        return len(ops) == 2 and ops[0] == ops[1]
    return False


def _instr_from_match(m) -> dict:
    """Build an instruction record from a _disasm_line OR _padding_line match.
    Both regexes capture the same four groups in the same order -- address,
    bytes, mnemonic, operands -- and this is the single source of truth for the
    record shape the comparison consumes.
    """
    addr_hex, bytes_str, mnemonic, operands = m.groups()
    return {
        "address": addr_hex,
        "bytes": bytes_str.split(),
        "mnemonic": mnemonic,
        "operands": (operands or "").strip(),
    }


def extract_function_body(o_file: Path, symbol: str,
                          objdump: str = "llvm-objdump") -> list:
    """Disassemble `symbol` in o_file and return its whole-body instruction list.
    Maps the C name to the object file's label first (Mach-O prepends an
    underscore, mtibbits/volk#225), then thin wrapper over
    extract_function_body_from_text (see it for the parse contract); kept so
    callers and tests can pass a file path + objdump. On a not-found label
    after a FILTERED fast-path disassembly, retries unfiltered so the
    similar-labels report is computed from the whole object, not from
    filter-emptied output. The retry is behavior, not just diagnostics: if
    the symbol filter fails to match a label that whole-object disassembly
    does render (an objdump version/symbol-table quirk), the retry recovers
    the extraction instead of skipping the tuple.
    """
    label = object_symbol_name(o_file, symbol)
    text = _disassemble(o_file, objdump, symbol=label)
    try:
        return extract_function_body_from_text(text, label,
                                               source_label=str(o_file))
    except SymbolNotEmittedError:
        if not _uses_symbol_filter(objdump):
            raise
        # The fast path FILTERED the disassembly to the requested label; when
        # that label is wrong the output contains zero labels, which would
        # starve the similar-labels diagnostic ("no similar labels among 0
        # seen"). Retry unfiltered -- error path only, so the extra
        # disassembly costs nothing on a passing build -- so the raised
        # error can report what the object file actually contains
        # (mtibbits/volk#225).
        text = _disassemble(o_file, objdump)
        return extract_function_body_from_text(text, label,
                                               source_label=str(o_file))


def extract_function_body_from_text(text: str, label: str,
                                    source_label: str = "<disassembly>") -> list:
    """Return [{address, bytes, mnemonic, operands}, ...] for the whole body of
    the object-file label `label` in `text` (on Mach-O, callers pass the
    underscore-prefixed label object_symbol_name() produced -- the underscore
    is the harness's mapping, not part of the C name): every instruction line
    from the `<label>:` header up to (exclusive) the next symbol header or a
    blank line that ends the block. Trailing inter-function padding
    (alignment NOPs / data16) is stripped.

    Raises SymbolNotEmittedError if the label is absent (inlined, not emitted
    standalone) and RuntimeError if an in-body line is unparsable and not
    recognizable padding, or if the body is empty.
    """
    in_body = False
    saw_symbol = False
    instrs = []
    labels_seen = []

    for line in text.splitlines():
        m = _label_line.match(line)
        if m:
            label_name = m.group(1)
            labels_seen.append(label_name)
            if label_name == label:
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
                # Lenient fallback for NOP/data16 padding lines whose single-tab
                # byte/mnemonic spacing _disasm_line cannot match. Padding is
                # removed by the trailing-strip below, so absorbing it here is
                # safe and keeps the build green on clang-15+/gcc-11/14.
                pm = _padding_line.match(line)
                if pm:
                    instrs.append(_instr_from_match(pm))
                    continue
                # Any other in-body line we cannot parse must be loud, not
                # silently dropped: a dropped instruction would weaken the
                # comparison without anyone noticing.
                raise RuntimeError(
                    f"unparsable disassembly line for {label!r} in "
                    f"{source_label}: {line!r}")
            instrs.append(_instr_from_match(mi))

    if not saw_symbol:
        near = [seen for seen in labels_seen
                if label in seen or seen in label]
        if near:
            # Closest-length first so the exact underscore twin cannot be
            # truncated out of the window; say when truncation happened.
            near = sorted(near, key=lambda seen: (abs(len(seen) - len(label)),
                                                  seen))
            shown = near[:8]
            more = f" (+{len(near) - len(shown)} more)" if len(near) > 8 else ""
            hint = f"similar labels present: {shown}{more}"
        else:
            hint = f"no similar labels among {len(labels_seen)} seen"
        raise SymbolNotEmittedError(
            f"symbol label {label!r} not found in disassembly of "
            f"{source_label}. The impl must be emitted standalone (its "
            f"address taken for the dispatch table) for its label to appear "
            f"in the object file; {hint}.")
    # Strip trailing padding (NOP family + data16): bytes emitted after the
    # function's final control-flow terminator to align the NEXT function. They
    # belong to no function and vary with inter-function layout, so they are not
    # part of this function's codegen. Internal alignment nops (e.g. before a
    # hot loop) are mid-body and are preserved.
    while instrs and _is_trailing_padding(instrs[-1]):
        instrs.pop()
    if not instrs:
        raise RuntimeError(
            f"symbol {label!r} found in {source_label} but its body is empty")
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


def check_require_mnemonic(body: list, pattern: str):
    """Assert at least one instruction in `body` has a mnemonic matching the
    regex `pattern`. Returns (ok, diff). Equivalence to a reference does not
    prove an impl emits the intended ISA -- a scalar fallback could be
    byte-identical to a scalar reference and pass; this guards against that. On
    failure, diff names the pattern and the sorted set of mnemonics present.
    """
    rx = re.compile(pattern)
    if any(rx.search(instr["mnemonic"]) for instr in body):
        return True, ""
    seen = sorted({instr["mnemonic"] for instr in body})
    return False, (f"required-mnemonic assertion FAILED: no instruction "
                   f"mnemonic matches /{pattern}/.\n  mnemonics present: {seen}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Codegen-equivalence test harness for volk fusion impls",
        epilog="See mtibbits/volk#78 for context.",
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
    skipped = []
    skipped_by_kernel = Counter()
    for t in tuples:
        try:
            a_o = resolve_object(args.build_lib_dir, t["impl_a"]["machine_o"])
            b_o = resolve_object(args.build_lib_dir, t["impl_b"]["machine_o"])
            a_instrs = extract_function_body(
                a_o, t["impl_a"]["symbol"], objdump=args.objdump)
            b_instrs = extract_function_body(
                b_o, t["impl_b"]["symbol"], objdump=args.objdump)
        except SymbolNotEmittedError as e:
            # Opt-in (require_standalone): for tuples whose dispatch relies on the
            # impl existing as a real, separately-dispatchable symbol, "inlined
            # away" is a regression, not an acceptable outcome -- hard-fail it.
            if t.get("require_standalone"):
                msg = ("require_standalone assertion FAILED: implementation "
                       "was not emitted as a standalone dispatchable symbol. "
                       f"{e}")
                failures.append((tuple_id(t), msg))
                continue
            # Default: compiler inlined the impl rather than emitting it
            # standalone: nothing to compare, so skip with a loud warning
            # instead of failing the build. A present-but-divergent body is
            # unaffected -- it still fails below.
            print(f"codegen-equivalence: WARNING: skipping {tuple_id(t)}: {e}",
                  file=sys.stderr)
            skipped.append(tuple_id(t))
            skipped_by_kernel[t["kernel"]] += 1
            continue
        except RuntimeError as e:
            errors.append((tuple_id(t), str(e)))
            continue

        if t["criterion"] == "byte_identical":
            ok, diff = compare_byte_identical(a_instrs, b_instrs)
        else:
            ok, diff = compare_within_noise(a_instrs, b_instrs)
        # Optional required-mnemonic assertion: only meaningful once the pair is
        # otherwise equivalent. Checked against each impl independently.
        if ok and "require_mnemonic" in t:
            for label, instrs in (("impl_a", a_instrs), ("impl_b", b_instrs)):
                mok, mdiff = check_require_mnemonic(instrs, t["require_mnemonic"])
                if not mok:
                    ok = False
                    diff = f"[{label}] {mdiff}"
                    break
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

    # Zero-coverage guard (mtibbits/volk#165): a skip is legitimate per-tuple,
    # but declared-yet-unverified coverage must not report success. The global
    # all-skip case gets its own headline because "the check verified nothing
    # at all" is a different operator situation than one kernel losing
    # coverage. (A require_standalone tuple that was inlined away lands in
    # `failures` -- exited above -- never in `skipped`, so it cannot skew this
    # accounting.)
    checked = len(tuples) - len(skipped)
    declared_by_kernel = Counter(t["kernel"] for t in tuples)
    uncovered = sorted(k for k, n in declared_by_kernel.items()
                       if skipped_by_kernel[k] == n)
    # `checked == 0` is tested explicitly rather than relying on it implying
    # `uncovered` non-empty: that implication is a property of the loop's
    # current buckets (every tuple checks or skips), not of this guard, and a
    # future non-skip consumption path must not let a zero-checked run print
    # ok again.
    if uncovered or checked == 0:
        if checked == 0:
            headline = "zero coverage"
            lead = (f"All {len(tuples)} declared tuples were skipped (impl "
                    "not emitted standalone),\nso nothing was verified. A "
                    "toolchain change that inlines every compared\nimpl "
                    "would otherwise turn this check into a silent no-op.")
            shown = skipped
        else:
            headline = ("zero codegen coverage for kernel(s): "
                        + ", ".join(uncovered))
            lead = ("Every declared tuple for the named kernel(s) was "
                    "skipped (impl not emitted\nstandalone), leaving them "
                    "unverified while the rest of the run passed.")
            # Name only the uncovered kernels' skips: a covered kernel's
            # incidental skip must not be presented as removable. Derived from
            # the skip list itself (not kernel membership) so the label stays
            # truthful if the uncovered predicate is ever loosened below 100%.
            shown = [tid for tid in skipped
                     if tid.split(".", 1)[0] in uncovered]
        print(f"CODEGEN-EQUIVALENCE CHECK FAILED: {headline}",
              file=sys.stderr)
        print("", file=sys.stderr)
        print(lead, file=sys.stderr)
        print("", file=sys.stderr)
        print("Fix the build so the symbols are emitted standalone (see the "
              "README's\nrequire-standalone notes). Removing the affected "
              "tuples from the manifest\nsilences the check instead of "
              "fixing it -- a deliberate de-scoping that\nneeds review, not "
              "an equivalent outcome. See mtibbits/volk#165.",
              file=sys.stderr)
        if checked == 0:
            print("Note: an emptied manifest prints ok (0 tuples declared); "
                  "that green means\nzero coverage was accepted, not that "
                  "codegen was verified.", file=sys.stderr)
        print(f"  skipped: {shown}", file=sys.stderr)
        sys.exit(1)

    extra = (f", {len(skipped)} skipped -- not emitted standalone: {skipped}"
             if skipped else "")
    print(f"codegen-equivalence: ok ({checked} tuples checked{extra})")


if __name__ == "__main__":
    main()
