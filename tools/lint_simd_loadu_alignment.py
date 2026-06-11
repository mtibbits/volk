#!/usr/bin/env python3
"""Lint: unaligned SIMD loads of under-aligned local scratch arrays.

Catches the class of bug fixed in mtibbits/volk#101: a SIMD *unaligned* load
intrinsic (`_mmN_loadu_*`) applied to a function-local array that is NOT declared
`__VOLK_ATTR_ALIGNED(>=N/8)`. GCC at -O2/-O3 may lower the unaligned intrinsic to an
*aligned* move (vmovaps/vmovdqa) because it over-assumes the local's alignment; if the
array lacks an explicit alignment attribute, that assumption is not guaranteed under all
stack layouts (e.g. AddressSanitizer's frame instrumentation), and the aligned move
#GP-faults -> SIGSEGV.

This is a STATIC net for the *pattern*. It is deterministic (no dependence on stack
layout / optimization / sanitizer luck), so it catches present and future instances when
run in CI. It is intentionally conservative: it only flags loads whose operand is a
local array declared in the same function. Loads of function-parameter pointers (whose
alignment the compiler cannot assume) are not flagged.

Known false-negatives (heuristic, not a parser -- the safe direction; it never blocks a
correct PR, it may miss a risky one): the operand reached through a pointer variable that
aliases a local; an array whose size is a macro or hex literal (`buf[SIZE]`, `buf[0x20]`)
rather than a decimal literal; comma multi-declarators; and a declaration whose `[N]` and
its `;`/`=` are not on the same source line. Extend the regexes if these arise.

Exit status: 0 = clean, 1 = findings, 2 = usage error.

Usage:
    tools/lint_simd_loadu_alignment.py [paths ...]      # default: kernels/volk/*.h
"""
import re
import sys
import glob

# SIMD unaligned-load intrinsics -> required byte alignment for the aligned form.
# 512-bit -> 64, 256-bit -> 32, 128-bit -> 16.
LOADU_RE = re.compile(
    r'_mm(?P<bits>512|256|)_loadu_(?:ps|pd|si128|si256|si512|epi\d+|epu\d+)\s*\('
    r'\s*'
    r'(?:\(\s*[A-Za-z_][\w\s]*\*+\s*\)\s*)?'   # optional C cast, e.g. (float*) / (const __m512i *)
    r'&?\s*'                                     # optional address-of
    r'(?P<operand>[A-Za-z_]\w*)'
)
BITS_TO_ALIGN = {'512': 64, '256': 32, '': 16}  # '' == _mm_loadu_* (128-bit)

# A local array declaration, optional inline alignment attribute on the same line.
# Tolerates a trailing initializer (`[8] = { 0 };`) so initialized scratch arrays
# are still recorded.
ARRAY_DECL_RE = re.compile(
    r'^(?P<indent>\s*)'
    r'(?:(?P<attr_inline>__VOLK_ATTR_ALIGNED\(\s*(?P<inline_n>\d+)\s*\))\s+)?'
    r'(?:static\s+|const\s+|volatile\s+)*'
    r'[A-Za-z_]\w*(?:\s+[A-Za-z_]\w*)*\s+'      # type words
    # name[N] terminated by ';' (bare decl) or '=' (initializer, which may run
    # onto later lines). The required `type name` prefix keeps element
    # assignments like `data[3] = x;` from matching as declarations.
    r'(?P<name>[A-Za-z_]\w*)\s*\[\s*\d+\s*\]\s*(?:;|=)'
)
ATTR_ONLY_RE = re.compile(r'^\s*__VOLK_ATTR_ALIGNED\(\s*(?P<n>\d+)\s*\)\s*$')
FUNC_START_RE = re.compile(r'static\s+inline\b')


def code_only(lines):
    """Parallel list of lines with // comments, /* */ block comments, and string/char
    literal contents removed -- so brace counting and matching ignore text in comments
    and string literals. Line count and positions are preserved.
    """
    out = []
    in_block = False
    for line in lines:
        res = []
        i, n = 0, len(line)
        while i < n:
            c = line[i]
            if in_block:
                if c == '*' and i + 1 < n and line[i + 1] == '/':
                    in_block = False
                    i += 2
                else:
                    i += 1
                continue
            if c == '/' and i + 1 < n and line[i + 1] == '/':
                break  # line comment: drop the rest
            if c == '/' and i + 1 < n and line[i + 1] == '*':
                in_block = True
                i += 2
                continue
            if c in '"\'':
                quote = c
                i += 1
                while i < n:  # skip the literal body (kept out of the code text)
                    if line[i] == '\\' and i + 1 < n:
                        i += 2
                        continue
                    if line[i] == quote:
                        i += 1
                        break
                    i += 1
                continue
            res.append(c)
            i += 1
        out.append(''.join(res))
    return out


def split_functions(code_lines):
    """Yield (start_idx, end_idx_exclusive) per top-level function, using comment/string-
    stripped lines for `static inline` detection and brace balancing.
    """
    i, n = 0, len(code_lines)
    while i < n:
        if FUNC_START_RE.search(code_lines[i]):
            j = i
            while j < n and '{' not in code_lines[j]:
                j += 1
            if j >= n:
                break
            depth = 0
            k = j
            started = False
            while k < n:
                depth += code_lines[k].count('{') - code_lines[k].count('}')
                if '{' in code_lines[k]:
                    started = True
                if started and depth <= 0:
                    break
                k += 1
            yield (i, k + 1)
            i = k + 1
        else:
            i += 1


def array_alignments(body_lines):
    """Map local array name -> declared alignment in bytes (0 if none)."""
    aligns = {}
    for idx, raw in enumerate(body_lines):
        m = ARRAY_DECL_RE.match(raw)
        if not m:
            continue
        name = m.group('name')
        if m.group('inline_n'):
            aligns[name] = int(m.group('inline_n'))
            continue
        # attribute may sit on the immediately-preceding non-blank line
        a = 0
        p = idx - 1
        while p >= 0 and body_lines[p].strip() == '':
            p -= 1
        if p >= 0:
            am = ATTR_ONLY_RE.match(body_lines[p])
            if am:
                a = int(am.group('n'))
        aligns[name] = a
    return aligns


def lint_file(path):
    findings = []
    with open(path, 'r', errors='replace') as f:
        lines = f.read().split('\n')
    cl = code_only(lines)  # match on comment/string-stripped text, report original
    for start, end in split_functions(cl):
        body_cl = cl[start:end]
        aligns = array_alignments(body_cl)
        if not aligns:
            continue
        for off, raw_cl in enumerate(body_cl):
            for m in LOADU_RE.finditer(raw_cl):
                operand = m.group('operand')
                if operand not in aligns:
                    continue  # not a local array (e.g. a parameter pointer) -> safe
                need = BITS_TO_ALIGN[m.group('bits')]
                have = aligns[operand]
                if have < need:
                    findings.append((
                        path, start + off + 1, operand, need, have,
                        lines[start + off].strip(),
                    ))
    return findings


def main(argv):
    paths = argv[1:] or sorted(glob.glob('kernels/volk/*.h'))
    if not paths:
        sys.stderr.write('no input files (run from the volk source root)\n')
        return 2
    all_findings = []
    for p in paths:
        all_findings.extend(lint_file(p))
    if not all_findings:
        print(f'OK: no under-aligned SIMD loadu-of-local-array in {len(paths)} file(s)')
        return 0
    print(f'FOUND {len(all_findings)} risky SIMD load(s) of under-aligned local arrays:\n')
    for path, line, name, need, have, src in all_findings:
        cur = f'{have}B' if have else 'unaligned'
        print(f'{path}:{line}: `{name}` loaded at {need*8}-bit width but declared {cur} '
              f'(needs __VOLK_ATTR_ALIGNED({need}))')
        print(f'    {src}')
    print('\nFix: add __VOLK_ATTR_ALIGNED(<width>) to each flagged local array declaration.')
    return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
