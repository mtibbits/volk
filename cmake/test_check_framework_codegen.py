#!/usr/bin/env python3
# Copyright 2026 Free Software Foundation, Inc.
#
# This file is part of VOLK
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#
"""
Informal test suite for cmake/check_framework_codegen.py (mtibbits/volk#78).

Not wired into ctest -- documents the script's internal contracts and is
runnable standalone:  python3 cmake/test_check_framework_codegen.py

Exercises manifest parsing, whole-function disassembly extraction, and the
byte_identical / within_noise comparison criteria. The extraction tests
assemble tiny fixtures with the system toolchain, so they require `cc` and
a disassembler (llvm-objdump or objdump) on PATH.
"""

import contextlib
import json
import subprocess
import sys
import tempfile
from importlib import util
from pathlib import Path

THIS_DIR = Path(__file__).parent
SCRIPT = THIS_DIR / "check_framework_codegen.py"


def _load_module():
    spec = util.spec_from_file_location("cfc", str(SCRIPT))
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _which_objdump():
    for cand in ("llvm-objdump", "llvm-objdump-18", "objdump"):
        if subprocess.run(["which", cand], capture_output=True).returncode == 0:
            return cand
    raise RuntimeError("no disassembler found (llvm-objdump / objdump)")


@contextlib.contextmanager
def _patched_disassemble(mod, fake):
    """Swap mod._disassemble for `fake` and guarantee restoration -- a leaked
    monkeypatch would silently poison every later extraction test."""
    real = mod._disassemble
    mod._disassemble = fake
    try:
        yield
    finally:
        mod._disassemble = real


def test_manifest_parse():
    """Script accepts a JSON manifest and lists declared tuples via --list-only."""
    manifest = {
        "tuples": [
            {
                "kernel": "volk_32f_x2_add_32f",
                "isa": "avx",
                "alignment": "a",
                "impl_a": {
                    "symbol": "volk_32f_x2_add_32f_a_avx",
                    "machine_o": "volk_machine_avx_64_mmx_orc.c.o",
                },
                "impl_b": {
                    "symbol": "volk_32f_x2_add_32f_a_avx_ref",
                    "machine_o": "codegen_bootstrap_ref.c.o",
                },
                "criterion": "byte_identical",
            }
        ]
    }
    manifest_path = Path("/tmp/cge_test_manifest.json")
    manifest_path.write_text(json.dumps(manifest))
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--manifest", str(manifest_path), "--list-only"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "volk_32f_x2_add_32f.avx.a" in result.stdout, result.stdout


def test_empty_manifest_ok():
    """An empty manifest is a valid zero-false-positives configuration."""
    manifest_path = Path("/tmp/cge_empty_manifest.json")
    manifest_path.write_text('{"tuples":[]}')
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--manifest", str(manifest_path), "--build-lib-dir", "/tmp"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "0 tuples" in result.stdout, result.stdout


def test_bad_criterion_rejected():
    """A manifest tuple with an unknown criterion fails with exit 2."""
    manifest = {"tuples": [{
        "kernel": "k", "isa": "avx", "alignment": "a",
        "impl_a": {"symbol": "s", "machine_o": "m.o"},
        "impl_b": {"symbol": "s2", "machine_o": "m2.o"},
        "criterion": "bogus",
    }]}
    manifest_path = Path("/tmp/cge_bad_manifest.json")
    manifest_path.write_text(json.dumps(manifest))
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--manifest", str(manifest_path), "--list-only"],
        capture_output=True, text=True,
    )
    assert result.returncode == 2, (result.returncode, result.stdout)
    assert "bogus" in result.stderr


def test_function_body_extraction():
    """extract_function_body collects the whole function body (prologue through
    epilogue) for the named symbol, and stops at the next symbol."""
    objdump = _which_objdump()
    fixture_c = Path("/tmp/cge_fixture.c")
    fixture_c.write_text(
        "#include <immintrin.h>\n"
        "void fixture_fn(float* c, const float* a, const float* b, unsigned n){\n"
        "    unsigned e = n / 8;\n"
        "    for (unsigned i = 0; i < e; i++) {\n"
        "        __m256 x = _mm256_load_ps(a + i*8);\n"
        "        __m256 y = _mm256_load_ps(b + i*8);\n"
        "        _mm256_store_ps(c + i*8, _mm256_add_ps(x, y));\n"
        "    }\n"
        "}\n"
        "void other_fn(void){}\n"
    )
    fixture_o = Path("/tmp/cge_fixture.o")
    subprocess.run(["cc", "-O3", "-mavx", "-c", str(fixture_c),
                    "-o", str(fixture_o)], check=True)
    mod = _load_module()
    instrs = mod.extract_function_body(fixture_o, "fixture_fn", objdump=objdump)
    mnemonics = [i["mnemonic"] for i in instrs]
    # Whole-body extraction includes prologue (endbr64) and the SIMD loop.
    assert any(m.startswith("vadd") for m in mnemonics), mnemonics
    # The next function's body must NOT bleed in: fixture_fn ends before
    # other_fn. A 1-instruction `ret`-only other_fn would be the giveaway, so
    # assert the extracted body is the add loop, not a degenerate stub.
    assert len(instrs) > 3, mnemonics


def test_byte_identical_same():
    mod = _load_module()
    a = [{"address": "10", "bytes": ["c5", "f4", "58", "04", "02"],
          "mnemonic": "vaddps", "operands": "(%rdx,%rax), %ymm1, %ymm0"}]
    ok, diff = mod.compare_byte_identical(a, a)
    assert ok, diff


def test_byte_identical_diff():
    mod = _load_module()
    a = [{"address": "10", "bytes": ["c5", "f4", "58", "04", "02"],
          "mnemonic": "vaddps", "operands": "(%rdx,%rax), %ymm1, %ymm0"}]
    b = [{"address": "10", "bytes": ["c5", "f4", "59", "04", "02"],
          "mnemonic": "vmulps", "operands": "(%rdx,%rax), %ymm1, %ymm0"}]
    ok, diff = mod.compare_byte_identical(a, b)
    assert not ok
    assert "vaddps" in diff and "vmulps" in diff, diff


def test_within_noise_register_swap():
    """Register reassignment within the same operand class passes within_noise."""
    mod = _load_module()
    a = [{"address": "10", "bytes": ["aa"], "mnemonic": "vaddps",
          "operands": "(%rsi,%rax), %ymm0, %ymm1"}]
    b = [{"address": "10", "bytes": ["bb"], "mnemonic": "vaddps",
          "operands": "(%rdx,%rcx), %ymm2, %ymm3"}]
    ok, diff = mod.compare_within_noise(a, b)
    assert ok, diff


def test_within_noise_mnemonic_diff_fails():
    mod = _load_module()
    a = [{"address": "10", "bytes": ["aa"], "mnemonic": "vaddps",
          "operands": "%ymm0, %ymm1, %ymm2"}]
    b = [{"address": "10", "bytes": ["bb"], "mnemonic": "vmulps",
          "operands": "%ymm0, %ymm1, %ymm2"}]
    ok, diff = mod.compare_within_noise(a, b)
    assert not ok
    assert "vaddps" in diff and "vmulps" in diff, diff


def test_within_noise_real_two_symbols():
    """End-to-end within_noise over two real, differently-named compilations of
    the same loop. Exercises the branch-target annotation path (<symbol+off>):
    the two functions have different symbol names, so their internal jump
    operands annotate differently -- within_noise must tolerate that. This is
    the regression guard for the branch-operand false-failure bug."""
    objdump = _which_objdump()
    src = ("#include <immintrin.h>\n"
           "void NAME(float* c, const float* a, const float* b, unsigned n){\n"
           "    unsigned e = n / 8;\n"
           "    for (unsigned i = 0; i < e; i++) {\n"
           "        __m256 x = _mm256_loadu_ps(a + i*8);\n"
           "        __m256 y = _mm256_loadu_ps(b + i*8);\n"
           "        _mm256_storeu_ps(c + i*8, _mm256_add_ps(x, y));\n"
           "    }\n"
           "    for (unsigned i = e*8; i < n; i++) c[i] = a[i] + b[i];\n"
           "}\n")
    a_c = Path("/tmp/cge_wn_a.c"); a_c.write_text(src.replace("NAME", "wn_fn_a"))
    b_c = Path("/tmp/cge_wn_b.c"); b_c.write_text(src.replace("NAME", "wn_fn_b"))
    a_o = Path("/tmp/cge_wn_a.o"); b_o = Path("/tmp/cge_wn_b.o")
    for s, o in ((a_c, a_o), (b_c, b_o)):
        subprocess.run(["cc", "-O3", "-mavx", "-c", str(s), "-o", str(o)],
                       check=True)
    mod = _load_module()
    ia = mod.extract_function_body(a_o, "wn_fn_a", objdump=objdump)
    ib = mod.extract_function_body(b_o, "wn_fn_b", objdump=objdump)
    # The two bodies have internal jne/je to <wn_fn_a+off> vs <wn_fn_b+off>.
    # byte_identical would fail on the differing symbol bytes; within_noise
    # must PASS because the operand classes match once annotations are stripped.
    ok, diff = mod.compare_within_noise(ia, ib)
    assert ok, f"within_noise should tolerate differing branch-target symbol "\
               f"names but failed:\n{diff}"


def test_gnu_objdump_continuation_lines():
    """GNU objdump wraps instructions longer than 7 bytes onto a second,
    mnemonic-less line (e.g. the 9-byte nopw alignment pad before a hot loop).
    The harness must stitch those bytes onto the prior instruction, not error.
    This guards the GNU-objdump path (the rest of the suite prefers
    llvm-objdump). Skips cleanly if GNU objdump is unavailable."""
    import shutil
    gnu = shutil.which("objdump")
    if not gnu or "llvm" in gnu:
        print("  (skip test_gnu_objdump_continuation_lines: no GNU objdump)")
        return
    fixture_c = Path("/tmp/cge_gnu_fixture.c")
    fixture_c.write_text(
        "#include <immintrin.h>\n"
        "void gnu_fn(float* c, const float* a, const float* b, unsigned n){\n"
        "    unsigned e = n / 8;\n"
        "    for (unsigned i = 0; i < e; i++) {\n"
        "        __m256 x = _mm256_load_ps(a + i*8);\n"
        "        __m256 y = _mm256_load_ps(b + i*8);\n"
        "        _mm256_store_ps(c + i*8, _mm256_add_ps(x, y));\n"
        "    }\n"
        "}\n"
    )
    fixture_o = Path("/tmp/cge_gnu_fixture.o")
    subprocess.run(["cc", "-O3", "-mavx", "-c", str(fixture_c),
                    "-o", str(fixture_o)], check=True)
    mod = _load_module()
    # Must not raise on the continuation line GNU objdump emits for the nopw pad.
    instrs = mod.extract_function_body(fixture_o, "gnu_fn", objdump=gnu)
    # If llvm-objdump is also present, the stitched GNU byte stream must match
    # llvm's byte-for-byte, proving the stitch reassembles the wrapped bytes.
    llvm = shutil.which("llvm-objdump") or shutil.which("llvm-objdump-18")
    if llvm:
        ill = mod.extract_function_body(fixture_o, "gnu_fn", objdump=llvm)
        ok, diff = mod.compare_byte_identical(instrs, ill)
        assert ok, f"GNU vs llvm byte streams differ after stitching:\n{diff}"


def test_require_mnemonic_present_passes():
    """A required-mnemonic assertion passes when the pattern is in the body."""
    mod = _load_module()
    body = [{"address": "10", "bytes": ["aa"], "mnemonic": "vaddps",
             "operands": "%ymm0, %ymm1, %ymm2"}]
    ok, diff = mod.check_require_mnemonic(body, "^vadd")
    assert ok, diff


def test_require_mnemonic_absent_fails():
    """It fails (with the observed mnemonics) when the pattern is absent."""
    mod = _load_module()
    body = [{"address": "10", "bytes": ["aa"], "mnemonic": "addss",
             "operands": "%xmm0, %xmm1"}]
    ok, diff = mod.check_require_mnemonic(body, "^vadd")
    assert not ok
    assert "^vadd" in diff and "addss" in diff, diff


# ---------------------------------------------------------------------------
# Cross-compiler robustness (mtibbits/volk#145)
# ---------------------------------------------------------------------------

def test_extract_from_text_seam_basic():
    """The text seam parses a literal disassembly block: collects the body of
    the named symbol, stops at the next symbol, strips the trailing NOP."""
    mod = _load_module()
    text = (
        "0000000000000000 <fixture>:\n"
        "       0: c5 fc 58 04 06               \tvaddps\t(%rsi,%rax), %ymm0, %ymm0\n"
        "       5: c3                           \tretq\n"
        "       6: 90                           \tnop\n"
        "0000000000000010 <other>:\n"
        "      10: c3                           \tretq\n"
    )
    instrs = mod.extract_function_body_from_text(text, "fixture")
    assert [i["mnemonic"] for i in instrs] == ["vaddps", "retq"], instrs


def test_trailing_multibyte_nop_does_not_raise():
    """The multi-byte alignment NOP clang-15+/gcc-11/14 emit as trailing pad
    renders with a SINGLE tab between the last byte and the mnemonic, which
    _disasm_line cannot match. The padding fallback must parse it so the
    trailing-strip drops it -- not raise 'unparsable'. The line below is the
    real rendering captured from `llvm-objdump` on a clang-18 build of
    volk_machine_avx_*.o (mtibbits/volk#145)."""
    mod = _load_module()
    # Normal short instructions render space-padded (>=2 ws before the tab);
    # only the full-width 10-byte NOP renders with a lone tab -- the real
    # captured failing form from llvm-objdump on a clang-18 machine .o.
    text = (
        "0000000000000000 <f1>:\n"
        "       0: c5 fc 58 c1                  \tvaddps\t%ymm1, %ymm0, %ymm0\n"
        "       4: c3                           \tretq\n"
        "       5: 66 2e 0f 1f 84 00 00 00 00 00\tnopw\t%cs:(%rax,%rax)\n"
        "0000000000000020 <f2>:\n"
        "      20: c3                           \tretq\n"
    )
    instrs = mod.extract_function_body_from_text(text, "f1")
    assert [i["mnemonic"] for i in instrs] == ["vaddps", "retq"], instrs


def test_trailing_data16_padding_stripped():
    """A trailing `data16` padding instruction (how GNU objdump renders the
    multi-byte NOP pad) is not part of the function body and must be stripped,
    so a data16-vs-none pad does not fail byte_identical."""
    mod = _load_module()
    with_pad = (
        "0000000000000000 <f>:\n"
        "       0: c3                           \tretq\n"
        "       1: 66 66 2e 0f 1f 84 00 \tdata16\tcs nopw 0x0(%rax,%rax,1)\n"
    )
    without_pad = (
        "0000000000000000 <f>:\n"
        "       0: c3                           \tretq\n"
    )
    a = mod.extract_function_body_from_text(with_pad, "f")
    b = mod.extract_function_body_from_text(without_pad, "f")
    assert [i["mnemonic"] for i in a] == ["retq"], a
    ok, diff = mod.compare_byte_identical(a, b)
    assert ok, diff


def test_symbol_not_standalone_raises_typed():
    """When a symbol is absent (inlined, not emitted standalone) the parser
    raises the typed SymbolNotEmittedError so main() can skip-with-warning,
    not the bare RuntimeError that aggregates into a build-failing CHECK ERROR."""
    mod = _load_module()
    text = ("0000000000000000 <other>:\n"
            "       0: c3\tretq\n")
    try:
        mod.extract_function_body_from_text(text, "inlined_away")
        assert False, "expected SymbolNotEmittedError"
    except mod.SymbolNotEmittedError as e:
        assert "inlined_away" in str(e), str(e)
        # The fixture's only label is the unrelated <other>, so the
        # similar-labels report must state there is nothing similar
        # (mtibbits/volk#225).
        assert "no similar labels" in str(e), str(e)
    # It must be a RuntimeError subclass so existing `except RuntimeError`
    # callers still see it if they do not special-case it.
    assert issubclass(mod.SymbolNotEmittedError, RuntimeError)


def test_object_symbol_name_macho_prefixes():
    """Mach-O mangles C symbols with a leading underscore; the label the
    disassembler shows is '_sym'. Detection is by file magic, not host
    platform, so a cross-compiled Mach-O .o on a Linux host resolves
    correctly (mtibbits/volk#225)."""
    mod = _load_module()
    with tempfile.TemporaryDirectory() as td:
        o = Path(td) / "m.o"
        o.write_bytes(b"\xcf\xfa\xed\xfe" + b"\x00" * 28)  # MH_MAGIC_64 (LE)
        assert mod.object_symbol_name(o, "fwk_fn") == "_fwk_fn"
        o32 = Path(td) / "m32.o"
        o32.write_bytes(b"\xce\xfa\xed\xfe" + b"\x00" * 28)  # MH_MAGIC (LE)
        assert mod.object_symbol_name(o32, "fwk_fn") == "_fwk_fn"
        fat = Path(td) / "fat.o"
        fat.write_bytes(b"\xca\xfe\xba\xbe" + b"\x00" * 28)  # FAT_MAGIC
        assert mod.object_symbol_name(fat, "fwk_fn") == "_fwk_fn"
        fat64 = Path(td) / "fat64.o"
        fat64.write_bytes(b"\xca\xfe\xba\xbf" + b"\x00" * 28)  # FAT_MAGIC_64
        assert mod.object_symbol_name(fat64, "fwk_fn") == "_fwk_fn"


def test_object_symbol_name_elf_and_coff_unchanged():
    """Non-Mach-O objects keep the C name: ELF (Linux lanes) and a non-magic
    blob (COFF has no single 4-byte magic; absence of Mach-O magic must mean
    'no prefix', preserving today's behavior everywhere else)."""
    mod = _load_module()
    with tempfile.TemporaryDirectory() as td:
        o = Path(td) / "e.o"
        o.write_bytes(b"\x7fELF" + b"\x00" * 28)
        assert mod.object_symbol_name(o, "fwk_fn") == "fwk_fn"
        c = Path(td) / "c.obj"
        c.write_bytes(b"\x64\x86\x00\x00" + b"\x00" * 28)
        assert mod.object_symbol_name(c, "fwk_fn") == "fwk_fn"


def test_object_symbol_name_missing_file_raises():
    """Missing .o keeps the established 'object file not found' diagnostic
    (exit-2 class), now raised at the naming seam which runs first."""
    mod = _load_module()
    try:
        mod.object_symbol_name(Path("/nonexistent/x.o"), "fwk_fn")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "object file not found" in str(e), str(e)


# REAL captured llvm-objdump-18 output of a cross-compiled Mach-O object
# (clang -target x86_64-apple-macos12 -O3 -mavx on a machine-TU-shaped
# fixture: static inline impl, address taken in a dispatch table). The
# Mach-O label carries the Darwin ABI's leading underscore. Captured for
# mtibbits/volk#225; regeneration recipe in that issue's plan.
_MACHO_REAL_DISASM = """\

probe_macho.o:	file format mach-o 64-bit x86-64

Disassembly of section __TEXT,__text:

0000000000000000 <_volk_32f_x2_add_32f_a_avx>:
       0: 55                           	pushq	%rbp
       1: 48 89 e5                     	movq	%rsp, %rbp
       4: 85 c9                        	testl	%ecx, %ecx
       6: 0f 84 57 01 00 00            	je	0x163 <_volk_32f_x2_add_32f_a_avx+0x163>
       c: 89 c8                        	movl	%ecx, %eax
       e: 83 f9 20                     	cmpl	$0x20, %ecx
      11: 0f 83 a8 00 00 00            	jae	0xbf <_volk_32f_x2_add_32f_a_avx+0xbf>
      17: 31 c9                        	xorl	%ecx, %ecx
      19: 49 89 c1                     	movq	%rax, %r9
      1c: 49 89 c8                     	movq	%rcx, %r8
      1f: 49 83 e1 03                  	andq	$0x3, %r9
      23: 74 25                        	je	0x4a <_volk_32f_x2_add_32f_a_avx+0x4a>
      25: 49 89 c8                     	movq	%rcx, %r8
      28: 0f 1f 84 00 00 00 00 00      	nopl	(%rax,%rax)
      30: c4 a1 7a 10 04 86            	vmovss	(%rsi,%r8,4), %xmm0
      36: c4 a1 7a 58 04 82            	vaddss	(%rdx,%r8,4), %xmm0, %xmm0
      3c: c4 a1 7a 11 04 87            	vmovss	%xmm0, (%rdi,%r8,4)
      42: 49 ff c0                     	incq	%r8
      45: 49 ff c9                     	decq	%r9
      48: 75 e6                        	jne	0x30 <_volk_32f_x2_add_32f_a_avx+0x30>
      4a: 48 29 c1                     	subq	%rax, %rcx
      4d: 48 83 f9 fc                  	cmpq	$-0x4, %rcx
      51: 0f 87 0c 01 00 00            	ja	0x163 <_volk_32f_x2_add_32f_a_avx+0x163>
      57: 66 0f 1f 84 00 00 00 00 00   	nopw	(%rax,%rax)
      60: c4 a1 7a 10 04 86            	vmovss	(%rsi,%r8,4), %xmm0
      66: c4 a1 7a 58 04 82            	vaddss	(%rdx,%r8,4), %xmm0, %xmm0
      6c: c4 a1 7a 11 04 87            	vmovss	%xmm0, (%rdi,%r8,4)
      72: c4 a1 7a 10 44 86 04         	vmovss	0x4(%rsi,%r8,4), %xmm0
      79: c4 a1 7a 58 44 82 04         	vaddss	0x4(%rdx,%r8,4), %xmm0, %xmm0
      80: c4 a1 7a 11 44 87 04         	vmovss	%xmm0, 0x4(%rdi,%r8,4)
      87: c4 a1 7a 10 44 86 08         	vmovss	0x8(%rsi,%r8,4), %xmm0
      8e: c4 a1 7a 58 44 82 08         	vaddss	0x8(%rdx,%r8,4), %xmm0, %xmm0
      95: c4 a1 7a 11 44 87 08         	vmovss	%xmm0, 0x8(%rdi,%r8,4)
      9c: c4 a1 7a 10 44 86 0c         	vmovss	0xc(%rsi,%r8,4), %xmm0
      a3: c4 a1 7a 58 44 82 0c         	vaddss	0xc(%rdx,%r8,4), %xmm0, %xmm0
      aa: c4 a1 7a 11 44 87 0c         	vmovss	%xmm0, 0xc(%rdi,%r8,4)
      b1: 49 83 c0 04                  	addq	$0x4, %r8
      b5: 4c 39 c0                     	cmpq	%r8, %rax
      b8: 75 a6                        	jne	0x60 <_volk_32f_x2_add_32f_a_avx+0x60>
      ba: e9 a4 00 00 00               	jmp	0x163 <_volk_32f_x2_add_32f_a_avx+0x163>
      bf: 49 89 f8                     	movq	%rdi, %r8
      c2: 49 29 f0                     	subq	%rsi, %r8
      c5: 31 c9                        	xorl	%ecx, %ecx
      c7: 49 81 f8 80 00 00 00         	cmpq	$0x80, %r8
      ce: 0f 82 45 ff ff ff            	jb	0x19 <_volk_32f_x2_add_32f_a_avx+0x19>
      d4: 49 89 f8                     	movq	%rdi, %r8
      d7: 49 29 d0                     	subq	%rdx, %r8
      da: 49 81 f8 80 00 00 00         	cmpq	$0x80, %r8
      e1: 0f 82 32 ff ff ff            	jb	0x19 <_volk_32f_x2_add_32f_a_avx+0x19>
      e7: 89 c1                        	movl	%eax, %ecx
      e9: 83 e1 e0                     	andl	$-0x20, %ecx
      ec: 4c 8d 04 85 00 00 00 00      	leaq	(,%rax,4), %r8
      f4: 49 83 e0 80                  	andq	$-0x80, %r8
      f8: 45 31 c9                     	xorl	%r9d, %r9d
      fb: 0f 1f 44 00 00               	nopl	(%rax,%rax)
     100: c4 a1 7c 10 04 0e            	vmovups	(%rsi,%r9), %ymm0
     106: c4 a1 7c 10 4c 0e 20         	vmovups	0x20(%rsi,%r9), %ymm1
     10d: c4 a1 7c 10 54 0e 40         	vmovups	0x40(%rsi,%r9), %ymm2
     114: c4 a1 7c 10 5c 0e 60         	vmovups	0x60(%rsi,%r9), %ymm3
     11b: c4 a1 7c 58 04 0a            	vaddps	(%rdx,%r9), %ymm0, %ymm0
     121: c4 a1 74 58 4c 0a 20         	vaddps	0x20(%rdx,%r9), %ymm1, %ymm1
     128: c4 a1 6c 58 54 0a 40         	vaddps	0x40(%rdx,%r9), %ymm2, %ymm2
     12f: c4 a1 64 58 5c 0a 60         	vaddps	0x60(%rdx,%r9), %ymm3, %ymm3
     136: c4 a1 7c 11 04 0f            	vmovups	%ymm0, (%rdi,%r9)
     13c: c4 a1 7c 11 4c 0f 20         	vmovups	%ymm1, 0x20(%rdi,%r9)
     143: c4 a1 7c 11 54 0f 40         	vmovups	%ymm2, 0x40(%rdi,%r9)
     14a: c4 a1 7c 11 5c 0f 60         	vmovups	%ymm3, 0x60(%rdi,%r9)
     151: 49 83 e9 80                  	subq	$-0x80, %r9
     155: 4d 39 c8                     	cmpq	%r9, %r8
     158: 75 a6                        	jne	0x100 <_volk_32f_x2_add_32f_a_avx+0x100>
     15a: 48 39 c1                     	cmpq	%rax, %rcx
     15d: 0f 85 b6 fe ff ff            	jne	0x19 <_volk_32f_x2_add_32f_a_avx+0x19>
     163: 5d                           	popq	%rbp
     164: c5 f8 77                     	vzeroupper
     167: c3                           	retq
"""


def test_extract_function_body_macho_wiring_real_capture():
    """extract_function_body maps the C name to its Mach-O label BEFORE
    disassembly: the label reaches --disassemble-symbols (else llvm-objdump
    silently disassembles nothing for the unprefixed name, rc=0 measured)
    and the header match. The disassembly text is the REAL captured
    llvm-objdump-18 output of a cross-compiled Mach-O machine-TU-shaped
    fixture (static inline impl, address taken in a dispatch table) --
    a real produced artifact fed through the consumer, not a hand-written
    format claim (mtibbits/volk#225)."""
    mod = _load_module()
    with tempfile.TemporaryDirectory() as td:
        o = Path(td) / "m.o"
        o.write_bytes(b"\xcf\xfa\xed\xfe" + b"\x00" * 28)
        seen = {}

        def fake_disassemble(o_file, objdump, symbol=None):
            seen["symbol"] = symbol
            return _MACHO_REAL_DISASM

        with _patched_disassemble(mod, fake_disassemble):
            instrs = mod.extract_function_body(
                o, "volk_32f_x2_add_32f_a_avx", objdump="llvm-objdump")
    assert seen["symbol"] == "_volk_32f_x2_add_32f_a_avx", seen
    # Structural pins, stable across a toolchain re-capture: whole real
    # body parses, prologue/epilogue mnemonics, and the AVX payload.
    assert len(instrs) >= 50, len(instrs)
    assert instrs[0]["mnemonic"] == "pushq", instrs[0]
    assert instrs[-1]["mnemonic"] == "retq", instrs[-1]
    assert any(i["mnemonic"] == "vaddps" for i in instrs)


def test_extract_function_body_elf_label_unchanged():
    """Regression pin for the untouched branch (GREEN before and after the
    fix -- declared, not born-red): an ELF object keeps the C name
    end-to-end, so no prefix leaks into non-Mach-O lookups and Linux-lane
    results stay byte-identical (AC2)."""
    mod = _load_module()
    with tempfile.TemporaryDirectory() as td:
        o = Path(td) / "e.o"
        o.write_bytes(b"\x7fELF" + b"\x00" * 28)
        seen = {}

        def fake_disassemble(o_file, objdump, symbol=None):
            seen["symbol"] = symbol
            return (
                "0000000000000000 <fwk_fn>:\n"
                "       0: c3                           \tretq\n"
            )

        with _patched_disassemble(mod, fake_disassemble):
            instrs = mod.extract_function_body(o, "fwk_fn", objdump="llvm-objdump")
    assert seen["symbol"] == "fwk_fn", seen
    assert [i["mnemonic"] for i in instrs] == ["retq"], instrs


def test_not_found_retries_unfiltered_for_similar_labels():
    """The llvm fast path FILTERS disassembly to the requested label
    (--disassemble-symbols), so a wrong label yields output with ZERO labels
    -- which would starve the similar-labels diagnostic into 'no similar
    labels among 0 seen'. On not-found, extract_function_body must retry
    UNFILTERED so the raised error reports what the object actually contains
    (mtibbits/volk#225 quality pass)."""
    mod = _load_module()
    with tempfile.TemporaryDirectory() as td:
        o = Path(td) / "m.o"
        o.write_bytes(b"\xcf\xfa\xed\xfe" + b"\x00" * 28)
        calls = []

        def fake_disassemble(o_file, objdump, symbol=None):
            calls.append(symbol)
            if symbol is not None:
                # llvm-objdump behavior for a missing symbol: header only,
                # zero label lines, rc=0 (measured).
                return "m.o:\tfile format mach-o 64-bit x86-64\n"
            return (
                "m.o:\tfile format mach-o 64-bit x86-64\n"
                "\n"
                "0000000000000000 <_wanted_fn_v2>:\n"
                "       0: c3                           \tretq\n"
            )

        with _patched_disassemble(mod, fake_disassemble):
            try:
                mod.extract_function_body(o, "wanted_fn", objdump="llvm-objdump")
                assert False, "expected SymbolNotEmittedError"
            except mod.SymbolNotEmittedError as e:
                assert "similar labels present" in str(e), str(e)
                assert "_wanted_fn_v2" in str(e), str(e)
                # The C name is what a reader can actually grep the manifest
                # and tree for -- the label's underscore exists nowhere there.
                assert "(C symbol 'wanted_fn')" in str(e), str(e)
    # First call filtered (the mapped label), second call the unfiltered retry.
    assert calls == ["_wanted_fn", None], calls


def test_not_found_retry_recovers_extraction():
    """The retry is behavior, not just diagnostics: when the symbol filter
    misses a label that whole-object disassembly DOES render (an objdump
    version/symbol-table quirk), the unfiltered retry must recover the
    extraction -- returning instructions where the pre-retry harness would
    have skipped the tuple. This is the branch that can turn a former skip
    into a real comparison, so it must be executed, not just described
    (mtibbits/volk#225 red-team)."""
    mod = _load_module()
    with tempfile.TemporaryDirectory() as td:
        o = Path(td) / "m.o"
        o.write_bytes(b"\xcf\xfa\xed\xfe" + b"\x00" * 28)
        calls = []

        def fake_disassemble(o_file, objdump, symbol=None):
            calls.append(symbol)
            if symbol is not None:
                # Filter misses: header only, zero label lines, rc=0.
                return "m.o:\tfile format mach-o 64-bit x86-64\n"
            return (
                "m.o:\tfile format mach-o 64-bit x86-64\n"
                "\n"
                "0000000000000000 <_wanted_fn>:\n"
                "       0: c3                           \tretq\n"
            )

        with _patched_disassemble(mod, fake_disassemble):
            instrs = mod.extract_function_body(o, "wanted_fn",
                                               objdump="llvm-objdump")
    assert calls == ["_wanted_fn", None], calls
    assert [i["mnemonic"] for i in instrs] == ["retq"], instrs


def test_not_found_no_retry_when_unfiltered():
    """GNU objdump has no symbol filter, so the first disassembly was already
    whole-object -- a not-found there must raise WITHOUT a second disassembly
    (on macOS, which resolves Apple's non-'llvm'-named objdump, a retry would
    re-disassemble the whole machine .o for identical text on every skip)."""
    mod = _load_module()
    with tempfile.TemporaryDirectory() as td:
        o = Path(td) / "e.o"
        o.write_bytes(b"\x7fELF" + b"\x00" * 28)
        calls = []

        def fake_disassemble(o_file, objdump, symbol=None):
            calls.append(symbol)
            return (
                "0000000000000000 <other_fn>:\n"
                "       0: c3                           \tretq\n"
            )

        with _patched_disassemble(mod, fake_disassemble):
            try:
                mod.extract_function_body(o, "wanted_fn", objdump="objdump")
                assert False, "expected SymbolNotEmittedError"
            except mod.SymbolNotEmittedError:
                pass
    # Exactly one disassembly: unfiltered tools get no retry.
    assert calls == ["wanted_fn"], calls


def test_symbol_not_emitted_reports_similar_labels():
    """When the requested label is absent but a near-name IS present (the
    Mach-O underscore family of failures), the error must say so -- one
    line separates 'emitted under a name we didn't ask for' from 'not
    emitted', so the skip warning discriminates at the source instead of
    forcing stderr archaeology (mtibbits/volk#225)."""
    mod = _load_module()
    text = ("0000000000000000 <_wanted_fn>:\n"
            "       0: c3                           \tretq\n")
    try:
        mod.extract_function_body_from_text(text, "wanted_fn")
        assert False, "expected SymbolNotEmittedError"
    except mod.SymbolNotEmittedError as e:
        assert "similar labels present" in str(e), str(e)
        assert "_wanted_fn" in str(e), str(e)


def test_padding_strip_does_not_swallow_real_instruction():
    """Negative control: a trailing NON-padding instruction (an extra vaddps)
    must remain and still cause a divergence -- the strip is padding-only."""
    mod = _load_module()
    longer = ("0000000000000000 <f>:\n"
              "       0: c5 fc 58 c1                  \tvaddps\t%ymm1, %ymm0, %ymm0\n"
              "       4: c5 fc 58 c1                  \tvaddps\t%ymm1, %ymm0, %ymm0\n"
              "       8: c3                           \tretq\n")
    shorter = ("0000000000000000 <f>:\n"
               "       0: c5 fc 58 c1                  \tvaddps\t%ymm1, %ymm0, %ymm0\n"
               "       4: c3                           \tretq\n")
    a = mod.extract_function_body_from_text(longer, "f")
    b = mod.extract_function_body_from_text(shorter, "f")
    ok, diff = mod.compare_byte_identical(a, b)
    assert not ok and "count differs" in diff, (ok, diff)


def test_end_to_end_divergent_pair_fails_build():
    """Negative control (end-to-end): a manifest comparing two genuinely
    divergent impls (add vs mul) exits 1 with CHECK FAILED -- proving the
    hardened parser still breaks the build on a real codegen divergence."""
    import shutil
    cc = shutil.which("cc")
    objdump = _which_objdump()
    if not cc:
        print("  (skip test_end_to_end_divergent_pair_fails_build: no cc)")
        return
    add = ("#include <immintrin.h>\n"
           "__m256 nc_fn(__m256 a,__m256 b){return _mm256_add_ps(a,b);}\n")
    mul = ("#include <immintrin.h>\n"
           "__m256 nc_fn(__m256 a,__m256 b){return _mm256_mul_ps(a,b);}\n")
    a_c = Path("/tmp/cge_nc_a.c"); a_c.write_text(add)
    b_c = Path("/tmp/cge_nc_b.c"); b_c.write_text(mul)
    a_o = Path("/tmp/cge_nc_a.o"); b_o = Path("/tmp/cge_nc_b.o")
    for s, o in ((a_c, a_o), (b_c, b_o)):
        subprocess.run([cc, "-O3", "-mavx", "-c", str(s), "-o", str(o)],
                       check=True)
    manifest = {"tuples": [{
        "kernel": "nc", "isa": "avx", "alignment": "a",
        "impl_a": {"symbol": "nc_fn", "machine_o": str(a_o)},
        "impl_b": {"symbol": "nc_fn", "machine_o": str(b_o)},
        "criterion": "byte_identical",
    }]}
    mpath = Path("/tmp/cge_nc_manifest.json"); mpath.write_text(json.dumps(manifest))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(mpath),
         "--build-lib-dir", "/tmp", "--objdump", objdump],
        capture_output=True, text=True)
    assert result.returncode == 1, (result.returncode, result.stdout, result.stderr)
    assert "CHECK FAILED" in result.stderr, result.stderr


def test_padding_fallback_data16_single_tab():
    """Exercises the _padding_line fallback's `data16` arm specifically: a
    single-tab `data16` pad line (which _disasm_line cannot parse) must be
    caught by the fallback and then stripped. A space-padded `data16` line would
    not reach the fallback -- it matches the primary _disasm_line -- so without
    this case the fallback's data16 branch is untested (red-team Nit G2)."""
    mod = _load_module()
    text = (
        "0000000000000000 <f>:\n"
        "       0: c3                           \tretq\n"
        "       1: 66 66 2e 0f 1f 84 00 00 00\tdata16\tcs nopw 0x0(%rax,%rax,1)\n"
    )
    instrs = mod.extract_function_body_from_text(text, "f")
    assert [i["mnemonic"] for i in instrs] == ["retq"], instrs


def test_padding_line_mnemonics_are_strippable():
    """Invariant (red-team Nit J): every mnemonic the _padding_line fallback
    accepts must also be removed by _is_trailing_padding -- otherwise a
    fallback-absorbed pad would survive into the comparison and silently change
    it. Pin representative members of the accepted set so the invariant is
    self-defending if someone extends one side without the other."""
    mod = _load_module()
    for m in ("nop", "nopw", "nopl", "nopq", "data16"):
        assert mod._is_trailing_padding({"mnemonic": m, "operands": ""}), m


def test_trailing_xchg_and_int3_padding_stripped():
    """Trailing `xchg %ax,%ax` (0x66 0x90, the legacy 2-byte NOP) and `int3`
    (0xcc gap fill) are inter-function alignment padding and must be stripped.
    Real-world: clang-15 on ubuntu-22.04 emits a trailing `xchg %ax,%ax` after
    the function's final jmp, which failed byte_identical before this
    (mtibbits/volk#145)."""
    mod = _load_module()
    xchg_pad = (
        "0000000000000000 <f>:\n"
        "       0: eb 00                        \tjmp\t0x2 <f+0x2>\n"
        "       2: 66 90                        \txchg\t%ax, %ax\n"
    )
    int3_pad = (
        "0000000000000000 <f>:\n"
        "       0: c3                           \tretq\n"
        "       1: cc                           \tint3\n"
    )
    plain = (
        "0000000000000000 <f>:\n"
        "       0: c3                           \tretq\n"
    )
    a = mod.extract_function_body_from_text(xchg_pad, "f")
    b = mod.extract_function_body_from_text(int3_pad, "f")
    c = mod.extract_function_body_from_text(plain, "f")
    assert [i["mnemonic"] for i in a] == ["jmp"], a
    assert [i["mnemonic"] for i in b] == ["retq"], b
    assert mod.compare_byte_identical(b, c)[0]


def test_real_xchg_is_not_stripped():
    """Safety: a genuine register exchange `xchg %rax,%rbx` (different operands)
    must NOT be treated as padding -- only the self-exchange NOP idiom is."""
    mod = _load_module()
    assert not mod._is_trailing_padding(
        {"mnemonic": "xchg", "operands": "%rax, %rbx"})
    assert mod._is_trailing_padding(
        {"mnemonic": "xchg", "operands": "%ax, %ax"})


def test_require_standalone_must_be_bool():
    """parse_manifest rejects a non-boolean require_standalone with exit 2."""
    manifest = {"tuples": [{
        "kernel": "k", "isa": "avx", "alignment": "a",
        "impl_a": {"symbol": "s", "machine_o": "m.o"},
        "impl_b": {"symbol": "s2", "machine_o": "m2.o"},
        "criterion": "byte_identical",
        "require_standalone": "yes",  # not a bool
    }]}
    manifest_path = Path("/tmp/cge_reqstandalone_bad_manifest.json")
    manifest_path.write_text(json.dumps(manifest))
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--manifest", str(manifest_path), "--list-only"],
        capture_output=True, text=True,
    )
    assert result.returncode == 2, (result.returncode, result.stdout)
    assert "require_standalone" in result.stderr, result.stderr


def _compile_present_fn_o(tag):
    """Compile a TU defining `present_fn` into a .o; return (cc, objdump, o_path)
    or (None, None, None) when the toolchain is unavailable (caller self-skips)."""
    import shutil
    cc = shutil.which("cc")
    if not cc:
        return None, None, None
    objdump = _which_objdump()
    src = ("#include <immintrin.h>\n"
           "__m256 present_fn(__m256 a,__m256 b){return _mm256_add_ps(a,b);}\n")
    c = Path(f"/tmp/cge_{tag}.c"); c.write_text(src)
    o = Path(f"/tmp/cge_{tag}.o")
    subprocess.run([cc, "-O3", "-mavx", "-c", str(c), "-o", str(o)], check=True)
    return cc, objdump, o


def test_require_standalone_inlined_hard_fails():
    """A tuple marked require_standalone whose symbol is absent (inlined away)
    is a HARD failure: main() exits 1 and names the tuple. The .o defines
    `present_fn` but the tuple references `inlined_away`, so the first
    extraction raises SymbolNotEmittedError -- the deterministic stand-in for
    a compiler having inlined the impl."""
    cc, objdump, o = _compile_present_fn_o("reqstandalone_fail")
    if not cc:
        print("  (skip test_require_standalone_inlined_hard_fails: no cc)")
        return
    manifest = {"tuples": [{
        "kernel": "nc", "isa": "avx", "alignment": "a",
        "impl_a": {"symbol": "inlined_away", "machine_o": str(o)},
        "impl_b": {"symbol": "present_fn", "machine_o": str(o)},
        "criterion": "byte_identical",
        "require_standalone": True,
    }]}
    mpath = Path("/tmp/cge_reqstandalone_fail_manifest.json")
    mpath.write_text(json.dumps(manifest))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(mpath),
         "--build-lib-dir", "/tmp", "--objdump", objdump],
        capture_output=True, text=True)
    assert result.returncode == 1, (result.returncode, result.stdout, result.stderr)
    assert "CHECK FAILED" in result.stderr, result.stderr
    assert "require_standalone assertion FAILED" in result.stderr, result.stderr
    assert "nc.avx.a" in result.stderr, result.stderr


def test_unmarked_inlined_still_skips():
    """Negative control: an inlined-away tuple WITHOUT the opt-in keeps
    skip-with-warning behavior -- exit 0 -- provided the kernel retains real
    coverage (a second, checked tuple in the SAME kernel). The all-skip
    configuration is now a zero-coverage failure (mtibbits/volk#165),
    covered separately."""
    cc, objdump, o = _compile_present_fn_o("reqstandalone_skip")
    if not cc:
        print("  (skip test_unmarked_inlined_still_skips: no cc)")
        return
    manifest = {"tuples": [
        {
            "kernel": "nc", "isa": "avx", "alignment": "a",
            "impl_a": {"symbol": "inlined_away", "machine_o": str(o)},
            "impl_b": {"symbol": "present_fn", "machine_o": str(o)},
            "criterion": "byte_identical",
            # no require_standalone
        },
        {
            "kernel": "nc", "isa": "avx", "alignment": "u",
            "impl_a": {"symbol": "present_fn", "machine_o": str(o)},
            "impl_b": {"symbol": "present_fn", "machine_o": str(o)},
            "criterion": "byte_identical",
        },
    ]}
    mpath = Path("/tmp/cge_reqstandalone_skip_manifest.json")
    mpath.write_text(json.dumps(manifest))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(mpath),
         "--build-lib-dir", "/tmp", "--objdump", objdump],
        capture_output=True, text=True)
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert "WARNING: skipping nc.avx.a" in result.stderr, result.stderr
    assert "1 tuples checked" in result.stdout, result.stdout


def test_all_skipped_zero_coverage_fails():
    """Global zero coverage: every declared tuple skips (inlined away), so
    nothing was verified -- the run must FAIL loudly, not print
    `ok (0 tuples checked)` (mtibbits/volk#165)."""
    cc, objdump, o = _compile_present_fn_o("zerocov_global")
    if not cc:
        print("  (skip test_all_skipped_zero_coverage_fails: no cc)")
        return
    manifest = {"tuples": [
        {
            "kernel": "ka", "isa": "avx", "alignment": "a",
            "impl_a": {"symbol": "inlined_away", "machine_o": str(o)},
            "impl_b": {"symbol": "present_fn", "machine_o": str(o)},
            "criterion": "byte_identical",
        },
        {
            "kernel": "kb", "isa": "avx", "alignment": "u",
            "impl_a": {"symbol": "also_inlined_away", "machine_o": str(o)},
            "impl_b": {"symbol": "present_fn", "machine_o": str(o)},
            "criterion": "byte_identical",
        },
    ]}
    mpath = Path("/tmp/cge_zerocov_global_manifest.json")
    mpath.write_text(json.dumps(manifest))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(mpath),
         "--build-lib-dir", "/tmp", "--objdump", objdump],
        capture_output=True, text=True)
    assert result.returncode == 1, (result.returncode, result.stdout, result.stderr)
    assert "CHECK FAILED" in result.stderr, result.stderr
    assert "zero coverage" in result.stderr, result.stderr
    assert "ok (" not in result.stdout, result.stdout


def test_per_kernel_all_skip_fails():
    """Per-kernel all-skip: kernels `bhole` and `ahole` have every tuple
    skipped while kernel `covd` is checked (with one additional skip of its
    own) -- the run must name the hole kernels sorted in the coverage
    diagnostic and exit non-zero; `covd` (which has coverage) must not be
    flagged, and its skipped tuple must not be listed as removable
    (mtibbits/volk#165)."""
    cc, objdump, o = _compile_present_fn_o("zerocov_kernel")
    if not cc:
        print("  (skip test_per_kernel_all_skip_fails: no cc)")
        return
    manifest = {"tuples": [
        {
            "kernel": "bhole", "isa": "avx", "alignment": "a",
            "impl_a": {"symbol": "inlined_away", "machine_o": str(o)},
            "impl_b": {"symbol": "present_fn", "machine_o": str(o)},
            "criterion": "byte_identical",
        },
        {
            "kernel": "ahole", "isa": "avx", "alignment": "a",
            "impl_a": {"symbol": "inlined_away", "machine_o": str(o)},
            "impl_b": {"symbol": "present_fn", "machine_o": str(o)},
            "criterion": "byte_identical",
        },
        {
            "kernel": "covd", "isa": "avx", "alignment": "a",
            "impl_a": {"symbol": "present_fn", "machine_o": str(o)},
            "impl_b": {"symbol": "present_fn", "machine_o": str(o)},
            "criterion": "byte_identical",
        },
        {
            "kernel": "covd", "isa": "avx", "alignment": "u",
            "impl_a": {"symbol": "inlined_away", "machine_o": str(o)},
            "impl_b": {"symbol": "present_fn", "machine_o": str(o)},
            "criterion": "byte_identical",
        },
    ]}
    mpath = Path("/tmp/cge_zerocov_kernel_manifest.json")
    mpath.write_text(json.dumps(manifest))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(mpath),
         "--build-lib-dir", "/tmp", "--objdump", objdump],
        capture_output=True, text=True)
    assert result.returncode == 1, (result.returncode, result.stdout, result.stderr)
    assert "CHECK FAILED" in result.stderr, result.stderr
    guard_lines = [ln for ln in result.stderr.splitlines()
                   if "zero codegen coverage for kernel(s):" in ln]
    assert guard_lines, result.stderr
    assert "kernel(s): ahole, bhole" in guard_lines[0], result.stderr
    assert "covd" not in guard_lines[0], result.stderr
    skip_lines = [ln for ln in result.stderr.splitlines()
                  if ln.startswith("  skipped:")]
    assert skip_lines, result.stderr
    assert "ahole.avx.a" in skip_lines[0], result.stderr
    assert "bhole.avx.a" in skip_lines[0], result.stderr
    assert "covd" not in skip_lines[0], result.stderr


def test_partial_kernel_coverage_ok():
    """A kernel with >=1 checked tuple is covered even if its other tuples
    skip: one skip within kernel `pk` (which also has a checked tuple) stays
    exit 0. Guards the per-kernel check against over-firing
    (mtibbits/volk#165)."""
    cc, objdump, o = _compile_present_fn_o("partialcov")
    if not cc:
        print("  (skip test_partial_kernel_coverage_ok: no cc)")
        return
    manifest = {"tuples": [
        {
            "kernel": "pk", "isa": "avx", "alignment": "a",
            "impl_a": {"symbol": "present_fn", "machine_o": str(o)},
            "impl_b": {"symbol": "present_fn", "machine_o": str(o)},
            "criterion": "byte_identical",
        },
        {
            "kernel": "pk", "isa": "avx", "alignment": "u",
            "impl_a": {"symbol": "inlined_away", "machine_o": str(o)},
            "impl_b": {"symbol": "present_fn", "machine_o": str(o)},
            "criterion": "byte_identical",
        },
    ]}
    mpath = Path("/tmp/cge_partialcov_manifest.json")
    mpath.write_text(json.dumps(manifest))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(mpath),
         "--build-lib-dir", "/tmp", "--objdump", objdump],
        capture_output=True, text=True)
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert "1 tuples checked" in result.stdout, result.stdout


def test_divergence_precedes_coverage_guard():
    """Ordering pin: a real divergence in kernel `dv` plus an all-skip kernel
    `hole2` must report the DIVERGENCE (per-tuple diff, exit 1), not the
    coverage message -- the guard sits after the failures exit and must never
    mask a divergence diagnostic (mtibbits/volk#165). Green before and after
    the guard lands; it pins the guard's position, not its existence."""
    cc, objdump, o = _compile_present_fn_o("dv_guard")
    if not cc:
        print("  (skip test_divergence_precedes_coverage_guard: no cc)")
        return
    mul = ("#include <immintrin.h>\n"
           "__m256 present_fn(__m256 a,__m256 b){return _mm256_mul_ps(a,b);}\n")
    b_c = Path("/tmp/cge_dv_mul.c"); b_c.write_text(mul)
    b_o = Path("/tmp/cge_dv_mul.o")
    subprocess.run([cc, "-O3", "-mavx", "-c", str(b_c), "-o", str(b_o)],
                   check=True)
    manifest = {"tuples": [
        {
            "kernel": "dv", "isa": "avx", "alignment": "a",
            "impl_a": {"symbol": "present_fn", "machine_o": str(o)},
            "impl_b": {"symbol": "present_fn", "machine_o": str(b_o)},
            "criterion": "byte_identical",
        },
        {
            "kernel": "hole2", "isa": "avx", "alignment": "a",
            "impl_a": {"symbol": "inlined_away", "machine_o": str(o)},
            "impl_b": {"symbol": "present_fn", "machine_o": str(o)},
            "criterion": "byte_identical",
        },
    ]}
    mpath = Path("/tmp/cge_dv_guard_manifest.json")
    mpath.write_text(json.dumps(manifest))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(mpath),
         "--build-lib-dir", "/tmp", "--objdump", objdump],
        capture_output=True, text=True)
    assert result.returncode == 1, (result.returncode, result.stdout, result.stderr)
    assert "byte_identical comparison FAILED" in result.stderr, result.stderr
    assert "zero codegen coverage" not in result.stderr, result.stderr


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
