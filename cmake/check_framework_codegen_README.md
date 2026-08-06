# Codegen-Equivalence Test Harness

A build-time check that asserts byte-identical-or-within-noise machine code
between declared `(kernel, ISA, alignment)` impl pairs. It is the codegen-side
counterpart to the dispatch-table integrity check (mtibbits/volk#58, PR #59):
that check verifies the right impls are *wired into dispatch*; this one
verifies a framework-instantiated impl *emits the machine code its hand-written
reference emits*.

Context: mtibbits/volk#78.

## Why this exists

The volk fusion framework rewrites hand-written SIMD kernels as instantiations
of a generic driver. The framework's core correctness invariant is that the
generated inner loop is byte-identical (or within measurement noise) to the
hand-written impl it replaces. Without a permanent check, the first downstream
framework refactor that subtly perturbs codegen on one ISA lands unnoticed.
This harness runs in every build and breaks it on a regression, with a diff.

## When to add a tuple

When a new framework-instantiated impl ships, declare a tuple comparing it
against the hand-written impl it replaces (or, for a pure framework rewrite,
against the pre-rewrite impl captured as a reference). The build then fails
fast if the framework's codegen ever diverges.

## What gets compared: the whole function

The harness compares the **whole function body** of each impl — every
instruction from the `<symbol>:` disassembly header to the next symbol — not
just the inner loop. This needs **no source markers**: you declare a tuple
naming the two symbols, and the harness disassembles and compares them.

> **Why whole-function, not inner-loop?** An earlier design bracketed just the
> inner loop with source-level asm labels. That was abandoned: inserting a
> label symbol mid-function makes its address a basic-block boundary the
> optimizer must respect, which *perturbs the surrounding codegen* (an added
> test instruction, register-allocation churn). Instrumenting a kernel for the
> harness would change the kernel that ships — unacceptable. Whole-function
> comparison leaves production codegen untouched and is the stronger contract:
> a framework instantiation that reproduces a hand-written impl produces the
> same *function*, not merely the same loop.

Requirements on a compared impl:

- It must be **emitted standalone** — its symbol must appear in the object
  file. Volk's dispatch-table kernels satisfy this automatically because their
  address is taken for the dispatch array. A purely-inlined-away `static
  inline` with no address taken would have no symbol to find (the harness
  errors clearly: "symbol not found").
- Trailing NOP-family padding after the function's final `ret`/`jmp` is
  alignment padding for the *next* function; the harness strips it before
  comparing, so inter-function layout differences do not cause false failures.
  Internal alignment nops (mid-body, e.g. before a hot loop) are preserved.

## Manifest format

Tuples are declared from CMake via the `ADD_CODEGEN_EQUIVALENCE_TUPLE` macro
(defined in `cmake/CodegenEquivalence.cmake`). One invocation per tuple:

```cmake
ADD_CODEGEN_EQUIVALENCE_TUPLE(
    KERNEL    volk_32f_x2_add_32f
    ISA       avx
    ALIGNMENT a
    CRITERION byte_identical
    IMPL_A_SYMBOL    volk_32f_x2_add_32f_a_avx
    IMPL_A_MACHINE_O volk_machine_avx_64_mmx_orc.c.o
    IMPL_B_SYMBOL    volk_32f_x2_add_32f_a_avx_ref
    IMPL_B_MACHINE_O codegen_bootstrap_ref.c.o
)
```

- `SYMBOL` is the function symbol the harness disassembles and compares.
- `MACHINE_O` is the **bare object-file name**. The harness searches for it
  recursively under the build tree (`build/lib/CMakeFiles/`), so different
  OBJECT libraries landing their `.o` in different `<target>.dir/` subdirs
  need no path hardcoding. An absolute path is also accepted.
- When the two impls are compiled with the same codegen-relevant flags, use
  `byte_identical`. When a wrapper or a different flag set introduces benign
  register/immediate churn without changing the instruction stream's shape,
  use `within_noise`.
- Two optional opt-in fields may also be declared per tuple:
  `REQUIRE_MNEMONIC <regex>` (assert a present symbol contains a given
  instruction — see *Required-mnemonic assertion*) and `REQUIRE_STANDALONE`
  (a bare boolean flag that makes an inlined-away impl a hard failure — see
  *Require-standalone assertion*). Both default off; omitting them leaves the
  emitted manifest JSON byte-identical.

`FINALIZE_CODEGEN_EQUIVALENCE_HARNESS()` is called once (already wired in
`lib/CMakeLists.txt`) after all tuple declarations and after `volk_obj` is
defined. It writes `build/codegen_equivalence_manifest.json` and creates the
`volk_check_framework_codegen` target, which the `volk` shared library depends
on.

## Equivalence criteria

- **`byte_identical`** — the raw instruction-byte sequences must match exactly
  between impl A and impl B. Use this for framework instantiations designed to
  compile to the same machine code as the hand-written predecessor (Approach A
  in the fusion design doc §4).
- **`within_noise`** — mnemonic sequences must match in order, and operand
  *classes* must match; differences in specific register identifiers (`%ymm0`
  vs `%ymm2`), immediate values, and hex displacements are tolerated.
  Instruction **order is not** tolerated — a reordered loop body fails. Use
  this when a static-inline wrapper (Approach B) or a different optimization
  flag set introduces register-allocation churn without changing the
  instruction stream's shape.

## Required-mnemonic assertion (scalar-fallback guard)

Equivalence to a reference does not prove an impl emits the intended ISA — a
scalar fallback could be byte-identical to a scalar reference and pass every
criterion above. Declare an optional `REQUIRE_MNEMONIC <regex>` on a tuple to
*additionally* assert the compared function contains at least one instruction
whose mnemonic matches the regex, checked against **each** impl independently:

```cmake
ADD_CODEGEN_EQUIVALENCE_TUPLE(
    ...
    REQUIRE_MNEMONIC "^vaddps"   # the function MUST emit a packed vaddps
)
```

The regex is matched against each instruction's mnemonic (pass if any matches).
On failure the build halts, naming the tuple, the pattern, and the mnemonics
actually present. The field is **opt-in**: tuples without it are unaffected and
produce identical manifest JSON to before. Use it on framework-instantiation
tuples to guarantee the instantiation didn't silently fall back to scalar code.

## Require-standalone assertion (inlined-away guard)

By default, when the compiler inlines an impl rather than emitting it as a
standalone symbol there is nothing to disassemble, so the tuple is
**skipped with a warning** and the build stays green (see *Diagnostics* and
*Cross-compiler robustness* below) — bounded by the *Zero-coverage guard*:
aggregate all-skip configurations fail. That is the right default for impls
whose standalone emission is compiler-dependent — but it leaves a
masked-regression class for impls that the dispatch table relies on existing
as real, separately dispatchable symbols: if such an impl gets inlined away
by a future refactor, the symbol silently disappears and the check still
passes while the kernel retains other coverage.

Declare the optional boolean flag `REQUIRE_STANDALONE` (pass the **bare keyword**,
no value) on such a tuple to flip that outcome: when the impl is inlined away,
the checker reports a **hard failure** (exit 1) for that tuple instead of
skip-with-warning.

```cmake
ADD_CODEGEN_EQUIVALENCE_TUPLE(
    ...
    REQUIRE_STANDALONE           # inlined-away is a regression here, not OK
)
```

This is an **orthogonal axis** to `REQUIRE_MNEMONIC`: the latter asserts which
instructions a *present* symbol contains; `REQUIRE_STANDALONE` asserts the symbol
*exists* as a dispatchable standalone in the first place. The field is **opt-in**:
tuples without it are unaffected and produce identical manifest JSON to before
(unmarked tuples keep the skip-with-warning behavior, exit 0 while real
coverage remains — see *Zero-coverage guard*).

## Zero-coverage guard (mtibbits/volk#165)

Skipping is per-tuple, but success is not unconditional. Two aggregate
configurations fail loudly (exit 1) instead of reporting `ok`:

- **Global zero coverage** — every declared tuple was skipped
  (`checked == 0`). The run verified nothing, so it does not report
  success. Diagnostic: `CHECK FAILED: zero coverage`.
- **Per-kernel all-skip** — every tuple belonging to one kernel was
  skipped while other kernels were checked. That kernel has no codegen
  coverage, so the run fails and names it. Diagnostic:
  `CHECK FAILED: zero codegen coverage for kernel(s): <names>`.
  A kernel with at least one checked tuple is covered; its remaining
  skips stay warnings.

Both diagnostics point at the usual cause — a toolchain change that
starts inlining the compared impls — and the remedies: fix the build so
the symbols are emitted standalone, or deliberately remove the affected
tuples from the manifest. The empty-manifest case is distinct and stays
green: zero tuples *declared* is a valid configuration
(`ok (0 tuples declared)`, exit 0); zero tuples *verified out of some
declared* is not.

**Blind spots (what this guard cannot decide):** coverage is aggregated
on the bare kernel name, so a kernel checked on one ISA but wholly
skipped on another still passes (an ISA-level hole); and tuples that
silently stop being *declared* — e.g. the CMake machine-name gate in
`lib/CMakeLists.txt` ceasing to match — land in the legitimate
`ok (0 tuples declared)` path. Both are outside this guard's contract.

## How the check runs

1. **CMake configure:** each `ADD_CODEGEN_EQUIVALENCE_TUPLE` appends a JSON
   fragment to a global property.
2. **CMake configure end:** `FINALIZE_CODEGEN_EQUIVALENCE_HARNESS` writes the
   manifest and creates the `volk_check_framework_codegen` custom target,
   resolving a disassembler (`llvm-objdump`, then `llvm-objdump-18`, then
   `objdump`).
3. **Build:** after `volk_obj` compiles, the target runs
   `cmake/check_framework_codegen.py --manifest ... --build-lib-dir
   build/lib/CMakeFiles --objdump <found>`.
4. The script disassembles each declared `SYMBOL` in its `MACHINE_O`, extracts
   the whole function body, and applies the criterion. On failure it halts the
   build with a per-tuple diff.

With **zero tuples declared** the manifest is `{"tuples":[]}` and the check
prints `codegen-equivalence: ok (0 tuples declared)` and exits 0 — no false
positives on a clean tree.

## Diagnostics

- `symbol '<x>' not found in disassembly of <o>` — the impl was inlined away
  (no address taken, so no symbol emitted), or the symbol is in a different
  `.o` than declared. Skipped-with-warning by default; a hard failure if the
  tuple declares `REQUIRE_STANDALONE`. Aggregate skips are bounded by the
  zero-coverage guard (see above).
- `require_standalone assertion FAILED: implementation was not emitted as a
  standalone dispatchable symbol (inlined away)` — a tuple declaring
  `REQUIRE_STANDALONE` had its impl inlined away; the build fails (exit 1).
- `object file '<x>' not found under <dir>` — the `MACHINE_O` name does not
  match any compiled object; check the build actually produced it.
- `object file '<x>' is ambiguous` — more than one `.o` with that name exists;
  give a more specific (relative or absolute) path.
- `byte_identical comparison FAILED` — the function diverged from the
  reference; the offending instruction index and both byte sequences are
  printed.
- `within_noise comparison FAILED` — either mnemonics differ (compiler chose a
  different instruction) or operand classes differ (e.g. a register operand
  became a memory operand). The offending index is printed.
- `CHECK FAILED: zero coverage` — every declared tuple was skipped; the run
  verified nothing (see *Zero-coverage guard*). Exit 1.
- `CHECK FAILED: zero codegen coverage for kernel(s): <names>` — the named
  kernels' declared tuples were all skipped while other kernels were checked;
  only the named kernels' tuples are listed as removable. Exit 1.

## Adding a tuple, end-to-end

1. Confirm both impls are emitted standalone (their symbols appear via `nm` on
   the relevant `.o`).
2. Call `ADD_CODEGEN_EQUIVALENCE_TUPLE` in `lib/CMakeLists.txt` naming the two
   symbols, their object files, and the criterion.
3. Rebuild. Success prints `codegen-equivalence: ok (N tuples checked)`;
   a regression halts the build with the diff.

## Running the script's own tests

`cmake/test_check_framework_codegen.py` is an informal standalone suite (not
wired into ctest) covering manifest parsing, whole-function extraction against
a real-toolchain fixture, and both comparison criteria:

```bash
python3 cmake/test_check_framework_codegen.py
```

`check_framework_codegen.py --manifest <m> --list-only` prints the declared
tuple ids without disassembling, useful for inspecting a generated manifest.

## Limitations

- If any tuple **errors** (a missing symbol or `.o`), the harness exits 2 and
  reports only the errors — accumulated codegen **failures** from other tuples
  are not shown until the erroring entry is fixed. Fix manifest errors first.
- `MACHINE_O` is resolved by bare-name recursive search under the build tree,
  which assumes the object name is unique there. If a future build compiles the
  same source into more than one OBJECT-library dir, give `MACHINE_O` as a path
  relative to `build/lib/CMakeFiles` (or absolute) to disambiguate.
- `within_noise` operand-class normalization is x86_64-validated. Comparing on
  another architecture requires extending the register table in
  `_operand_class`.

## Cross-compiler robustness (mtibbits/volk#145)

The harness runs across the full CI compiler matrix (gcc-11..14, clang-14..19,
macOS clang, static builds, with either `llvm-objdump` or GNU `objdump`). It
normalizes away codegen *noise* that differs by compiler but is not a real
equivalence violation, while still hard-failing on a genuine divergence.

**Stripped / tolerated (not a failure):**

- **Trailing alignment NOPs** of any encoding, including the multi-byte forms
  clang-15+ and gcc-11/14 emit after the function's final terminator
  (`66 2e 0f 1f 84 00 00 00 00 00  nopw %cs:(%rax,%rax)`). These pad the *next*
  function's alignment and belong to no function. Some objdump variants render
  such a line with only a single tab between the byte column and the mnemonic,
  which the primary line regex cannot parse; a padding-only fallback regex
  (`_padding_line`) handles that form. **Mid-body** alignment NOPs (e.g. before
  a hot loop) are *preserved* — only the trailing run is stripped.
- **Trailing `data16` padding** — how GNU objdump renders a multi-byte NOP pad
  (`data16 cs nopw …`); treated as padding like the NOP family.
- **Symbol not emitted standalone** (e.g. macOS clang inlines the impl): there
  is nothing to compare, so the tuple is **skipped with a warning** and the
  build stays green — unless the skip leaves a kernel (or the whole run) with
  zero checked tuples, which the zero-coverage guard turns into a hard failure
  (exit 1). The summary line reports `N skipped`. (Exception: a tuple
  that declares `REQUIRE_STANDALONE` hard-fails instead — see *Require-standalone
  assertion*.)

**Still a hard failure:**

- A genuine post-normalization divergence (differing mnemonic sequence or
  operand class) → `CHECK FAILED`, exit 1.
- A missing/ambiguous `.o`, malformed manifest, an unparsable *non-padding*
  in-body line, or an empty function body → `CHECK ERROR`, exit 2.
- Zero coverage — every declared tuple skipped, or one kernel's declared
  tuples all skipped (see *Zero-coverage guard*) → `CHECK FAILED`, exit 1.

## See also

- the dispatch-table integrity check (mtibbits/volk#58, PR #59; hardened by
  #132 fail-closed parse guards and #166 active-machine scoping) — the
  integrity check this harness is modeled on.
- the fusion-framework epic (mtibbits/volk#77) and its design notes for the
  C/C++ integration approaches (A vs B) that inform whether a given impl pair
  should be compared with `byte_identical` or `within_noise`.
