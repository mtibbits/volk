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
  `.o` than declared.
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

## See also

- the dispatch-table integrity check (mtibbits/volk#58, PR #59) — the
  integrity check this harness is modeled on.
- the fusion-framework epic (mtibbits/volk#77) and its design notes for the
  C/C++ integration approaches (A vs B) that inform whether a given impl pair
  should be compared with `byte_identical` or `within_noise`.
