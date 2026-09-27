---
title: The real dot product
summary: One loop, a ladder of instruction sets. The kernel every FIR filter is built on.
file: kernels/volk/volk_32f_x2_dot_prod_32f.h
start: generic
---
# The real dot product

`volk_32f_x2_dot_prod_32f` multiplies two float vectors element by element and adds
the products into one number. That is a dot product, and in signal processing it is
the inner loop of a FIR filter: `input` holds samples, `taps` holds coefficients, and
the result is one filtered output sample. Filters call this millions of times a second,
which is why VOLK ships fifteen hand-written versions of a three-line loop. This lesson
walks up that ladder. Every rung computes the same answer to within floating-point
rounding: the SIMD versions add the products in a different order, so the last bits
differ, and VOLK's tests compare the versions to a tolerance rather than bit for bit.
What changes from rung to rung is the number of multiplications per instruction.

The reference page for this kernel is in the
[VOLK Doxygen documentation](https://mtibbits.github.io/volk/volk__32f__x2__dot__prod__32f_8h.html).

## The reference: one product at a time

The [generic version](@generic) is the definition. Its [loop](@generic-loop) takes one
float from each vector, multiplies them, and adds the product to a running total. Every
other version in this file must reproduce that total, and VOLK's test suite checks that
they do. When you are lost in intrinsics, come back here: this is what they mean.

## SSE: four lanes and four accumulators

The [SSE version](@u-sse) processes sixteen floats per trip through its [main loop](@sse-loop).
An SSE register holds four floats, so the loop loads four registers from each vector,
multiplies them pairwise with `_mm_mul_ps`, and adds each product register into one of
[four accumulators](@sse-lanes) with `_mm_add_ps`. Why four accumulators rather than
one? Each `_mm_add_ps` must wait for the previous add into the same register to finish.
Four independent accumulators let the CPU overlap four adds, so the loop runs closer to
the speed of the loads.

Two ideas introduced here recur on every rung.

**The horizontal reduction.** After the loop, the four accumulators hold sixteen partial
sums in sixteen lanes. [Combining them](@sse-combine) adds the four registers together,
stores the resulting four lanes to memory, and adds those four floats one by one. SIMD is
fast lane by lane and slow across lanes, so this "lanes to one number" step is always
done once, at the end.

**The tail loop.** Sixteen does not divide every `num_points`. The
[tail loop](@sse-tail) finishes the last zero to fifteen elements the way the generic
version would. Every fixed-width SIMD rung has one; only RVV, at the bottom of the file,
sizes its final trip to fit instead.

## SSE3: a rung with nothing on it

The [SSE3 version](@u-sse3) is the SSE version with the operands of
[`_mm_add_ps` swapped](@sse3-add). It uses no SSE3 instruction at all: SSE3's
`_mm_hadd_ps` could have shortened the reduction, but the code does not reach for it. The
rung exists so the dispatcher has an SSE3 entry. Real libraries contain rungs like this;
knowing that saves you a long search for a difference that is not there.

## SSE4.1: a dot-product instruction

SSE4.1 adds `_mm_dp_ps`, which multiplies two registers and sums the products in one
instruction. The [SSE4.1 version](@u-sse4-1) uses it [four times per trip](@sse4-dp) with
masks `0xF1`, `0xF2`, `0xF4`, and `0xF8`. The high nibble `F` says "use all four lanes";
the low nibble picks which output lane receives the sum: lane 0, 1, 2, or 3. The
instruction writes zero to the lanes it was not asked for, so the four results, each in
its own lane with zeros elsewhere, can be [combined with `_mm_or_ps`](@sse4-or) rather
than added. It is elegant, and on most CPUs it is not faster: `dpps` has a long latency,
and the plain multiply-add rungs win.

## AVX: eight lanes, two accumulators

An AVX register is 256 bits wide, eight floats. The [AVX version](@u-avx) still handles
sixteen floats per trip, so it needs only two loads from each vector and
[two accumulators](@avx-lanes) instead of four. The reduction adds the two registers,
then the eight lanes; the tail loop is unchanged. Doubling the lanes halved the
bookkeeping, and that is the whole change.

## AVX2 with FMA: multiply and add in one step

The [AVX2 with FMA version](@u-avx2-fma) is the shortest x86 variant in the file, and
the one most modern x86 machines run. Inside its [loop](@fma-loop), one instruction does
the work of two: [`_mm256_fmadd_ps`](@fma-step) multiplies `aVal1` by `bVal1` and adds
the result into `dotProdVal`, with one rounding instead of two. Notice that this variant
keeps a single accumulator, so each fused multiply-add waits for the previous one; the
NEONv8 version below shows the four-accumulator alternative with the same operation.
The [reduction](@fma-reduce) stores eight lanes and adds them, and the
[tail loop](@fma-tail) mops up.

## AVX-512F: sixteen lanes

The [AVX-512F version](@u-avx512f) is the FMA version at twice the width:
[`_mm512_fmadd_ps`](@avx512-step) handles sixteen floats per instruction. On CPUs that
support it, this is the widest rung, though some CPUs lower their clock speed while
executing 512-bit instructions, which is one reason VOLK lets you measure the choice
rather than assume it.

## NEON: the same ladder on Arm

Arm's NEON registers hold four floats, like SSE. The [NEON version](@neon) keeps
[four accumulators](@neon-acc) and uses [`vmlaq_f32`](@neon-mla), a multiply-accumulate
that does the multiply and the add in one instruction but with two roundings, so it is
not fused. The [reduction](@neon-reduce) folds four lanes to two with `vadd_f32`, two to
one with `vpadd_f32`, and reads the lane out. The [NEONv8 version](@neonv8) for 64-bit
Arm swaps in the fused [`vfmaq_f32`](@neonv8-fma), the true counterpart of
`_mm256_fmadd_ps`, and reduces with a single [`vaddvq_f32`](@neonv8-reduce), an
across-vector add that older NEON lacked.

## Choosing a rung

![Default dispatch order](diagrams/dispatch.dot)

VOLK compiles every variant the compiler supports and, at run time, picks one for the
machine it is on. By default it takes the highest-ranked variant the CPU's feature flags
allow, in the order the tree above shows: the ranking comes from the order instruction
sets are listed in VOLK's architecture table, not from timing. A second rule sits under
the tree: when every pointer passed to the call is suitably aligned, the dispatcher runs
the `a_` twin of the chosen variant instead of the `u_` one. Running `volk_profile` times
every variant on the actual CPU and records the winners in a preferences file; only then
is the choice measured, which is how the SSE4.1 rung loses to plain SSE on many machines
despite its clever instruction.

## The aligned twins and RVV

Every x86 variant appears twice in this file. The `u_` versions use unaligned loads such
as `_mm_loadu_ps`; the `a_` versions use aligned loads such as `_mm_load_ps`, which
require the data to start on a 16, 32, or 64 byte boundary and were once markedly faster.
NEON, NEONv8, and RVV appear once, because their loads do not care. The Variants panel
lists them all. The RISC-V vector version, `rvv`, uses a different model in which the
hardware chooses the vector length per iteration; it is listed for completeness and gets
its own lesson later.

```quiz
questions:
  - q: How many floats does one _mm256_fmadd_ps, the FMA instruction on 256-bit registers, multiply and accumulate?
    choices: ["16", "4", "8"]
    answer: 2
  - q: Why does the SSE version keep four accumulators instead of one?
    choices:
      - Independent accumulators let the CPU overlap adds that would otherwise wait on each other
      - Four accumulators are needed to hold sixteen floats
      - The horizontal reduction requires exactly four registers
    answer: 0
  - q: What does the tail loop do?
    choices:
      - Adds the lanes of the accumulator into one float
      - Handles the last few elements when num_points is not a multiple of the vector width
      - Aligns the input pointers before the SIMD loop
    answer: 1
```
