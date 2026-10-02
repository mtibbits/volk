# Curriculum roadmap: VOLK by example

This file plans the chapters of the [reVision lessons](https://mtibbits.github.io/volk/learn/),
in the order a student meets them. Each chapter teaches SIMD by reading real VOLK kernels.
For each chapter it says what the student can do afterwards, which ideas the chapter
introduces, which kernels carry those ideas, and what it assumes from earlier chapters.

The lessons the site actually builds are listed in `docs/revision/curriculum.yaml`. That
file is what the reVision build reads; this one is a plan for the people writing lessons.

## Chapter 1: SIMD by example

Status: Shipped ([#235](https://github.com/mtibbits/volk/issues/235)).

**After this chapter you can…** open a VOLK kernel header and follow one calculation from
the plain C loop up through SSE, AVX, AVX-512 and Arm NEON. You can say how many numbers
each instruction works on, why the fast versions keep several running totals instead of
one, how those totals become a single answer at the end, what the short loop after the
main loop is for, and how VOLK picks which version runs on your machine.

**Ideas introduced**

- Lanes, and the ladder of x86 instruction sets from SSE to AVX-512, beside Arm NEON.
- Several running totals (accumulators), so the multiply-add units never wait.
- Horizontal reduction: adding the lanes of a register into one number.
- The tail loop that handles the elements left over after the last full register.
- Fused multiply-add (FMA).
- Run-time dispatch: choosing the best version for the machine the program runs on.
- Complex multiplication by duplicating and swapping the real and imaginary parts, then
  one add-subtract instruction (addsub, or fmaddsub with FMA).
- NEON's de-interleaving load, which separates real and imaginary parts as it loads.

**Kernels**

- `volk_32f_x2_dot_prod_32f.h`: the real dot product, from `generic` up the ladder from
  `u_sse` to `u_avx512f`, plus `neon` and `neonv8`.
- `volk_32fc_x2_dot_prod_32fc.h`: the complex dot product. `u_sse3` duplicates with
  `_mm_moveldup_ps` and `_mm_movehdup_ps`, swaps with `_mm_shuffle_ps` and finishes with
  `_mm_addsub_ps`; `u_avx_fma` uses `_mm256_fmaddsub_ps`; `neon` loads with `vld2q_f32`.

**Assumes**

C loops, arrays and pointers; what floating-point and complex numbers are.

## Chapter 2: One loop for every vector length (RISC-V Vector)

Status: Planned: epic [#238](https://github.com/mtibbits/volk/issues/238).

**After this chapter you can…** read a RISC-V Vector (RVV) kernel and explain how one loop
runs unchanged on machines with short or long vector registers. Each time round, the loop
asks the hardware how many elements it may take, so there is no separate loop for
leftovers. You can also explain two ways an RVV kernel splits interleaved complex numbers
into real and imaginary parts, and compare them with the Arm load from chapter 1.

**Ideas introduced**

- Vector-length-agnostic strip-mining: the loop sets the vector length on every trip
  (vsetvl), so there is no tail loop.
- Register grouping (LMUL): one instruction works on a group of registers, written m8 or
  m2 in the intrinsic names.
- Tail-undisturbed accumulation (the _tu forms): a short last trip leaves the unused part
  of the running total alone.
- Reduction by folding a register group in half, and in half again, then one reduce
  instruction.
- Splitting complex numbers by narrowing shifts (vnsrl) versus segment loads (vlseg2).

**Kernels**

- `volk_32f_x2_dot_prod_32f.h`, the same header as chapter 1's first lesson, now its
  `rvv` version: `__riscv_vsetvl_e32m8` on every trip, `__riscv_vfmacc_tu` for the running
  total, `RISCV_SHRINK8` to fold the register group and `__riscv_vfredusum` for the final
  sum. The macro is defined in `include/volk/volk_rvv_intrinsics.h`, so a lesson can
  describe its body but not show it.
- `volk_32fc_x2_conjugate_dot_prod_32fc.h` (candidate, pending
  [#240](https://github.com/mtibbits/volk/issues/240)'s kernel choice; alternative:
  `volk_32fc_32f_dot_prod_32fc.h`). `rvv` loads each complex number as one 64-bit value
  and splits it with `__riscv_vnsrl` by 0 and by 32 bits; `rvvseg` uses the segment load
  `__riscv_vlseg2e32_v_f32m2x2`. Both flip the sign of the second input's imaginary part
  with `__riscv_vfneg` for the conjugate, keep running totals with `__riscv_vfadd_tu` and
  fold them with `RISCV_SHRINK2`. This is a kernel chapter 1 did not cover.

**Assumes**

Chapter 1 (both lessons).

## Chapter 3: Memory: aligned and unaligned

Status: Not yet planned.

**After this chapter you can…** explain why most x86 kernels in VOLK come in pairs, an
aligned version and an unaligned one, and what the aligned version may assume about
addresses. You can explain how VOLK checks every pointer at run time to choose between
them, and how `volk_malloc` makes buffers that qualify. You can also read the simplest
kernel shape: one output per input, no running total, and one number copied into every
lane.

**Ideas introduced**

- Element-wise loops, with no reduction.
- Aligned and unaligned loads and stores: `_mm_load_ps` against `_mm_loadu_ps`, and
  `_mm_store_ps` against `_mm_storeu_ps`.
- Dispatch on alignment: the dispatcher combines every pointer argument with a bitwise OR,
  asks `volk_is_aligned` once, and calls the aligned or the unaligned version. This code
  is generated from `tmpl/volk_dynamic_dispatch.tmpl.c`, not written in the kernel.
- Why NEON and RVV need only one version: their loads do not care about alignment.
- Broadcasting: copying one number into every lane.

**Kernels**

- `volk_32f_x2_multiply_32f.h`: multiplies two arrays element by element. The pairs
  `u_sse` and `a_sse`, `u_avx` and `a_avx`, and `u_avx512f` and `a_avx512f` differ only in
  their loads and stores; `neon` appears once; `rvv` is the chapter 2 loop without a
  running total.
- `volk_32f_s32f_multiply_32f.h`: multiplies an array by one number. `u_sse` copies it
  into every lane with `_mm_set_ps1`, `u_avx` with `_mm256_set1_ps` and `neonv8` with
  `vdupq_n_f32`; `u_neon` uses `vmulq_n_f32`, which takes the number directly; `rvv`
  passes it straight to `__riscv_vfmul`.

**Assumes**

Chapter 1 (lanes, the tail loop, dispatch); chapter 2 (the RVV loop).

## Chapter 4: Integers: saturation and narrowing

Status: Not yet planned.

**After this chapter you can…** explain what happens when an integer sum does not fit:
plain adds wrap around to the other end of the range, and saturating adds stop at the
largest or smallest value. You can find the one instruction that does this on each
architecture, read the plain C version that detects overflow without an if-statement, and
follow a float-to-16-bit conversion that clamps, rounds, and squeezes 32-bit lanes into
16-bit lanes.

**Ideas introduced**

- Integer lanes: 8-bit and 16-bit numbers, many more to a register than floats.
- Wrap-around versus saturation, for signed and unsigned numbers.
- A branch-free overflow test in plain C.
- Clamping with min and max.
- Saturating narrowing (packing): squeezing wide lanes into narrow ones.
- AVX2 packs that work on each 128-bit half separately, and the permute across the halves
  that puts the results back in order.

**Kernels**

- `volk_16i_x2_add_saturated_16i.h`: `generic` detects overflow with exclusive-ors and a
  shift, then picks the result with masks instead of an if-statement; `u_sse2`, `u_avx2`
  and `u_avx512bw` use `_mm_adds_epi16`, `_mm256_adds_epi16` and `_mm512_adds_epi16`;
  `neon` uses `vqaddq_s16`; `rvv` uses `__riscv_vsadd`.
- `volk_8u_x2_add_saturated_8u.h`: the unsigned version, with `_mm_adds_epu8` in
  `u_sse2`, `vqaddq_u8` in `neon` and `__riscv_vsaddu` in `rvv`.
- `volk_32f_s32f_convert_16i.h`: scales floats and converts them to 16-bit integers.
  `u_sse2` clamps with `_mm_max_ps` and `_mm_min_ps`, converts with `_mm_cvtps_epi32` and
  narrows with `_mm_packs_epi32`; `u_avx2` adds `_mm256_permute4x64_epi64` to fix the order
  across the halves; `u_avx512` narrows with saturation in one step with
  `_mm512_cvtsepi32_epi16`; `neon` uses `vqmovn_s32`; `rvv` uses `__riscv_vfncvt_x`.

**Assumes**

Chapter 1 (lanes); chapter 3 (the element-wise shape and the aligned and unaligned pairs).

## Chapter 5: Comparisons and masks: choosing without branching

Status: Not yet planned.

**After this chapter you can…** rewrite an if-statement inside a loop as a comparison that
makes a mask, followed by a select that uses it, so every lane does the same work. You can
read the forms VOLK uses on each architecture, and follow a kernel that finds the largest
value in an array and its position, tracking both in every lane until the end.

**Ideas introduced**

- A comparison that produces a mask: all ones or all zeros in each lane.
- Selecting with and, and-not and or, versus a single blend instruction.
- AVX-512 mask registers.
- NEON's bit-select.
- RVV masks, merges, and a vector that holds each lane's number.
- Branch-free scalar code.
- Carrying an index through a reduction.

**Kernels**

- `volk_32f_binary_slicer_8i.h`: turns each float into 1 or 0 by its sign. `generic` uses
  an if/else, while `generic_branchless` stores the comparison's result directly; `u_avx2`
  compares with `_mm256_cmp_ps`, converts and shifts the mask down to 0 or 1, narrows with
  `_mm256_packs_epi32` and `_mm256_packs_epi16`, and restores the order with
  `_mm256_permute4x64_epi64` and `_mm256_shuffle_epi8`.
- `volk_32f_index_max_32u.h`: finds the largest value and its position. `u_sse` compares
  with `_mm_cmpgt_ps` and selects with `_mm_and_ps`, `_mm_andnot_ps` and `_mm_or_ps`;
  `u_sse4_1` and `u_avx` select with `_mm_blendv_ps` and `_mm256_blendv_ps`; `u_avx512f`
  uses a `__mmask16` mask register with `_mm512_mask_blend_ps`; `neon` compares with
  `vcleq_f32` and selects with `vandq_u32`, `vbicq_u32` and `vorrq_u32`; `neonv8` compares
  with `vcgtq_f32` and uses the bit-select `vbslq_u32`; `rvv` compares with
  `__riscv_vmfgt`, merges with `__riscv_vmerge_tu` and numbers the lanes with
  `__riscv_vid_v_u32m4`, then at the end finds the maximum with `__riscv_vfredmax` and the
  lanes that hold it with `__riscv_vmfeq`.

**Assumes**

Chapter 1 (reduction); chapter 4 (packing, and the permute across halves).

## Chapter 6: Rearranging data: shuffles and permutes

Status: Not yet planned.

**After this chapter you can…** read code whose only job is to move numbers between lanes:
splitting interleaved complex samples into separate real and imaginary arrays, joining
them back, and swapping the two bytes of every 16-bit sample. You can explain why the same
move takes one instruction on one architecture and several on another, including why AVX
code often needs an extra step to cross between the two halves of a register.

**Ideas introduced**

- Shuffles with a fixed pattern written into the instruction (the `_MM_SHUFFLE` macro).
- Unpacking and interleaving.
- Permutes across the two halves of an AVX register.
- Permutes driven by a vector of indices.
- De-interleaving loads and interleaving stores.
- Byte shuffles and table lookups.
- Dedicated instructions in newer instruction-set levels.

**Kernels**

- `volk_32fc_deinterleave_32f_x2.h`: splits complex samples into a real array and an
  imaginary array. `a_sse` uses `_mm_shuffle_ps` with `_MM_SHUFFLE` patterns that pick the
  even-numbered and the odd-numbered elements; `u_avx` adds `_mm256_permute2f128_ps` to
  `_mm256_shuffle_ps`; `u_avx512f` uses `_mm512_permutexvar_ps`; `neon` uses `vld2q_f32`;
  `rvv` uses `__riscv_vnsrl`, and `rvvseg` uses `__riscv_vlseg2e32_v_u32m4x2`.
- `volk_32f_x2_interleave_32fc.h`: the reverse, joining two arrays into complex samples.
  `a_sse` uses `_mm_unpacklo_ps` and `_mm_unpackhi_ps`; `u_avx` adds
  `_mm256_permute2f128_ps`; `neon` uses `vst2q_f32`; `rvv` builds each 64-bit pair with
  widening arithmetic (`__riscv_vwaddu_vv` and `__riscv_vwmaccu`), and `rvvseg` uses the
  segment store `__riscv_vsseg2e32`.
- `volk_16u_byteswap.h`: swaps the two bytes of every 16-bit value. `u_sse2` shifts and
  combines with `_mm_slli_epi16`, `_mm_srli_epi16` and `_mm_or_si128`; `u_avx2` uses the
  byte shuffle `_mm256_shuffle_epi8`; `neon_table` uses a table lookup (`vtbl4_u8`);
  `neonv8` uses `vrev16q_u8`; `rvv` uses `__riscv_vrgather` with indices from
  `RISCV_PERM8`; `rva23` uses `__riscv_vrev8`.

**Assumes**

Chapter 1 (the complex swizzle and `vld2q_f32`); chapter 2 (splitting by narrowing shifts
and by segment loads); chapter 4 (the permute across halves).

## Chapter 7: Math functions from bits and polynomials

Status: Not yet planned.

**After this chapter you can…** explain how a kernel computes exp, log2 or sine with only
multiplies, adds and bit operations. It shrinks the input into a small range, approximates
the function there with a short polynomial, and rebuilds the answer, sometimes by writing
the power of two straight into the float's exponent bits. You can say what accuracy this
gives and how VOLK's tests judge it.

**Ideas introduced**

- Range reduction, with a constant split into a large and a small part for accuracy.
- Polynomial approximation as nested multiply-adds (Horner's rule), with FMA where the
  machine has it.
- Reading and writing the bits of a float as an integer.
- Accuracy against speed, and the tolerances VOLK's tests use.

**Kernels**

- `volk_32f_exp_32f.h`: `u_sse2` and `u_avx2` scale the input by log2(e), reduce it with
  the split constants `exp_C1` and `exp_C2`, evaluate the polynomial `exp_p0` to `exp_p5`
  written out in the same file, and build the power of two with `_mm_cvttps_epi32`, an
  added bias, `_mm_slli_epi32` by 23 bits and `_mm_castsi128_ps`. Everything is in one
  file, which suits a lesson that shows one file.
- `volk_32f_log2_32f.h`: `u_sse4_1` takes the exponent out of the float's bits with
  `_mm_castps_si128`, a mask, `_mm_srli_epi32` by 23 bits and a subtraction of the bias,
  then approximates the rest with `_mm_log2_poly_sse` from
  `include/volk/volk_sse_intrinsics.h`; `u_avx2` does the same with the 256-bit forms and
  `_mm256_log2_poly_avx2` from `include/volk/volk_avx2_intrinsics.h`. Its accuracy is
  under adjudication in [#173](https://github.com/mtibbits/volk/issues/173).
- `volk_32f_sin_32f.h` (optional: teach the x86 path; RVV range-reduction bug
  [#150](https://github.com/mtibbits/volk/issues/150)): `u_avx2_fma` finds the nearest
  multiple of π/2 with `_mm256_round_ps`, then subtracts it in two `_mm256_fnmadd_ps`
  steps, one for the high part of π/2 and one for the low part. Its polynomial,
  `_mm256_sin_poly_avx2_fma`, is in `include/volk/volk_avx2_fma_intrinsics.h`.

**Assumes**

Chapter 1 (FMA); chapter 4 (converting between floats and integers); chapter 5 (select).

## Kernels considered and not chosen

- `volk_32f_x2_add_32f.h`: the same shape as the chapter 3 kernels, but a rewrite is
  planned in [#77](https://github.com/mtibbits/volk/issues/77),
  [#79](https://github.com/mtibbits/volk/issues/79) and
  [#80](https://github.com/mtibbits/volk/issues/80).
- `volk_16ic_x2_dot_prod_16ic.h`: integer saturation inside a reduction, but its
  saturation behaviour is under adjudication in
  [#220](https://github.com/mtibbits/volk/issues/220).

## Where the RVV material sits

RISC-V Vector (RVV) material is chapter 2. Epic
[#238](https://github.com/mtibbits/volk/issues/238) assigns the chapter-placement decision
to its design spec, [#240](https://github.com/mtibbits/volk/issues/240), which has not
decided it yet; this placement is a proposal for that decision to follow. If
[#240](https://github.com/mtibbits/volk/issues/240) decides otherwise,
[#238](https://github.com/mtibbits/volk/issues/238) updates this file to match.

## How this roadmap is maintained

This is a plan, not a commitment. Each chapter is delivered by its own epic. That epic
links to this file and, when it completes, updates its chapter's entry here to match what
shipped: title, outcome, ideas and kernels. A design spec that decides differently from
this file wins, and the epic updates this file to match. Kernel names here were checked
against `kernels/volk/` on `dev/all-prs` when this file last changed; each epic re-checks
the names for its own chapter. The reVision build does not read this file.
