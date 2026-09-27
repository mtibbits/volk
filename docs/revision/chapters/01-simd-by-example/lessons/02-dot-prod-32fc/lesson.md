---
title: The complex dot product
summary: Interleaved parts, one swizzle, one sign flip. What changes when the samples are complex.
file: kernels/volk/volk_32fc_x2_dot_prod_32fc.h
start: generic
---
# The complex dot product

`volk_32fc_x2_dot_prod_32fc` is the [real dot product](../dot-prod-32f/) for complex
samples, which is what a radio actually produces. Each element is a pair of floats, real
and imaginary, stored side by side in memory. Multiplying two of them needs four products
and a subtraction:

![Complex multiply](diagrams/complex-mul.dot)

Everything in this file is about doing that with instructions that only know how to
multiply lanes straight across. The
[Doxygen page](https://mtibbits.github.io/volk/volk__32fc__x2__dot__prod__32fc_8h.html)
has the formal interface.

## The reference

The [generic version](@generic) unrolls its [loop](@generic-loop) two complex products at
a time and writes the four multiplies out by hand: real times real minus imaginary times
imaginary for the real part, the two cross products added for the imaginary part. As
before, this is the answer every other version must reproduce.

## SSE3: duplicate, swizzle, and flip a sign

An SSE register holds two complex numbers as `ar, ai, br, bi`. Multiplying it lane by lane
against `cr, ci, dr, di` gives `ar·cr, ai·ci, ...`, which is only half the products, in the
wrong places. The [SSE3 version](@u-sse3) fixes that in three moves.

First it [duplicates](@sse3-dup) the taps two ways: `_mm_moveldup_ps` copies the real
parts into both lanes of each pair, `cr, cr, dr, dr`, and `_mm_movehdup_ps` copies the
imaginary parts, `ci, ci, di, di`. Multiplying the samples by the first gives `ar·cr, ai·cr`,
both products that involve `cr`.

Second it [swizzles](@sse3-swizzle) the samples with `_mm_shuffle_ps` and the constant
`0xB1`, which swaps each pair to `ai, ar, bi, br`. Multiplying that by the imaginary
duplicate gives `ai·ci, ar·ci`. Now every one of the four products exists, two per register.

Third, [`_mm_addsub_ps`](@sse3-addsub) subtracts in the even lanes and adds in the odd
ones, producing `ar·cr − ai·ci` and `ai·cr + ar·ci` in one instruction: the real and
imaginary parts of the product, already interleaved the way the output wants them.

## AVX and AVX with FMA

The [AVX version](@u-avx) is the same three moves at eight floats, four complex numbers,
per register, with one wrinkle: it processes pairs of complex numbers, so an odd
`num_points` leaves one element for a [one-item tail](@avx-odd-tail). The
[AVX with FMA version](@u-avx-fma) fuses the last multiply with the add-subtract using
[`_mm256_fmaddsub_ps`](@avx-fmaddsub), the complex-arithmetic cousin of the
`_mm256_fmadd_ps` you met in lesson 1.

## NEON: load the parts apart

Arm avoids the swizzle entirely. The [NEON version](@neon) uses [`vld2q_f32`](@neon-ld2),
a de-interleaving load that puts four real parts in one register and four imaginary parts
in another. With the parts separated, the four products are four plain lane-wise
multiplies and the sign flip is a multiply-subtract: the [NEONv8 version](@neonv8) uses
[`vfmsq_f32`](@neonv8-fms) to subtract `ai·bi` from the running real sum. Three tuned
NEON siblings in the Variants panel differ in unrolling and accumulator count; they exist
because the dispatcher measures rather than guesses.

## Everything else

The aligned twins mirror lesson 1. The two RISC-V variants differ in how they load
interleaved data, plain loads versus a segmented load that does what `vld2q_f32` does.

```quiz
questions:
  - q: What does `_mm_addsub_ps` compute across a register holding `p0, p1, p2, p3`?
    choices:
      - "p0+p1, p2+p3"
      - "subtract in even lanes, add in odd lanes"
      - "the horizontal sum of all four lanes"
    answer: 1
  - q: What does `_mm_shuffle_ps(x, x, 0xB1)` do to `ar, ai, br, bi`?
    choices:
      - "Reverses the register to bi, br, ai, ar"
      - "Swaps each pair to ai, ar, bi, br"
      - "Duplicates the real parts"
    answer: 1
  - q: Why does the NEON version need no shuffle?
    choices:
      - "NEON registers are wider"
      - "vld2q_f32 loads real and imaginary parts into separate registers"
      - "NEON has a complex multiply instruction"
    answer: 1
```
