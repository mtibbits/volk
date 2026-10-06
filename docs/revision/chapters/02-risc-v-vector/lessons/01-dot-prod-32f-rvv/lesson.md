---
title: The real dot product on RISC-V Vector
summary: The same kernel on a vector unit whose width the chip decides, so the loop asks for its length every trip and needs no tail loop.
file: kernels/volk/volk_32f_x2_dot_prod_32f.h
start: rvv
---
# The real dot product on RISC-V Vector

Every SIMD version lesson 1.1 taught had a register width fixed by its instruction set:
four floats for SSE and NEON, eight for AVX, sixteen for AVX-512F. The code was written for
that width, and a scalar tail loop finished whatever did not fit. The RISC-V Vector extension,
RVV, does not fix the width. The instruction set describes vector registers, and the chip
designer decides how wide they are, so the same compiled program may run on a chip with small
registers or one with large registers. A program cannot hard-code a width it does not know, so
it has to ask.

The [RVV version](@rvv) of this kernel sits at the bottom of the same file as lesson 1.1. It is
short, and it uses intrinsics from the C compiler plus one macro from VOLK's own
[RVV helper header](@rvv-include). This lesson reads it in four ideas: asking for the vector
length, eight registers acting as one, a running total that a short trip cannot spoil, and
folding the eight registers into one before adding that register's lanes into one number.

## Recap: the fixed-width answer

The [AVX2 with FMA version](@u-avx2-fma) from lesson 1.1 handles eight floats per instruction,
because that is what an AVX register holds. Its loop runs while at least eight elements remain,
and a [scalar tail loop](@fma-tail) handles the last zero to seven, one at a time, the way the
generic version would. Every fixed-width version has a tail loop like it, and the tail loop is
the price of a width chosen in advance.

## Ask the hardware

The RVV version replaces "how many elements fit in a register" with a question. At the top of
every trip through the [loop](@rvv-loop), the [length request](@rvv-vsetvl),
`__riscv_vsetvl_e32m8`, takes `n`, the number of elements still to do, and returns `vl`, how
many this trip will handle. The answer is never more than the elements left, and never more than
the register group holds. The loop then subtracts `vl` from `n` and moves both pointers on by
`vl`.

When the elements do not divide into full trips, a trip near the end is simply shorter. It may
be the last trip alone, or the last two, because the hardware is allowed to spread the remaining
work more evenly over both of them. When `num_points` is a whole multiple of what the group
holds, every trip is full and none is short. Either way the same loop body runs every time, so
there is no tail loop.

## Register groups

The `m8` in the type `vfloat32m8_t` and in the intrinsic names means that LMUL, the vector
length multiplier, is 8: eight vector registers act as one register group, so one instruction
covers eight registers' worth of floats. For example, on a chip whose vector registers each held
four floats, a group of eight would hold 32. A different chip gives a different number, and the
code never needs to know which.

Before the loop, the accumulator `vsum` is [set to zero across the whole group](@rvv-zero):
`__riscv_vfmv_v_f_f32m8` writes 0 into every lane, and its length comes from
`__riscv_vsetvlmax_e32m8`, the group's full length, rather than from any trip. That matters at
the end, because the fold adds every lane of the group, including lanes that no trip ever
writes when there are fewer elements than the group holds. Each trip then
[loads `vl` floats from each input](@rvv-load) with `__riscv_vle32_v_f32m8`.

## The running total and `_tu`

The [multiply-accumulate](@rvv-fmacc), `__riscv_vfmacc_tu`, multiplies the two loads lane by
lane and adds the products into `vsum`. It is a fused multiply-add, with one rounding, like
`_mm256_fmadd_ps` in lesson 1.1. Like the AVX2 with FMA version in the recap, this loop sends
every multiply-add through one accumulator, so each one waits for the one before it; whether the
register group hides that wait depends on the hardware.

On a short trip only the first `vl` lanes are updated. The lanes past `vl` still hold what
earlier trips left there: partial sums, or the starting zero in lanes no trip has reached, as
when one short trip is the whole loop. The fold at the end adds every one of those lanes. The
`_tu` suffix means "tail undisturbed": those lanes keep their old values. Without it the
hardware would be free to overwrite them, and the result could be wrong. In RVV, "tail" means
the lanes of a register past `vl`. It is not lesson 1.1's tail loop, the scalar loop for
leftover elements, which this version does not have.

## Folding the group

After the loop, the [`RISCV_SHRINK8` call](@rvv-shrink) turns the eight-register group into one
register. `RISCV_SHRINK8` is a macro defined in `include/volk/volk_rvv_intrinsics.h`. A lesson
shows one file, so this lesson cannot show that header; the diagram below shows what the macro
does instead.

![The fold from eight slices down to one register is what the call RISCV_SHRINK8(vfadd, f, 32, vsum) expands to; the last two steps are the kernel's own code. The macro's text is in include/volk/volk_rvv_intrinsics.h, lines 33-48 at commit 1c638a0.](diagrams/shrink8.dot)

The macro takes the eight one-register slices of the group with `__riscv_vget_f32m1` and adds
adjacent pairs with `__riscv_vfadd`: slices 0 and 1, 2 and 3, 4 and 5, 6 and 7. It then adds
those four sums in pairs, and then the last pair. That is seven adds, each sized by
`__riscv_vsetvlmax_e32m1` to cover one whole register, and it leaves one register whose lanes
still hold separate partial sums.

## The last step

The [end of the function](@rvv-reduce) turns that register into one float. The `vl` declared after
the loop is a new variable, set by `__riscv_vsetvlmax_e32m1`: one register's full length, not a
trip count. `__riscv_vfredusum` adds every lane of the register, plus a starting zero that
`__riscv_vfmv_s_f_f32m1` places in lane 0 of a second register, and puts the sum in lane 0.
`__riscv_vfmv_f` copies lane 0 out as a float, and that is the result.

This version adds the products in a different order from the generic version in three places:
lane by lane in the loop, pair by pair in the `RISCV_SHRINK8` fold, and in the final
`__riscv_vfredusum`, whose "u" means unordered: the hardware may add the lanes in any order. The
products can round differently too, because `__riscv_vfmacc_tu` rounds once, while how the
generic version's separate multiply and add round depends on the compiler. So the last bits of
the result can differ from the generic version's, as they do for the SIMD versions in lesson 1.1.

```quiz
questions:
  - q: What does the vector-length request at the top of each trip return, and why does that remove the tail loop?
    choices:
      - The register group's full length every time, so the loop stops early and scalar code finishes the rest
      - A count no more than the elements left and no more than the group holds, so a trip near the end can simply be shorter
      - The number of elements left, rounded up to a whole register, so the extra lanes read safely past the end
    answer: 1
  - q: On a short trip, what could go wrong if the multiply-accumulate did not keep its tail undisturbed?
    choices:
      - The loads would read past the end of the input
      - The loop would run one trip too many
      - Lanes past the trip's length could lose the partial sums they already hold
    answer: 2
  - q: What does the fold of the eight-register group leave behind?
    choices:
      - One register, whose lanes still need adding into one number
      - One float, ready to store as the result
      - Two registers, one holding the even lanes and one the odd lanes
    answer: 0
```
