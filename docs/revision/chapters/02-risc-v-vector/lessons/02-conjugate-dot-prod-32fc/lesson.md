---
title: The conjugate dot product on RISC-V Vector
summary: Input times the conjugate of taps, and two RVV ways to pull interleaved complex numbers apart, by shifting or by a segment load.
file: kernels/volk/volk_32fc_x2_conjugate_dot_prod_32fc.h
start: generic
---
# The conjugate dot product on RISC-V Vector

`volk_32fc_x2_conjugate_dot_prod_32fc` adds up input times the conjugate of taps over every
element. It is the inner product behind correlation and matched filtering, where a signal is
compared with a known pattern. The [generic version](@generic) says it in one
[line](@generic-loop): each input sample is multiplied by the conjugate of its tap, `lv_conj`,
and the product is added to one complex total.

This lesson reads the file's two RISC-V Vector versions. Both split interleaved complex numbers
into real and imaginary parts, in two different ways: the `rvv` version loads each complex value
as one 64-bit number and narrows it with shifts, and the `rvvseg` version uses a segment load that
separates the parts while loading. After the split both do the same conjugate and the same
multiply, so the two loads can be compared directly. The lesson assumes lesson 1.2's complex
multiply and its interleaved storage, and the [previous lesson](../dot-prod-32f-rvv/)'s
vector-length loop, register groups, `_tu` and fold-then-reduce.

## The conjugate written out

Call the input a and the taps b, as the RVV code does with its `va` and `vb`, and as the file's
own comments do. The conjugate of b flips the sign of its imaginary part, so one product is

(ar + i·ai)(br − i·bi) = (ar·br + ai·bi) + i(ai·br − ar·bi)

Lesson 1.2's plain product, in the same letters, is (ar·br − ai·bi) + i(ai·br + ar·bi). The two
terms that involve bi change sign, and nothing else changes.

Lesson 1.2 used other letters, so it is worth saying which is which. Its diagram and its SSE3
section call the taps c and d, with real part ar·cr − ai·ci, and use b for a second input
sample. Its NEONv8 paragraph follows the code of that version, where b is the taps, as here. The
file's comments write j for the imaginary unit where this lesson writes i.

## Recap: Arm's de-interleaving load

Lesson 1.2's NEON version loaded the real and imaginary parts apart with `vld2q_f32`. The
[`neon` version](@neon) in this file does the same, but it reaches the conjugate by another
route, and that route is worth seeing before the RVV one.

It [points its two pointers the other way round](@neon-ptrs): `a_ptr` at the taps and `b_ptr`
at the input. So its [two de-interleaving loads](@neon-ld2), `vld2q_f32`, put the taps in `a_val`
and the input in `b_val`, the reverse of the letters in the previous section. Each trip of its
main loop, which handles four complex values at a time, forms taps times the conjugate of input,
with the signs set by its [multiply-subtract and multiply-add pair](@neon-mla), `vmlsq_f32` and
`vmlaq_f32`, and adds that to its running totals. There is no negation anywhere in that loop. The
scalar loop for leftover elements forms the same product one element at a time, using `lv_conj`
on the input. Then the version [conjugates the total once, at the end](@neon-conj). The conjugate
of (taps times the conjugate of input) is input times the conjugate of taps, so the answer is the
same.

Neither route is a version of the other. The `neonv8` version in the same file, which the
Variants panel does not list, takes a third path and computes input times the conjugate of taps
directly.

## Split one: shift and narrow

The [RVV version](@rvv) has no de-interleaving load to call on. Its loop asks for the vector
length every trip, as in the previous lesson: `__riscv_vsetvl_e32m2` returns a count no larger
than the elements left and no larger than the register group holds. Near the end a trip may be
shorter, the last one or the last two, and when `num_points` is a whole multiple of what the
group holds, every trip is full. Each trip then
[loads each complex value as one 64-bit integer](@rvv-load) with `__riscv_vle64_v_u64m4`: a real
part and an imaginary part, side by side in memory, read together as one 64-bit number.

The [four narrowing shifts](@rvv-split) pull the halves apart. `__riscv_vnsrl` shifts each
64-bit element right and keeps the low 32 bits of the result: shifted by 0 that is the low half,
which holds the real part, and shifted by 32 it is the high half, which holds the imaginary part.
`__riscv_vreinterpret_f32m2` relabels those bits as floats without changing them. Two shifts for
the input and two for the taps give the four parts `var`, `vai`, `vbr` and `vbi`.

![Two ways to split interleaved complex values into real and imaginary parts. The shift version narrows each 64-bit element twice; the segmented version separates the parts while loading, then takes each part out.](diagrams/deinterleave.dot)

The shift route works only if the real part lands in the low half of the 64-bit element. That
holds on a little-endian RISC-V system, where the float stored first in memory becomes the low
32 bits.

The length from the 32-bit request also fits the 64-bit load. The `m2` and `m4` in the names are
register-group sizes, as in the previous lesson: a group of two registers of 32-bit elements and
a group of four registers of 64-bit elements hold the same number of elements, so one `vl`
serves both.

## Conjugate, then an ordinary multiply

With the parts apart, the [conjugate](@rvv-conj) is one instruction: `__riscv_vfneg` flips the
sign of the taps' imaginary part, `vbi`. That is the whole conjugate. From here on the code is an
ordinary complex multiply of a by the conjugated b.

The [multiply](@rvv-mul) takes two statements. `__riscv_vfmul` forms ar·br and ar·(−bi).
`__riscv_vfnmsac` subtracts ai·(−bi) from the first, giving ar·br + ai·bi, the real part.
`__riscv_vfmacc` adds ai·br to the second, giving ai·br − ar·bi, the imaginary part. Those are
the two parts of the formula above.

So in these RVV versions the conjugate costs one negation per trip. Apart from its name, the
indentation that follows the name, and that one negation, the `rvv` function is line for line
the `rvv` version of lesson 1.2's plain complex dot product.

## Two running totals, then the fold

The [running totals](@rvv-acc) work as in the previous lesson, but there are two of them. Before
the loop, `__riscv_vfmv_v_f_f32m2` sets `vsumr` to zero across the whole group, sized by
`__riscv_vsetvlmax_e32m2`, the group's full length, and `vsumi` starts as a copy of it. Each trip,
`__riscv_vfadd_tu` adds the trip's real products into `vsumr` and its imaginary products into
`vsumi`. It updates only the first `vl` lanes and leaves the rest tail undisturbed, so the lanes
past `vl` keep what earlier trips left there: partial sums, or the starting zero in lanes no trip
has reached, as when one short trip is the whole loop.

The [end of the function](@rvv-reduce) folds and reduces each total. The `vl` declared after the
loop is a new variable, one register's full length from `__riscv_vsetvlmax_e32m1`.
`RISCV_SHRINK2` is a macro from VOLK's RVV helper header, `include/volk/volk_rvv_intrinsics.h`,
and for a two-register group it is a single add: it takes the group's two one-register halves
with `__riscv_vget_f32m1` and adds them with `__riscv_vfadd`, sized by
`__riscv_vsetvlmax_e32m1`. The previous lesson's eight-register fold needed seven adds; here one
is enough. `__riscv_vfredusum` then adds every lane of each folded register, plus a starting zero
that `__riscv_vfmv_s_f_f32m1` places in lane 0 of a second register, and leaves the sum in lane 0.
`__riscv_vfmv_f` copies each lane 0 out as a float, and `lv_cmake` joins the two floats into the
complex result.

These versions add the products in a different order from the generic version in three places:
lane by lane in the loop, in the `RISCV_SHRINK2` fold, and in the final `__riscv_vfredusum`, an
unordered sum. The products can also round differently: `__riscv_vfnmsac` and `__riscv_vfmacc`
are fused multiply-adds, which round once, while how the generic version's complex multiply
rounds depends on the compiler. So the last bits of the result can differ from the generic
version's.

## Split two: the segment load

The [segmented version](@rvvseg) splits the parts while loading. Its
[two segment loads](@rvvseg-load), `__riscv_vlseg2e32_v_f32m2x2`, read the interleaved pairs and
put field 0 of each pair, the real part, in one register group and field 1, the imaginary part,
in another, returned together as one two-part value. Calling `__riscv_vget_f32m2` twice per load
[takes each part out](@rvvseg-split) as its own register group, the other side of the diagram
above. This is RVV's counterpart of Arm's `vld2q_f32` from the recap. From there on the body is
the same as the `rvv` version's, negation and all.

The line-for-line match with lesson 1.2 holds for the shift version only: lesson 1.2's segmented
version uses four-register groups and `RISCV_SHRINK4`, where this file's uses two-register groups
and `RISCV_SHRINK2`. Which of the two versions here is faster depends on the hardware and has to
be measured; this lesson makes no claim either way.

```quiz
questions:
  - q: The shift version loads each complex value as one 64-bit integer. What does shifting it right by 32 and narrowing to 32 bits keep?
    choices:
      - "The real part"
      - "The sum of the real and imaginary parts"
      - "The imaginary part"
    answer: 2
  - q: How does the segment load differ from the shift route?
    choices:
      - "It separates the real and imaginary parts while loading, so no shifts are needed"
      - "It loads each complex value as one 64-bit integer, then shifts it like the other version"
      - "It conjugates the taps while loading, so no negation is needed"
    answer: 0
  - q: Where does the RVV version apply the conjugate?
    choices:
      - "It conjugates the total once, after the loop"
      - "It negates the taps' imaginary part, then does an ordinary complex multiply"
      - "It swaps the real and imaginary parts of the input before multiplying"
    answer: 1
```
