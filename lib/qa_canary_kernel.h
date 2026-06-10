/* -*- c++ -*- */
/*
 * Copyright 2026 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*
 * Test-only planted "kernels" for the output-canary negative control (#89,
 * child of epic #85). These are NEVER registered in the production kernel
 * list (init_test_list) and never run by the default qa suite; they exist
 * solely to exercise run_volk_canary_test's over-run detection.
 *
 * Each has a 1-float-in / 1-float-out map signature (matching the volk_fn_2arg
 * call shape the qa harness uses: out=buffs[0], in=buffs[1]) and deliberately
 * differs only in how it treats the region past num_points:
 *
 *   volk_32f_canaryok_32f       writes exactly [0,num_points) -> MUST PASS the
 *                               canary (proves it does not over-report).
 *   volk_32f_canaryonepast_32f  writes [0,num_points) then out[num_points]
 *                               (one element past) -> MUST trip the GUARD check
 *                               (over-run) in ANY build. The write lands in the
 *                               trailing sentinel guard (still inside the
 *                               own-malloc'd allocation), so it is SAFE in a
 *                               non-ASan build and is best-effort under ASan
 *                               (in-allocation, so ASan does not trip on it --
 *                               the canary is authoritative).
 *   volk_32f_canaryunwritten_32f writes only [0,num_points-1), leaving the last
 *                               in-bounds element untouched -> MUST trip the
 *                               UNWRITTEN check (the in-bounds gap detector that
 *                               ASan cannot provide). Proves that detector works
 *                               on a genuine map kernel.
 *   volk_32f_canaryfarpast_32f  writes [0,num_points) then a contiguous run
 *                               out[num_points .. num_points+K) far enough to
 *                               step past the whole guarded allocation into the
 *                               ASan redzone -> heap-buffer-overflow. This is
 *                               UNDEFINED BEHAVIOUR in a non-ASan build, so it
 *                               is run ONLY under the ASan demo path
 *                               (HARNESS_CANARY_ASAN_DEMO); it demonstrates that
 *                               the guarded path's own-malloc buffers are
 *                               bracketed by ASan redzones (acceptance #89-2).
 *   volk_32f_inputscribble_32f  copies [0,num_points) to the output correctly,
 *                               then writes one element of its INPUT buffer
 *                               (a bitwise complement of in[0]) -> MUST trip the
 *                               input-immutability check (#90). A correct
 *                               out-of-place kernel treats its input as const;
 *                               this one violates that, proving the detector
 *                               fires. Guarded by num_points > 0 (safe at vlen 0).
 *   volk_32f_misalignedfault_32f copies in -> out but performs an ALIGNED 16-byte
 *                               load on its input when num_points >= 4 -- the
 *                               movaps-in-a-_u_-kernel defect class (#91). On a
 *                               misaligned input it raises SIGSEGV (real aligned
 *                               load under __SSE2__; raise(SIGSEGV) fallback
 *                               elsewhere); on aligned input it runs correctly.
 */

#ifndef INCLUDED_VOLK_QA_CANARY_KERNEL_H
#define INCLUDED_VOLK_QA_CANARY_KERNEL_H

#include <volk/volk.h> // for volk_func_desc_t

// Number of elements the far-past planted kernel writes beyond num_points. Must
// exceed the canary's maximum trailing-guard span (guard_bytes + alignment =
// 2*alignment + 64 bytes; <= 192 B at VOLK's widest alignment of 64) so the
// contiguous over-write steps out of the allocation into the ASan redzone. 64
// floats = 260 B clears that today; if a future wider-alignment arch broke this,
// the ASan demo would NOT abort and the driver's explicit "ASan did not abort"
// WARNING + exit 2 surfaces it (it fails loud, never silently).
#define VOLK_CANARY_FAR_PAST_ELEMENTS 64u

// Map-signature planted impls: void(out, in, num_points, arch). Called through
// the volk_fn_2arg cast in run_volk_canary_test.
void volk_32f_canaryok_32f(void* out,
                           void* in,
                           unsigned int num_points,
                           const char* arch);
void volk_32f_canaryonepast_32f(void* out,
                                void* in,
                                unsigned int num_points,
                                const char* arch);
void volk_32f_canaryunwritten_32f(void* out,
                                  void* in,
                                  unsigned int num_points,
                                  const char* arch);
void volk_32f_canaryfarpast_32f(void* out,
                                void* in,
                                unsigned int num_points,
                                const char* arch);
// #90 input-immutability negative control: writes one element of its INPUT.
void volk_32f_inputscribble_32f(void* out,
                                void* in,
                                unsigned int num_points,
                                const char* arch);
// #91 misaligned negative control: aligned load inside an "unaligned" impl.
void volk_32f_misalignedfault_32f(void* out,
                                  void* in,
                                  unsigned int num_points,
                                  const char* arch);

// A synthetic single-impl ("generic") descriptor so the planted kernels flow
// through the SAME run_volk_canary_test path as real kernels. impl_alignment is
// false (unaligned): the canary supplies an aligned data region regardless.
volk_func_desc_t volk_canary_desc();

#endif /* INCLUDED_VOLK_QA_CANARY_KERNEL_H */
