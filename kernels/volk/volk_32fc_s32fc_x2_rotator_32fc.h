/* -*- c++ -*- */
/*
 * Copyright 2012, 2013, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_s32fc_x2_rotator_32fc
 *
 * \b Deprecation
 *
 * This kernel is deprecated, because passing in `lv_32fc_t` by value results in
 * Undefined Behaviour, causing a segmentation fault on some architectures.
 * Use `volk_32fc_s32fc_x2_rotator2_32fc` instead.
 *
 * \b Overview
 *
 * Applies a complex phase rotation to each sample of the input vector at a
 * fixed rate per sample, starting from an initial phase offset. Each output
 * sample is computed as out[n] = in[n] * phase, where phase is multiplied by
 * the phase increment after each sample to advance the rotation angle.
 *
 * This kernel implements a numerically controlled oscillator (NCO) combined
 * with a mixer, commonly used for frequency translation in digital receivers,
 * fine frequency correction in carrier synchronization loops, and signal
 * generation when applied to a DC input.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_s32fc_x2_rotator_32fc(lv_32fc_t* outVector, const lv_32fc_t* inVector,
 * const lv_32fc_t phase_inc, lv_32fc_t* phase, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li inVector: Input complex samples to be frequency-shifted (lv_32fc_t).
 * \li phase_inc: Complex phase increment per sample, representing the
 * rotational velocity as a unit-magnitude phasor (lv_32fc_t).
 * \li phase: Pointer to the current phase accumulator, updated in-place
 * across calls to maintain phase continuity (lv_32fc_t).
 * \li num_points: The number of complex samples to process.
 *
 * \b Outputs
 * \li outVector: The frequency-shifted output samples (lv_32fc_t).
 *
 * \b Example
 * Rotate a DC signal by 90 degrees per sample. With a unit DC input and
 * phase_inc = (0, 1), the output cycles through 1, j, -1, -j.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 * lv_32fc_t* in  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 * lv_32fc_t* out = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *     in[i] = lv_cmake(1.0f, 0.0f); // DC input
 * }
 *
 * // Rotate 90 degrees per sample: phase_inc = (0, 1)
 * lv_32fc_t phase_inc = lv_cmake(0.0f, 1.0f);
 * lv_32fc_t phase = lv_cmake(1.0f, 0.0f); // start at 0 radians
 *
 * volk_32fc_s32fc_x2_rotator_32fc(out, in, phase_inc, &phase, N);
 *
 * // Expected: out[0]=(1,0), out[1]=(0,1), out[2]=(-1,0), out[3]=(0,-1)
 * printf("Expected: (1,0), (0,1), (-1,0), (0,-1)\n");
 * for (unsigned int i = 0; i < N; ++i) {
 *     printf("out[%u] = (%+1.2f, %+1.2f)\n",
 *         i, lv_creal(out[i]), lv_cimag(out[i]));
 * }
 *
 * volk_free(in);
 * volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_s32fc_rotator_32fc_a_H
#define INCLUDED_volk_32fc_s32fc_rotator_32fc_a_H


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <volk/volk_32fc_s32fc_x2_rotator_32fc.h>
#include <volk/volk_complex.h>


#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_s32fc_x2_rotator_32fc_generic(lv_32fc_t* outVector,
                                                           const lv_32fc_t* inVector,
                                                           const lv_32fc_t phase_inc,
                                                           lv_32fc_t* phase,
                                                           unsigned int num_points)
{
    volk_32fc_s32fc_x2_rotator2_32fc_generic(
        outVector, inVector, &phase_inc, phase, num_points);
}

#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_NEON

static inline void volk_32fc_s32fc_x2_rotator_32fc_neon(lv_32fc_t* outVector,
                                                        const lv_32fc_t* inVector,
                                                        const lv_32fc_t phase_inc,
                                                        lv_32fc_t* phase,
                                                        unsigned int num_points)

{
    volk_32fc_s32fc_x2_rotator2_32fc_neon(
        outVector, inVector, &phase_inc, phase, num_points);
}

#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_AVX

static inline void volk_32fc_s32fc_x2_rotator_32fc_a_avx(lv_32fc_t* outVector,
                                                         const lv_32fc_t* inVector,
                                                         const lv_32fc_t phase_inc,
                                                         lv_32fc_t* phase,
                                                         unsigned int num_points)
{
    volk_32fc_s32fc_x2_rotator2_32fc_a_avx(
        outVector, inVector, &phase_inc, phase, num_points);
}

#endif /* LV_HAVE_AVX for aligned */


#ifdef LV_HAVE_AVX

static inline void volk_32fc_s32fc_x2_rotator_32fc_u_avx(lv_32fc_t* outVector,
                                                         const lv_32fc_t* inVector,
                                                         const lv_32fc_t phase_inc,
                                                         lv_32fc_t* phase,
                                                         unsigned int num_points)
{
    volk_32fc_s32fc_x2_rotator2_32fc_u_avx(
        outVector, inVector, &phase_inc, phase, num_points);
}

#endif /* LV_HAVE_AVX */

#endif /* INCLUDED_volk_32fc_s32fc_rotator_32fc_a_H */
