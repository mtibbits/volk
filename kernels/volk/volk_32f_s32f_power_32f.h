/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_s32f_power_32f
 *
 * \b Overview
 *
 * Raises each element of the input vector to the specified scalar power
 * and stores the results in the output vector: c[i] = powf(a[i], power).
 *
 * Element-wise exponentiation is used in signal processing for power-law
 * compression and expansion, non-linear amplitude transformations, and
 * computing signal statistics such as higher-order moments. For example,
 * squaring a signal (power = 2) is a common step in energy detection and
 * spectral analysis pipelines.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_s32f_power_32f(float* cVector, const float* aVector, const float power,
 * unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: The input vector of signal samples (float).
 * \li power: The exponent to raise each input value to (float).
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li cVector: The output vector of transformed samples (float).
 *
 * \b Example
 * Cube a constant vector and verify the result.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 * float* in = (float*)volk_malloc(sizeof(float) * N, alignment);
 * float* out = (float*)volk_malloc(sizeof(float) * N, alignment);
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *     in[i] = 2.0f;
 * }
 * float power = 3.0f;
 *
 * // Expected: powf(2.0, 3.0) = 8.0 for each element
 * float expected = 8.0f;
 *
 * volk_32f_s32f_power_32f(out, in, power, N);
 *
 * printf("Expected: %f\n", expected);
 * printf("Result:   %f\n", out[0]);
 *
 * volk_free(in);
 * volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_s32f_power_32f_a_H
#define INCLUDED_volk_32f_s32f_power_32f_a_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_s32f_power_32f_generic(float* cVector,
                                                   const float* aVector,
                                                   const float power,
                                                   unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *cPtr++ = powf((*aPtr++), power);
    }
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32f_s32f_power_32f_a_H */
