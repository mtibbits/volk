/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_null_32f
 *
 * \b Overview
 *
 * Copies floating-point samples from the input vector to the output vector,
 * performing the identity operation: out[i] = in[i]. This is a null kernel
 * that applies no transformation to the signal data. It is primarily useful
 * for benchmarking VOLK dispatcher and memory-access overhead independently
 * of any arithmetic, allowing engineers to isolate the baseline cost of
 * vectorized data movement in a signal processing pipeline.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_null_32f(float* bVector, const float* aVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: The input vector of samples (float).
 * \li num_points: The number of float samples to copy.
 *
 * \b Outputs
 * \li bVector: The output vector (float).
 *
 * \b Example
 * Copy a small vector and verify the output matches the input.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 *
 * float* input = (float*)volk_malloc(sizeof(float) * N, alignment);
 * float* output = (float*)volk_malloc(sizeof(float) * N, alignment);
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *     input[i] = (float)(i + 1);
 * }
 *
 * volk_32f_null_32f(output, input, N);
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *     printf("Expected: %1.1f  Result: %1.1f\n", input[i], output[i]);
 * }
 *
 * volk_free(input);
 * volk_free(output);
 * \endcode
 */

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#ifndef INCLUDED_volk_32f_null_32f_a_H
#define INCLUDED_volk_32f_null_32f_a_H

#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_null_32f_generic(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;
    unsigned int number;

    for (number = 0; number < num_points; number++) {
        *bPtr++ = *aPtr++;
    }
}
#endif /* LV_HAVE_GENERIC */

#endif /* INCLUDED_volk_32f_null_32f_u_H */
