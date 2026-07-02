/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_s32f_power_32fc
 *
 * \b Overview
 *
 * Raises each element of a complex vector to a real-valued power:
 * cVector[i] = aVector[i] ^ power. The exponentiation is performed in polar
 * form by scaling the magnitude and multiplying the phase angle.
 *
 * Complex exponentiation is used in signal processing for operations such as
 * carrier recovery, M-th power nonlinearity-based frequency estimation, and
 * higher-order moment computation. For example, squaring a BPSK signal
 * removes data modulation and reveals the carrier frequency.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_s32f_power_32fc(lv_32fc_t* cVector, const lv_32fc_t* aVector, const
 * float power, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li aVector: The complex input vector of samples (lv_32fc_t).
 * \li power: The real-valued exponent applied to each complex sample.
 * \li num_points: The number of complex samples.
 *
 * \b Outputs
 * \li cVector: The complex output vector (lv_32fc_t).
 *
 * \b Example
 * Raise a vector of complex values to the power of 2 (squaring).
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 *
 * lv_32fc_t* input = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 * lv_32fc_t* output = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 *
 * float power = 2.0f;
 * for (unsigned int i = 0; i < N; ++i) {
 *     input[i] = lv_cmake(3.0f, 4.0f); // magnitude = 5
 * }
 *
 * // Expected: (3+4j)^2 = 9 + 24j - 16 = -7 + 24j
 * lv_32fc_t expected = lv_cmake(-7.0f, 24.0f);
 *
 * volk_32fc_s32f_power_32fc(output, input, power, N);
 *
 * printf("Expected: %+.1f%+.1fj\n", lv_creal(expected), lv_cimag(expected));
 * printf("Result:   %+.1f%+.1fj\n", lv_creal(output[0]), lv_cimag(output[0]));
 *
 * volk_free(input);
 * volk_free(output);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_s32f_power_32fc_a_H
#define INCLUDED_volk_32fc_s32f_power_32fc_a_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

//! raise a complex float to a real float power
static inline lv_32fc_t __volk_s32fc_s32f_power_s32fc_a(const lv_32fc_t exp,
                                                        const float power)
{
    const float arg = power * atan2f(lv_cimag(exp), lv_creal(exp));
    const float mag =
        powf(lv_creal(exp) * lv_creal(exp) + lv_cimag(exp) * lv_cimag(exp), power / 2);
    return mag * lv_cmake(cosf(arg), sinf(arg));
}

#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_s32f_power_32fc_generic(lv_32fc_t* cVector,
                                                     const lv_32fc_t* aVector,
                                                     const float power,
                                                     unsigned int num_points)
{
    lv_32fc_t* cPtr = cVector;
    const lv_32fc_t* aPtr = aVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *cPtr++ = __volk_s32fc_s32f_power_s32fc_a((*aPtr++), power);
    }
}

#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32fc_s32f_power_32fc_a_H */
