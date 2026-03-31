/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_s32f_x2_power_spectral_density_32f
 *
 * \b Overview
 *
 * Computes the power spectral density (PSD) of complex FFT output in decibels.
 * Each input sample is first normalized by the given factor, then the log power
 * is computed and divided by the resolution bandwidth (RBW):
 * output[i] = 10 * log10((r*r + i*i) / (norm^2 * rbw)).
 *
 * This kernel is used in spectral analysis to convert raw FFT output into
 * calibrated power spectral density, as displayed by spectrum analyzers. The
 * normalization factor accounts for FFT length and windowing gain, while the
 * RBW division converts power to power density (dB/Hz), enabling comparison
 * across different FFT sizes and sample rates.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_s32f_x2_power_spectral_density_32f(float* logPowerOutput,
 *   const lv_32fc_t* complexFFTInput, const float normalizationFactor,
 *   const float rbw, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li complexFFTInput: The complex FFT output samples (lv_32fc_t).
 * \li normalizationFactor: Scale factor applied to each sample before computing
 *   power (accounts for FFT length and window gain).
 * \li rbw: The resolution bandwidth of the FFT spectrum.
 * \li num_points: The number of FFT data points.
 *
 * \b Outputs
 * \li logPowerOutput: The power spectral density in dB for each point (float).
 *
 * \b Example
 * Compute PSD of a constant-magnitude complex signal with unit normalization and RBW.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 *
 * lv_32fc_t* input = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 * float* output = (float*)volk_malloc(sizeof(float) * N, alignment);
 *
 * // Input: (3, 4) has magnitude 5, so |x|^2 = 25
 * for (unsigned int i = 0; i < N; ++i) {
 *     input[i] = lv_cmake(3.0f, 4.0f);
 * }
 * float norm = 1.0f;
 * float rbw = 1.0f;
 *
 * // Expected: 10 * log10(25 / (1^2 * 1)) = 10 * log10(25) ~ 13.979 dB
 * float expected = 10.0f * log10f(25.0f);
 *
 * volk_32fc_s32f_x2_power_spectral_density_32f(output, input, norm, rbw, N);
 *
 * printf("Expected: %f\n", expected);
 * printf("Result:   %f\n", output[0]);
 *
 * volk_free(input);
 * volk_free(output);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_s32f_x2_power_spectral_density_32f_u_H
#define INCLUDED_volk_32fc_s32f_x2_power_spectral_density_32f_u_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void
volk_32fc_s32f_x2_power_spectral_density_32f_generic(float* logPowerOutput,
                                                     const lv_32fc_t* complexFFTInput,
                                                     const float normalizationFactor,
                                                     const float rbw,
                                                     unsigned int num_points)
{
    if (rbw != 1.0f)
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor * sqrtf(rbw), num_points);
    else
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor, num_points);
}

#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE2

static inline void
volk_32fc_s32f_x2_power_spectral_density_32f_u_sse2(float* logPowerOutput,
                                                      const lv_32fc_t* complexFFTInput,
                                                      const float normalizationFactor,
                                                      const float rbw,
                                                      unsigned int num_points)
{
    if (rbw != 1.0f)
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor * sqrtf(rbw), num_points);
    else
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor, num_points);
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_AVX

static inline void
volk_32fc_s32f_x2_power_spectral_density_32f_u_avx(float* logPowerOutput,
                                                     const lv_32fc_t* complexFFTInput,
                                                     const float normalizationFactor,
                                                     const float rbw,
                                                     unsigned int num_points)
{
    if (rbw != 1.0f)
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor * sqrtf(rbw), num_points);
    else
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor, num_points);
}
#endif /* LV_HAVE_AVX */

#if LV_HAVE_AVX && LV_HAVE_FMA

static inline void
volk_32fc_s32f_x2_power_spectral_density_32f_u_avx_fma(float* logPowerOutput,
                                                         const lv_32fc_t* complexFFTInput,
                                                         const float normalizationFactor,
                                                         const float rbw,
                                                         unsigned int num_points)
{
    if (rbw != 1.0f)
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor * sqrtf(rbw), num_points);
    else
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor, num_points);
}
#endif /* LV_HAVE_AVX && LV_HAVE_FMA */

#if LV_HAVE_AVX2 && LV_HAVE_FMA

static inline void
volk_32fc_s32f_x2_power_spectral_density_32f_u_avx2_fma(float* logPowerOutput,
                                                          const lv_32fc_t* complexFFTInput,
                                                          const float normalizationFactor,
                                                          const float rbw,
                                                          unsigned int num_points)
{
    if (rbw != 1.0f)
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor * sqrtf(rbw), num_points);
    else
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor, num_points);
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX512F

static inline void
volk_32fc_s32f_x2_power_spectral_density_32f_u_avx512f(float* logPowerOutput,
                                                         const lv_32fc_t* complexFFTInput,
                                                         const float normalizationFactor,
                                                         const float rbw,
                                                         unsigned int num_points)
{
    if (rbw != 1.0f)
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor * sqrtf(rbw), num_points);
    else
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor, num_points);
}
#endif /* LV_HAVE_AVX512F */

#if LV_HAVE_AVX512F && LV_HAVE_AVX512DQ

static inline void
volk_32fc_s32f_x2_power_spectral_density_32f_u_avx512dq(float* logPowerOutput,
                                                          const lv_32fc_t* complexFFTInput,
                                                          const float normalizationFactor,
                                                          const float rbw,
                                                          unsigned int num_points)
{
    if (rbw != 1.0f)
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor * sqrtf(rbw), num_points);
    else
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor, num_points);
}
#endif /* LV_HAVE_AVX512F && LV_HAVE_AVX512DQ */


#ifdef LV_HAVE_NEON

static inline void
volk_32fc_s32f_x2_power_spectral_density_32f_neon(float* logPowerOutput,
                                                   const lv_32fc_t* complexFFTInput,
                                                   const float normalizationFactor,
                                                   const float rbw,
                                                   unsigned int num_points)
{
    if (rbw != 1.0f)
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor * sqrtf(rbw), num_points);
    else
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor, num_points);
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8

static inline void
volk_32fc_s32f_x2_power_spectral_density_32f_neonv8(float* logPowerOutput,
                                                      const lv_32fc_t* complexFFTInput,
                                                      const float normalizationFactor,
                                                      const float rbw,
                                                      unsigned int num_points)
{
    if (rbw != 1.0f)
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor * sqrtf(rbw), num_points);
    else
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor, num_points);
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV

static inline void
volk_32fc_s32f_x2_power_spectral_density_32f_rvv(float* logPowerOutput,
                                                   const lv_32fc_t* complexFFTInput,
                                                   const float normalizationFactor,
                                                   const float rbw,
                                                   unsigned int num_points)
{
    if (rbw != 1.0f)
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor * sqrtf(rbw), num_points);
    else
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor, num_points);
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_RVVSEG

static inline void
volk_32fc_s32f_x2_power_spectral_density_32f_rvvseg(float* logPowerOutput,
                                                      const lv_32fc_t* complexFFTInput,
                                                      const float normalizationFactor,
                                                      const float rbw,
                                                      unsigned int num_points)
{
    if (rbw != 1.0f)
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor * sqrtf(rbw), num_points);
    else
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor, num_points);
}
#endif /* LV_HAVE_RVVSEG */

#endif /* INCLUDED_volk_32fc_s32f_x2_power_spectral_density_32f_u_H */

#ifndef INCLUDED_volk_32fc_s32f_x2_power_spectral_density_32f_a_H
#define INCLUDED_volk_32fc_s32f_x2_power_spectral_density_32f_a_H

#ifdef LV_HAVE_SSE2

static inline void
volk_32fc_s32f_x2_power_spectral_density_32f_a_sse2(float* logPowerOutput,
                                                      const lv_32fc_t* complexFFTInput,
                                                      const float normalizationFactor,
                                                      const float rbw,
                                                      unsigned int num_points)
{
    if (rbw != 1.0f)
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor * sqrtf(rbw), num_points);
    else
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor, num_points);
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_AVX

static inline void
volk_32fc_s32f_x2_power_spectral_density_32f_a_avx(float* logPowerOutput,
                                                     const lv_32fc_t* complexFFTInput,
                                                     const float normalizationFactor,
                                                     const float rbw,
                                                     unsigned int num_points)
{
    if (rbw != 1.0f)
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor * sqrtf(rbw), num_points);
    else
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor, num_points);
}
#endif /* LV_HAVE_AVX */

#if LV_HAVE_AVX && LV_HAVE_FMA

static inline void
volk_32fc_s32f_x2_power_spectral_density_32f_a_avx_fma(float* logPowerOutput,
                                                         const lv_32fc_t* complexFFTInput,
                                                         const float normalizationFactor,
                                                         const float rbw,
                                                         unsigned int num_points)
{
    if (rbw != 1.0f)
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor * sqrtf(rbw), num_points);
    else
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor, num_points);
}
#endif /* LV_HAVE_AVX && LV_HAVE_FMA */

#if LV_HAVE_AVX2 && LV_HAVE_FMA

static inline void
volk_32fc_s32f_x2_power_spectral_density_32f_a_avx2_fma(float* logPowerOutput,
                                                          const lv_32fc_t* complexFFTInput,
                                                          const float normalizationFactor,
                                                          const float rbw,
                                                          unsigned int num_points)
{
    if (rbw != 1.0f)
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor * sqrtf(rbw), num_points);
    else
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor, num_points);
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX512F

static inline void
volk_32fc_s32f_x2_power_spectral_density_32f_a_avx512f(float* logPowerOutput,
                                                         const lv_32fc_t* complexFFTInput,
                                                         const float normalizationFactor,
                                                         const float rbw,
                                                         unsigned int num_points)
{
    if (rbw != 1.0f)
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor * sqrtf(rbw), num_points);
    else
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor, num_points);
}
#endif /* LV_HAVE_AVX512F */

#if LV_HAVE_AVX512F && LV_HAVE_AVX512DQ

static inline void
volk_32fc_s32f_x2_power_spectral_density_32f_a_avx512dq(float* logPowerOutput,
                                                          const lv_32fc_t* complexFFTInput,
                                                          const float normalizationFactor,
                                                          const float rbw,
                                                          unsigned int num_points)
{
    if (rbw != 1.0f)
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor * sqrtf(rbw), num_points);
    else
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor, num_points);
}
#endif /* LV_HAVE_AVX512F && LV_HAVE_AVX512DQ */

#endif /* INCLUDED_volk_32fc_s32f_x2_power_spectral_density_32f_a_H */
