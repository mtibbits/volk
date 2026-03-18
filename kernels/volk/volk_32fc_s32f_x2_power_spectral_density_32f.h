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
 * Computes the power spectral density (PSD) of complex FFT data. Each output sample is
 * 10 * log10((real/norm)^2 + (imag/norm)^2) normalized by the resolution bandwidth (RBW).
 * When RBW is not 1.0, the normalization factor is scaled by sqrt(RBW) so that the result
 * represents power per unit bandwidth in dB.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_s32f_x2_power_spectral_density_32f(float* logPowerOutput, const
 * lv_32fc_t* complexFFTInput, const float normalizationFactor, const float rbw, unsigned
 * int num_points) \endcode
 *
 * \b Inputs
 * \li complexFFTInput: The complex data output from the FFT.
 * \li normalizationFactor: Each input value is divided by this factor before the power is
 * calculated.
 * \li rbw: The resolution bandwidth of the FFT spectrum.
 * \li num_points: The number of FFT data points.
 *
 * \b Outputs
 * \li logPowerOutput: The 10.0 * log10((r*r + i*i) / (norm*norm * rbw)) for each data
 * point.
 *
 * \b Example
 * Compute the power spectral density in dB/Hz from simulated FFT output.
 * \code
 * #include <volk/volk.h>
 * #include <stdio.h>
 * #include <math.h>
 *
 * int main() {
 *     unsigned int N = 8;
 *     unsigned int alignment = volk_get_alignment();
 *     float normalizationFactor = (float)N;
 *     // RBW = sample_rate / N; e.g. 1000 Hz / 8 = 125 Hz
 *     float rbw = 125.0f;
 *
 *     lv_32fc_t* fftOutput =
 *         (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 *     float* psd = (float*)volk_malloc(sizeof(float) * N, alignment);
 *
 *     // Simulate FFT output with a strong bin at index 1
 *     for (unsigned int i = 0; i < N; i++) {
 *         fftOutput[i] = lv_cmake(0.01f, 0.01f);
 *     }
 *     fftOutput[1] = lv_cmake(4.0f, 3.0f); // magnitude 5.0
 *
 *     volk_32fc_s32f_x2_power_spectral_density_32f(
 *         psd, fftOutput, normalizationFactor, rbw, N);
 *
 *     for (unsigned int i = 0; i < N; i++) {
 *         printf("bin[%u] = %+.2f dB/Hz\n", i, psd[i]);
 *     }
 *
 *     volk_free(fftOutput);
 *     volk_free(psd);
 *     return 0;
 * }
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
