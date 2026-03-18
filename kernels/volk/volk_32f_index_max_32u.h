/* -*- c++ -*- */
/*
 * Copyright 2016 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_index_max_32u
 *
 * \b Overview
 *
 * Returns Argmax_i x[i]. Finds and returns the index which contains the first maximum
 * value in the given vector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_index_max_32u(uint32_t* target, const float* src0, uint32_t num_points)
 * \endcode
 *
 * \b Inputs
 * \li src0: The input vector of floats.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li target: The index of the first maximum value in the input buffer.
 *
 * \b Example
 * \code
 *   int N = 10;
 *   uint32_t alignment = volk_get_alignment();
 *   float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   uint32_t* out = (uint32_t*)volk_malloc(sizeof(uint32_t), alignment);
 *
 *   for(uint32_t ii = 0; ii < N; ++ii){
 *       float x = (float)ii;
 *       // a parabola with a maximum at x=4
 *       in[ii] = -(x-4) * (x-4) + 5;
 *   }
 *
 *   volk_32f_index_max_32u(out, in, N);
 *
 *   printf("maximum is %1.2f at index %u\n", in[*out], *out);
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_index_max_32u_u_H
#define INCLUDED_volk_32f_index_max_32u_u_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_index_max_32u_generic(uint32_t* target, const float* src0, uint32_t num_points)
{
    if (num_points > 0) {
        float max = src0[0];
        uint32_t index = 0;

        uint32_t i = 1;

        for (; i < num_points; ++i) {
            if (src0[i] > max) {
                index = i;
                max = src0[i];
            }
        }
        target[0] = index;
    }
}

#endif /*LV_HAVE_GENERIC*/


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void
volk_32f_index_max_32u_u_sse2(uint32_t* target, const float* src0, uint32_t num_points)
{
    if (num_points > 0) {
        uint32_t number = 0;
        const uint32_t quarterPoints = num_points / 4;

        const float* inputPtr = src0;

        __m128i indexIncrementValues = _mm_set1_epi32(4);
        __m128i currentIndexes = _mm_set_epi32(-1, -2, -3, -4);

        float max = src0[0];
        uint32_t index = 0;
        __m128 maxValues = _mm_set1_ps(max);
        __m128i maxValuesIndex = _mm_setzero_si128();
        __m128 compareResults;
        __m128 currentValues;

        __VOLK_ATTR_ALIGNED(16) float maxValuesBuffer[4];
        __VOLK_ATTR_ALIGNED(16) uint32_t maxIndexesBuffer[4];

        for (; number < quarterPoints; number++) {
            currentValues = _mm_loadu_ps(inputPtr);
            inputPtr += 4;
            currentIndexes = _mm_add_epi32(currentIndexes, indexIncrementValues);
            compareResults = _mm_cmpgt_ps(currentValues, maxValues);
            __m128i cmpMask = _mm_castps_si128(compareResults);
            maxValuesIndex = _mm_or_si128(_mm_and_si128(cmpMask, currentIndexes),
                                          _mm_andnot_si128(cmpMask, maxValuesIndex));
            maxValues = _mm_or_ps(_mm_and_ps(compareResults, currentValues),
                                  _mm_andnot_ps(compareResults, maxValues));
        }

        // Calculate the largest value from the remaining 4 points
        _mm_store_ps(maxValuesBuffer, maxValues);
        _mm_store_si128((__m128i*)maxIndexesBuffer, maxValuesIndex);

        for (number = 0; number < 4; number++) {
            if (maxValuesBuffer[number] > max) {
                index = maxIndexesBuffer[number];
                max = maxValuesBuffer[number];
            } else if (maxValuesBuffer[number] == max) {
                if (index > maxIndexesBuffer[number])
                    index = maxIndexesBuffer[number];
            }
        }

        number = quarterPoints * 4;
        for (; number < num_points; number++) {
            if (src0[number] > max) {
                index = number;
                max = src0[number];
            }
        }
        target[0] = index;
    }
}

#endif /*LV_HAVE_SSE2*/


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_32f_index_max_32u_u_sse4_1(uint32_t* target, const float* src0, uint32_t num_points)
{
    if (num_points > 0) {
        uint32_t number = 0;
        const uint32_t quarterPoints = num_points / 4;

        const float* inputPtr = src0;

        __m128i indexIncrementValues = _mm_set1_epi32(4);
        __m128i currentIndexes = _mm_set_epi32(-1, -2, -3, -4);

        float max = src0[0];
        uint32_t index = 0;
        __m128 maxValues = _mm_set1_ps(max);
        __m128i maxValuesIndex = _mm_setzero_si128();
        __m128 compareResults;
        __m128 currentValues;

        __VOLK_ATTR_ALIGNED(16) float maxValuesBuffer[4];
        __VOLK_ATTR_ALIGNED(16) uint32_t maxIndexesBuffer[4];

        for (; number < quarterPoints; number++) {
            currentValues = _mm_loadu_ps(inputPtr);
            inputPtr += 4;
            currentIndexes = _mm_add_epi32(currentIndexes, indexIncrementValues);
            compareResults = _mm_cmpgt_ps(currentValues, maxValues);
            __m128i cmpMask = _mm_castps_si128(compareResults);
            maxValuesIndex =
                _mm_blendv_epi8(maxValuesIndex, currentIndexes, cmpMask);
            maxValues = _mm_blendv_ps(maxValues, currentValues, compareResults);
        }

        // Calculate the largest value from the remaining 4 points
        _mm_store_ps(maxValuesBuffer, maxValues);
        _mm_store_si128((__m128i*)maxIndexesBuffer, maxValuesIndex);

        for (number = 0; number < 4; number++) {
            if (maxValuesBuffer[number] > max) {
                index = maxIndexesBuffer[number];
                max = maxValuesBuffer[number];
            } else if (maxValuesBuffer[number] == max) {
                if (index > maxIndexesBuffer[number])
                    index = maxIndexesBuffer[number];
            }
        }

        number = quarterPoints * 4;
        for (; number < num_points; number++) {
            if (src0[number] > max) {
                index = number;
                max = src0[number];
            }
        }
        target[0] = index;
    }
}

#endif /*LV_HAVE_SSE4_1*/


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_index_max_32u_u_avx(uint32_t* target, const float* src0, uint32_t num_points)
{
    if (num_points > 0) {
        uint32_t number = 0;
        const uint32_t eighthPoints = num_points / 8;

        const float* inputPtr = src0;

        /* Track indices as two 128-bit halves (lo = lanes 0-3, hi = lanes 4-7) */
        __m128i indexIncrementValues = _mm_set1_epi32(8);
        __m128i curIdxLo = _mm_set_epi32(-5, -6, -7, -8);
        __m128i curIdxHi = _mm_set_epi32(-1, -2, -3, -4);

        float max = src0[0];
        uint32_t index = 0;
        __m256 maxValues = _mm256_set1_ps(max);
        __m128i maxIdxLo = _mm_setzero_si128();
        __m128i maxIdxHi = _mm_setzero_si128();

        __VOLK_ATTR_ALIGNED(32) float maxValuesBuffer[8];
        __VOLK_ATTR_ALIGNED(16) uint32_t maxIdxLoBuffer[4];
        __VOLK_ATTR_ALIGNED(16) uint32_t maxIdxHiBuffer[4];

        for (; number < eighthPoints; number++) {
            __m256 currentValues = _mm256_loadu_ps(inputPtr);
            inputPtr += 8;

            /* Increment indices */
            curIdxLo = _mm_add_epi32(curIdxLo, indexIncrementValues);
            curIdxHi = _mm_add_epi32(curIdxHi, indexIncrementValues);

            /* Compare using AVX 256-bit float compare */
            __m256 compareResults = _mm256_cmp_ps(currentValues, maxValues, _CMP_GT_OS);

            /* Split compare mask to 128-bit halves for integer blend (SSE4.1) */
            __m128i cmpLo = _mm_castps_si128(_mm256_castps256_ps128(compareResults));
            __m128i cmpHi = _mm_castps_si128(_mm256_extractf128_ps(compareResults, 1));

            /* Blend indices using SSE4.1 (available on all AVX CPUs) */
            maxIdxLo = _mm_blendv_epi8(maxIdxLo, curIdxLo, cmpLo);
            maxIdxHi = _mm_blendv_epi8(maxIdxHi, curIdxHi, cmpHi);

            /* Blend max values */
            maxValues = _mm256_blendv_ps(maxValues, currentValues, compareResults);
        }

        /* Extract results */
        _mm256_store_ps(maxValuesBuffer, maxValues);
        _mm_store_si128((__m128i*)maxIdxLoBuffer, maxIdxLo);
        _mm_store_si128((__m128i*)maxIdxHiBuffer, maxIdxHi);

        for (number = 0; number < 4; number++) {
            if (maxValuesBuffer[number] > max) {
                index = maxIdxLoBuffer[number];
                max = maxValuesBuffer[number];
            } else if (maxValuesBuffer[number] == max) {
                if (index > maxIdxLoBuffer[number])
                    index = maxIdxLoBuffer[number];
            }
        }
        for (number = 0; number < 4; number++) {
            if (maxValuesBuffer[number + 4] > max) {
                index = maxIdxHiBuffer[number];
                max = maxValuesBuffer[number + 4];
            } else if (maxValuesBuffer[number + 4] == max) {
                if (index > maxIdxHiBuffer[number])
                    index = maxIdxHiBuffer[number];
            }
        }

        number = eighthPoints * 8;
        for (; number < num_points; number++) {
            if (src0[number] > max) {
                index = number;
                max = src0[number];
            }
        }
        target[0] = index;
    }
}

#endif /*LV_HAVE_AVX*/


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_32f_index_max_32u_u_avx2(uint32_t* target, const float* src0, uint32_t num_points)
{
    if (num_points > 0) {
        uint32_t number = 0;
        const uint32_t quarterPoints = num_points / 8;

        const float* inputPtr = src0;

        __m256i indexIncrementValues = _mm256_set1_epi32(8);
        __m256i currentIndexes = _mm256_set_epi32(-1, -2, -3, -4, -5, -6, -7, -8);

        float max = src0[0];
        uint32_t index = 0;
        __m256 maxValues = _mm256_set1_ps(max);
        __m256i maxValuesIndex = _mm256_setzero_si256();
        __m256 compareResults;
        __m256 currentValues;

        __VOLK_ATTR_ALIGNED(32) float maxValuesBuffer[8];
        __VOLK_ATTR_ALIGNED(32) uint32_t maxIndexesBuffer[8];

        for (; number < quarterPoints; number++) {
            currentValues = _mm256_loadu_ps(inputPtr);
            inputPtr += 8;
            currentIndexes = _mm256_add_epi32(currentIndexes, indexIncrementValues);
            compareResults = _mm256_cmp_ps(currentValues, maxValues, _CMP_GT_OS);
            __m256i cmpMask = _mm256_castps_si256(compareResults);
            maxValuesIndex =
                _mm256_blendv_epi8(maxValuesIndex, currentIndexes, cmpMask);
            maxValues = _mm256_blendv_ps(maxValues, currentValues, compareResults);
        }

        // Calculate the largest value from the remaining 8 points
        _mm256_store_ps(maxValuesBuffer, maxValues);
        _mm256_store_si256((__m256i*)maxIndexesBuffer, maxValuesIndex);

        for (number = 0; number < 8; number++) {
            if (maxValuesBuffer[number] > max) {
                index = maxIndexesBuffer[number];
                max = maxValuesBuffer[number];
            } else if (maxValuesBuffer[number] == max) {
                if (index > maxIndexesBuffer[number])
                    index = maxIndexesBuffer[number];
            }
        }

        number = quarterPoints * 8;
        for (; number < num_points; number++) {
            if (src0[number] > max) {
                index = number;
                max = src0[number];
            }
        }
        target[0] = index;
    }
}

#endif /*LV_HAVE_AVX2*/


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void
volk_32f_index_max_32u_u_avx512f(uint32_t* target, const float* src0, uint32_t num_points)
{
    if (num_points > 0) {
        uint32_t number = 0;
        const uint32_t sixteenthPoints = num_points / 16;

        const float* inputPtr = src0;

        __m512i indexIncrementValues = _mm512_set1_epi32(16);
        __m512i currentIndexes = _mm512_set_epi32(
            -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16);

        float max = src0[0];
        uint32_t index = 0;
        __m512 maxValues = _mm512_set1_ps(max);
        __m512i maxValuesIndex = _mm512_setzero_si512();
        __mmask16 compareResults;
        __m512 currentValues;

        __VOLK_ATTR_ALIGNED(64) float maxValuesBuffer[16];
        __VOLK_ATTR_ALIGNED(64) uint32_t maxIndexesBuffer[16];

        for (; number < sixteenthPoints; number++) {
            currentValues = _mm512_loadu_ps(inputPtr);
            inputPtr += 16;
            currentIndexes = _mm512_add_epi32(currentIndexes, indexIncrementValues);
            compareResults = _mm512_cmp_ps_mask(currentValues, maxValues, _CMP_GT_OS);
            maxValuesIndex =
                _mm512_mask_blend_epi32(compareResults, maxValuesIndex, currentIndexes);
            maxValues = _mm512_mask_blend_ps(compareResults, maxValues, currentValues);
        }

        // Calculate the largest value from the remaining 16 points
        _mm512_store_ps(maxValuesBuffer, maxValues);
        _mm512_store_si512(maxIndexesBuffer, maxValuesIndex);

        for (number = 0; number < 16; number++) {
            if (maxValuesBuffer[number] > max) {
                index = maxIndexesBuffer[number];
                max = maxValuesBuffer[number];
            } else if (maxValuesBuffer[number] == max) {
                if (index > maxIndexesBuffer[number])
                    index = maxIndexesBuffer[number];
            }
        }

        number = sixteenthPoints * 16;
        for (; number < num_points; number++) {
            if (src0[number] > max) {
                index = number;
                max = src0[number];
            }
        }
        target[0] = index;
    }
}

#endif /*LV_HAVE_AVX512F*/


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32f_index_max_32u_neon(uint32_t* target, const float* src0, uint32_t num_points)
{
    if (num_points > 0) {
        uint32_t number = 0;
        const uint32_t quarterPoints = num_points / 4;

        const float* inputPtr = src0;
        float32x4_t indexIncrementValues = vdupq_n_f32(4);
        __VOLK_ATTR_ALIGNED(16)
        float currentIndexes_float[4] = { -4.0f, -3.0f, -2.0f, -1.0f };
        float32x4_t currentIndexes = vld1q_f32(currentIndexes_float);

        float max = src0[0];
        uint32_t index = 0;
        float32x4_t maxValues = vdupq_n_f32(max);
        uint32x4_t maxValuesIndex = vmovq_n_u32(0);
        uint32x4_t compareResults;
        uint32x4_t currentIndexes_u;
        float32x4_t currentValues;

        __VOLK_ATTR_ALIGNED(16) float maxValuesBuffer[4];
        __VOLK_ATTR_ALIGNED(16) uint32_t maxIndexesBuffer[4];

        for (; number < quarterPoints; number++) {
            currentValues = vld1q_f32(inputPtr);
            inputPtr += 4;
            currentIndexes = vaddq_f32(currentIndexes, indexIncrementValues);
            currentIndexes_u = vcvtq_u32_f32(currentIndexes);
            compareResults = vcleq_f32(currentValues, maxValues);
            maxValuesIndex = vorrq_u32(vandq_u32(compareResults, maxValuesIndex),
                                       vbicq_u32(currentIndexes_u, compareResults));
            maxValues = vmaxq_f32(currentValues, maxValues);
        }

        // Calculate the largest value from the remaining 4 points
        vst1q_f32(maxValuesBuffer, maxValues);
        vst1q_u32(maxIndexesBuffer, maxValuesIndex);
        for (number = 0; number < 4; number++) {
            if (maxValuesBuffer[number] > max) {
                index = maxIndexesBuffer[number];
                max = maxValuesBuffer[number];
            } else if (maxValuesBuffer[number] == max) {
                if (index > maxIndexesBuffer[number])
                    index = maxIndexesBuffer[number];
            }
        }

        number = quarterPoints * 4;
        for (; number < num_points; number++) {
            if (src0[number] > max) {
                index = number;
                max = src0[number];
            }
        }
        target[0] = index;
    }
}

#endif /*LV_HAVE_NEON*/


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>
#include <float.h>

static inline void
volk_32f_index_max_32u_neonv8(uint32_t* target, const float* src0, uint32_t num_points)
{
    if (num_points == 0)
        return;

    const uint32_t quarter_points = num_points / 4;
    const float* inputPtr = src0;

    // Use integer indices directly (no float conversion overhead)
    uint32x4_t vec_indices = { 0, 1, 2, 3 };
    const uint32x4_t vec_incr = vdupq_n_u32(4);

    float32x4_t vec_max = vdupq_n_f32(-FLT_MAX);
    uint32x4_t vec_max_idx = vdupq_n_u32(0);

    for (uint32_t i = 0; i < quarter_points; i++) {
        float32x4_t vec_val = vld1q_f32(inputPtr);
        inputPtr += 4;

        // Compare BEFORE max update to know which lanes change
        uint32x4_t gt_mask = vcgtq_f32(vec_val, vec_max);
        vec_max_idx = vbslq_u32(gt_mask, vec_indices, vec_max_idx);

        // vmaxq_f32 is single-cycle, no dependency on comparison result
        vec_max = vmaxq_f32(vec_val, vec_max);

        vec_indices = vaddq_u32(vec_indices, vec_incr);
    }

    // ARMv8 horizontal reduction - find max value across all lanes
    float max_val = vmaxvq_f32(vec_max);

    // Find which lane(s) have the max value, get minimum index among them
    uint32x4_t max_mask = vceqq_f32(vec_max, vdupq_n_f32(max_val));
    uint32x4_t idx_masked = vbslq_u32(max_mask, vec_max_idx, vdupq_n_u32(UINT32_MAX));
    uint32_t result_idx = vminvq_u32(idx_masked);

    // Handle tail elements
    for (uint32_t i = quarter_points * 4; i < num_points; i++) {
        if (src0[i] > max_val) {
            max_val = src0[i];
            result_idx = i;
        }
    }

    *target = result_idx;
}

#endif /*LV_HAVE_NEONV8*/


#ifdef LV_HAVE_RVV
#include <float.h>
#include <riscv_vector.h>

static inline void
volk_32f_index_max_32u_rvv(uint32_t* target, const float* src0, uint32_t num_points)
{
    vfloat32m4_t vmax = __riscv_vfmv_v_f_f32m4(-FLT_MAX, __riscv_vsetvlmax_e32m4());
    vuint32m4_t vmaxi = __riscv_vmv_v_x_u32m4(0, __riscv_vsetvlmax_e32m4());
    vuint32m4_t vidx = __riscv_vid_v_u32m4(__riscv_vsetvlmax_e32m4());
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, src0 += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t v = __riscv_vle32_v_f32m4(src0, vl);
        vbool8_t m = __riscv_vmfgt(v, vmax, vl);
        vmax = __riscv_vfmax_tu(vmax, vmax, v, vl);
        vmaxi = __riscv_vmerge_tu(vmaxi, vmaxi, vidx, m, vl);
        vidx = __riscv_vadd(vidx, vl, __riscv_vsetvlmax_e32m4());
    }
    size_t vl = __riscv_vsetvlmax_e32m4();
    float max = __riscv_vfmv_f(__riscv_vfredmax(RISCV_SHRINK4(vfmax, f, 32, vmax),
                                                __riscv_vfmv_v_f_f32m1(-FLT_MAX, 1),
                                                __riscv_vsetvlmax_e32m1()));
    // Find lanes with max value, set others to UINT32_MAX
    vbool8_t m = __riscv_vmfeq(vmax, max, vl);
    vuint32m4_t idx_masked =
        __riscv_vmerge(__riscv_vmv_v_x_u32m4(UINT32_MAX, vl), vmaxi, m, vl);
    // Find minimum index among lanes with max value
    *target = __riscv_vmv_x(__riscv_vredminu(RISCV_SHRINK4(vminu, u, 32, idx_masked),
                                             __riscv_vmv_v_x_u32m1(UINT32_MAX, 1),
                                             __riscv_vsetvlmax_e32m1()));
}
#endif /*LV_HAVE_RVV*/

#endif /*INCLUDED_volk_32f_index_max_32u_u_H*/


#ifndef INCLUDED_volk_32f_index_max_32u_a_H
#define INCLUDED_volk_32f_index_max_32u_a_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void
volk_32f_index_max_32u_a_sse2(uint32_t* target, const float* src0, uint32_t num_points)
{
    if (num_points > 0) {
        uint32_t number = 0;
        const uint32_t quarterPoints = num_points / 4;

        const float* inputPtr = src0;

        __m128i indexIncrementValues = _mm_set1_epi32(4);
        __m128i currentIndexes = _mm_set_epi32(-1, -2, -3, -4);

        float max = src0[0];
        uint32_t index = 0;
        __m128 maxValues = _mm_set1_ps(max);
        __m128i maxValuesIndex = _mm_setzero_si128();
        __m128 compareResults;
        __m128 currentValues;

        __VOLK_ATTR_ALIGNED(16) float maxValuesBuffer[4];
        __VOLK_ATTR_ALIGNED(16) uint32_t maxIndexesBuffer[4];

        for (; number < quarterPoints; number++) {

            currentValues = _mm_load_ps(inputPtr);
            inputPtr += 4;
            currentIndexes = _mm_add_epi32(currentIndexes, indexIncrementValues);

            compareResults = _mm_cmpgt_ps(currentValues, maxValues);

            __m128i cmpMask = _mm_castps_si128(compareResults);
            maxValuesIndex = _mm_or_si128(_mm_and_si128(cmpMask, currentIndexes),
                                          _mm_andnot_si128(cmpMask, maxValuesIndex));

            maxValues = _mm_or_ps(_mm_and_ps(compareResults, currentValues),
                                  _mm_andnot_ps(compareResults, maxValues));
        }

        // Calculate the largest value from the remaining 4 points
        _mm_store_ps(maxValuesBuffer, maxValues);
        _mm_store_si128((__m128i*)maxIndexesBuffer, maxValuesIndex);

        for (number = 0; number < 4; number++) {
            if (maxValuesBuffer[number] > max) {
                index = maxIndexesBuffer[number];
                max = maxValuesBuffer[number];
            } else if (maxValuesBuffer[number] == max) {
                if (index > maxIndexesBuffer[number])
                    index = maxIndexesBuffer[number];
            }
        }

        number = quarterPoints * 4;
        for (; number < num_points; number++) {
            if (src0[number] > max) {
                index = number;
                max = src0[number];
            }
        }
        target[0] = index;
    }
}

#endif /*LV_HAVE_SSE2*/


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_32f_index_max_32u_a_sse4_1(uint32_t* target, const float* src0, uint32_t num_points)
{
    if (num_points > 0) {
        uint32_t number = 0;
        const uint32_t quarterPoints = num_points / 4;

        const float* inputPtr = src0;

        __m128i indexIncrementValues = _mm_set1_epi32(4);
        __m128i currentIndexes = _mm_set_epi32(-1, -2, -3, -4);

        float max = src0[0];
        uint32_t index = 0;
        __m128 maxValues = _mm_set1_ps(max);
        __m128i maxValuesIndex = _mm_setzero_si128();
        __m128 compareResults;
        __m128 currentValues;

        __VOLK_ATTR_ALIGNED(16) float maxValuesBuffer[4];
        __VOLK_ATTR_ALIGNED(16) uint32_t maxIndexesBuffer[4];

        for (; number < quarterPoints; number++) {

            currentValues = _mm_load_ps(inputPtr);
            inputPtr += 4;
            currentIndexes = _mm_add_epi32(currentIndexes, indexIncrementValues);

            compareResults = _mm_cmpgt_ps(currentValues, maxValues);

            __m128i cmpMask = _mm_castps_si128(compareResults);
            maxValuesIndex =
                _mm_blendv_epi8(maxValuesIndex, currentIndexes, cmpMask);
            maxValues = _mm_blendv_ps(maxValues, currentValues, compareResults);
        }

        // Calculate the largest value from the remaining 4 points
        _mm_store_ps(maxValuesBuffer, maxValues);
        _mm_store_si128((__m128i*)maxIndexesBuffer, maxValuesIndex);

        for (number = 0; number < 4; number++) {
            if (maxValuesBuffer[number] > max) {
                index = maxIndexesBuffer[number];
                max = maxValuesBuffer[number];
            } else if (maxValuesBuffer[number] == max) {
                if (index > maxIndexesBuffer[number])
                    index = maxIndexesBuffer[number];
            }
        }

        number = quarterPoints * 4;
        for (; number < num_points; number++) {
            if (src0[number] > max) {
                index = number;
                max = src0[number];
            }
        }
        target[0] = index;
    }
}

#endif /*LV_HAVE_SSE4_1*/


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_index_max_32u_a_avx(uint32_t* target, const float* src0, uint32_t num_points)
{
    if (num_points > 0) {
        uint32_t number = 0;
        const uint32_t eighthPoints = num_points / 8;

        const float* inputPtr = src0;

        /* Track indices as two 128-bit halves (lo = lanes 0-3, hi = lanes 4-7) */
        __m128i indexIncrementValues = _mm_set1_epi32(8);
        __m128i curIdxLo = _mm_set_epi32(-5, -6, -7, -8);
        __m128i curIdxHi = _mm_set_epi32(-1, -2, -3, -4);

        float max = src0[0];
        uint32_t index = 0;
        __m256 maxValues = _mm256_set1_ps(max);
        __m128i maxIdxLo = _mm_setzero_si128();
        __m128i maxIdxHi = _mm_setzero_si128();

        __VOLK_ATTR_ALIGNED(32) float maxValuesBuffer[8];
        __VOLK_ATTR_ALIGNED(16) uint32_t maxIdxLoBuffer[4];
        __VOLK_ATTR_ALIGNED(16) uint32_t maxIdxHiBuffer[4];

        for (; number < eighthPoints; number++) {
            __m256 currentValues = _mm256_load_ps(inputPtr);
            inputPtr += 8;

            /* Increment indices */
            curIdxLo = _mm_add_epi32(curIdxLo, indexIncrementValues);
            curIdxHi = _mm_add_epi32(curIdxHi, indexIncrementValues);

            /* Compare using AVX 256-bit float compare */
            __m256 compareResults = _mm256_cmp_ps(currentValues, maxValues, _CMP_GT_OS);

            /* Split compare mask to 128-bit halves for integer blend (SSE4.1) */
            __m128i cmpLo = _mm_castps_si128(_mm256_castps256_ps128(compareResults));
            __m128i cmpHi = _mm_castps_si128(_mm256_extractf128_ps(compareResults, 1));

            /* Blend indices using SSE4.1 (available on all AVX CPUs) */
            maxIdxLo = _mm_blendv_epi8(maxIdxLo, curIdxLo, cmpLo);
            maxIdxHi = _mm_blendv_epi8(maxIdxHi, curIdxHi, cmpHi);

            /* Blend max values */
            maxValues = _mm256_blendv_ps(maxValues, currentValues, compareResults);
        }

        /* Extract results */
        _mm256_store_ps(maxValuesBuffer, maxValues);
        _mm_store_si128((__m128i*)maxIdxLoBuffer, maxIdxLo);
        _mm_store_si128((__m128i*)maxIdxHiBuffer, maxIdxHi);

        for (number = 0; number < 4; number++) {
            if (maxValuesBuffer[number] > max) {
                index = maxIdxLoBuffer[number];
                max = maxValuesBuffer[number];
            } else if (maxValuesBuffer[number] == max) {
                if (index > maxIdxLoBuffer[number])
                    index = maxIdxLoBuffer[number];
            }
        }
        for (number = 0; number < 4; number++) {
            if (maxValuesBuffer[number + 4] > max) {
                index = maxIdxHiBuffer[number];
                max = maxValuesBuffer[number + 4];
            } else if (maxValuesBuffer[number + 4] == max) {
                if (index > maxIdxHiBuffer[number])
                    index = maxIdxHiBuffer[number];
            }
        }

        number = eighthPoints * 8;
        for (; number < num_points; number++) {
            if (src0[number] > max) {
                index = number;
                max = src0[number];
            }
        }
        target[0] = index;
    }
}

#endif /*LV_HAVE_AVX*/


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_32f_index_max_32u_a_avx2(uint32_t* target, const float* src0, uint32_t num_points)
{
    if (num_points > 0) {
        uint32_t number = 0;
        const uint32_t quarterPoints = num_points / 8;

        const float* inputPtr = src0;

        __m256i indexIncrementValues = _mm256_set1_epi32(8);
        __m256i currentIndexes = _mm256_set_epi32(-1, -2, -3, -4, -5, -6, -7, -8);

        float max = src0[0];
        uint32_t index = 0;
        __m256 maxValues = _mm256_set1_ps(max);
        __m256i maxValuesIndex = _mm256_setzero_si256();
        __m256 compareResults;
        __m256 currentValues;

        __VOLK_ATTR_ALIGNED(32) float maxValuesBuffer[8];
        __VOLK_ATTR_ALIGNED(32) uint32_t maxIndexesBuffer[8];

        for (; number < quarterPoints; number++) {
            currentValues = _mm256_load_ps(inputPtr);
            inputPtr += 8;
            currentIndexes = _mm256_add_epi32(currentIndexes, indexIncrementValues);
            compareResults = _mm256_cmp_ps(currentValues, maxValues, _CMP_GT_OS);
            __m256i cmpMask = _mm256_castps_si256(compareResults);
            maxValuesIndex =
                _mm256_blendv_epi8(maxValuesIndex, currentIndexes, cmpMask);
            maxValues = _mm256_blendv_ps(maxValues, currentValues, compareResults);
        }

        // Calculate the largest value from the remaining 8 points
        _mm256_store_ps(maxValuesBuffer, maxValues);
        _mm256_store_si256((__m256i*)maxIndexesBuffer, maxValuesIndex);

        for (number = 0; number < 8; number++) {
            if (maxValuesBuffer[number] > max) {
                index = maxIndexesBuffer[number];
                max = maxValuesBuffer[number];
            } else if (maxValuesBuffer[number] == max) {
                if (index > maxIndexesBuffer[number])
                    index = maxIndexesBuffer[number];
            }
        }

        number = quarterPoints * 8;
        for (; number < num_points; number++) {
            if (src0[number] > max) {
                index = number;
                max = src0[number];
            }
        }
        target[0] = index;
    }
}

#endif /*LV_HAVE_AVX2*/


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void
volk_32f_index_max_32u_a_avx512f(uint32_t* target, const float* src0, uint32_t num_points)
{
    if (num_points > 0) {
        uint32_t number = 0;
        const uint32_t sixteenthPoints = num_points / 16;

        const float* inputPtr = src0;

        __m512i indexIncrementValues = _mm512_set1_epi32(16);
        __m512i currentIndexes = _mm512_set_epi32(
            -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16);

        float max = src0[0];
        uint32_t index = 0;
        __m512 maxValues = _mm512_set1_ps(max);
        __m512i maxValuesIndex = _mm512_setzero_si512();
        __mmask16 compareResults;
        __m512 currentValues;

        __VOLK_ATTR_ALIGNED(64) float maxValuesBuffer[16];
        __VOLK_ATTR_ALIGNED(64) uint32_t maxIndexesBuffer[16];

        for (; number < sixteenthPoints; number++) {
            currentValues = _mm512_load_ps(inputPtr);
            inputPtr += 16;
            currentIndexes = _mm512_add_epi32(currentIndexes, indexIncrementValues);
            compareResults = _mm512_cmp_ps_mask(currentValues, maxValues, _CMP_GT_OS);
            maxValuesIndex =
                _mm512_mask_blend_epi32(compareResults, maxValuesIndex, currentIndexes);
            maxValues = _mm512_mask_blend_ps(compareResults, maxValues, currentValues);
        }

        // Calculate the largest value from the remaining 16 points
        _mm512_store_ps(maxValuesBuffer, maxValues);
        _mm512_store_si512(maxIndexesBuffer, maxValuesIndex);

        for (number = 0; number < 16; number++) {
            if (maxValuesBuffer[number] > max) {
                index = maxIndexesBuffer[number];
                max = maxValuesBuffer[number];
            } else if (maxValuesBuffer[number] == max) {
                if (index > maxIndexesBuffer[number])
                    index = maxIndexesBuffer[number];
            }
        }

        number = sixteenthPoints * 16;
        for (; number < num_points; number++) {
            if (src0[number] > max) {
                index = number;
                max = src0[number];
            }
        }
        target[0] = index;
    }
}

#endif /*LV_HAVE_AVX512F*/

#endif /*INCLUDED_volk_32f_index_max_32u_a_H*/
