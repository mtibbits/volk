/* -*- c++ -*- */
/*
 * Copyright 2015 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_8u_x2_encodeframepolar_8u
 *
 * \b Overview
 *
 * Performs polar encoding on a frame of data using the Arikan polar transform.
 * The input data must be placed in the \p temp buffer before calling. The
 * encoded result is written to \p frame. The frame size must be a power of 2.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_8u_x2_encodeframepolar_8u(unsigned char* frame, unsigned char* temp,
 * unsigned int frame_size)
 * \endcode
 *
 * \b Inputs
 * \li temp: buffer containing the data to encode, also used as scratch space
 * during encoding (1 bit per byte).
 * \li frame_size: number of elements in the frame (must be a power of 2).
 *
 * \b Outputs
 * \li frame: the polar encoded frame.
 *
 * \b Example
 * \code
 * int frame_exp = 4;
 * int frame_size = 0x01 << frame_exp; // 16
 * unsigned int alignment = volk_get_alignment();
 *
 * unsigned char* frame =
 *     (unsigned char*)volk_malloc(sizeof(unsigned char) * frame_size, alignment);
 * unsigned char* temp =
 *     (unsigned char*)volk_malloc(sizeof(unsigned char) * frame_size, alignment);
 *
 * // Initialize temp with data to encode (1 bit per byte)
 * for (int i = 0; i < frame_size; ++i) {
 *     temp[i] = i % 2;
 * }
 *
 * volk_8u_x2_encodeframepolar_8u(frame, temp, frame_size);
 *
 * volk_free(frame);
 * volk_free(temp);
 * \endcode
 */

#ifndef VOLK_KERNELS_VOLK_VOLK_8U_X2_ENCODEFRAMEPOLAR_8U_U_H_
#define VOLK_KERNELS_VOLK_VOLK_8U_X2_ENCODEFRAMEPOLAR_8U_U_H_
#include <string.h>

static inline unsigned int log2_of_power_of_2(unsigned int val)
{
    // algorithm from: https://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
    static const unsigned int b[] = {
        0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0, 0xFF00FF00, 0xFFFF0000
    };

    unsigned int res = (val & b[0]) != 0;
    res |= ((val & b[4]) != 0) << 4;
    res |= ((val & b[3]) != 0) << 3;
    res |= ((val & b[2]) != 0) << 2;
    res |= ((val & b[1]) != 0) << 1;
    return res;
}

static inline void encodepolar_single_stage(unsigned char* frame_ptr,
                                            const unsigned char* temp_ptr,
                                            const unsigned int num_branches,
                                            const unsigned int frame_half)
{
    unsigned int branch, bit;
    for (branch = 0; branch < num_branches; ++branch) {
        for (bit = 0; bit < frame_half; ++bit) {
            *frame_ptr = *temp_ptr ^ *(temp_ptr + 1);
            *(frame_ptr + frame_half) = *(temp_ptr + 1);
            ++frame_ptr;
            temp_ptr += 2;
        }
        frame_ptr += frame_half;
    }
}

#ifdef LV_HAVE_GENERIC

static inline void volk_8u_x2_encodeframepolar_8u_generic(unsigned char* frame,
                                                          unsigned char* temp,
                                                          unsigned int frame_size)
{
    unsigned int stage = log2_of_power_of_2(frame_size);
    unsigned int frame_half = frame_size >> 1;
    unsigned int num_branches = 1;

    while (stage) {
        // encode stage
        encodepolar_single_stage(frame, temp, num_branches, frame_half);
        memcpy(temp, frame, sizeof(unsigned char) * frame_size);

        // update all the parameters.
        num_branches = num_branches << 1;
        frame_half = frame_half >> 1;
        --stage;
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSSE3
#include <tmmintrin.h>

static inline void volk_8u_x2_encodeframepolar_8u_u_ssse3(unsigned char* frame,
                                                          unsigned char* temp,
                                                          unsigned int frame_size)
{
    if (frame_size < 16) {
        volk_8u_x2_encodeframepolar_8u_generic(frame, temp, frame_size);
        return;
    }

    const unsigned int po2 = log2_of_power_of_2(frame_size);

    unsigned int stage = po2;
    unsigned char* frame_ptr = frame;
    const unsigned char* temp_ptr = temp;

    unsigned int frame_half = frame_size >> 1;
    unsigned int num_branches = 1;
    unsigned int branch;
    unsigned int bit;

    // prepare constants
    const __m128i mask_stage1 = _mm_set_epi8(0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF);

    // get some SIMD registers to play with.
    __m128i r_frame0, r_temp0, shifted;

    {
        __m128i r_frame1, r_temp1;
        const __m128i shuffle_separate =
            _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);

        while (stage > 4) {
            frame_ptr = frame;
            temp_ptr = temp;

            // for stage = 5 a branch has 32 elements. So upper stages are even bigger.
            for (branch = 0; branch < num_branches; ++branch) {
                for (bit = 0; bit < frame_half; bit += 16) {
                    r_temp0 = _mm_loadu_si128((const __m128i*)temp_ptr);
                    temp_ptr += 16;
                    r_temp1 = _mm_loadu_si128((const __m128i*)temp_ptr);
                    temp_ptr += 16;

                    shifted = _mm_srli_si128(r_temp0, 1);
                    shifted = _mm_and_si128(shifted, mask_stage1);
                    r_temp0 = _mm_xor_si128(shifted, r_temp0);
                    r_temp0 = _mm_shuffle_epi8(r_temp0, shuffle_separate);

                    shifted = _mm_srli_si128(r_temp1, 1);
                    shifted = _mm_and_si128(shifted, mask_stage1);
                    r_temp1 = _mm_xor_si128(shifted, r_temp1);
                    r_temp1 = _mm_shuffle_epi8(r_temp1, shuffle_separate);

                    r_frame0 = _mm_unpacklo_epi64(r_temp0, r_temp1);
                    _mm_storeu_si128((__m128i*)frame_ptr, r_frame0);

                    r_frame1 = _mm_unpackhi_epi64(r_temp0, r_temp1);
                    _mm_storeu_si128((__m128i*)(frame_ptr + frame_half), r_frame1);
                    frame_ptr += 16;
                }

                frame_ptr += frame_half;
            }
            memcpy(temp, frame, sizeof(unsigned char) * frame_size);

            num_branches = num_branches << 1;
            frame_half = frame_half >> 1;
            stage--;
        }
    }

    // This last part requires at least 16-bit frames.
    // Smaller frames are useless for SIMD optimization anyways. Just choose GENERIC!

    // reset pointers to correct positions.
    frame_ptr = frame;
    temp_ptr = temp;

    // prefetch first chunk
    __VOLK_PREFETCH(temp_ptr);

    const __m128i shuffle_stage4 =
        _mm_setr_epi8(0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15);
    const __m128i mask_stage4 = _mm_set_epi8(0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF);
    const __m128i mask_stage3 = _mm_set_epi8(0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF);
    const __m128i mask_stage2 = _mm_set_epi8(0x0,
                                             0x0,
                                             0xFF,
                                             0xFF,
                                             0x0,
                                             0x0,
                                             0xFF,
                                             0xFF,
                                             0x0,
                                             0x0,
                                             0xFF,
                                             0xFF,
                                             0x0,
                                             0x0,
                                             0xFF,
                                             0xFF);

    for (branch = 0; branch < num_branches; ++branch) {
        r_temp0 = _mm_loadu_si128((const __m128i*)temp_ptr);

        // prefetch next chunk
        temp_ptr += 16;
        __VOLK_PREFETCH(temp_ptr);

        // shuffle once for bit-reversal.
        r_temp0 = _mm_shuffle_epi8(r_temp0, shuffle_stage4);

        shifted = _mm_srli_si128(r_temp0, 8);
        shifted = _mm_and_si128(shifted, mask_stage4);
        r_frame0 = _mm_xor_si128(shifted, r_temp0);

        shifted = _mm_srli_si128(r_frame0, 4);
        shifted = _mm_and_si128(shifted, mask_stage3);
        r_frame0 = _mm_xor_si128(shifted, r_frame0);

        shifted = _mm_srli_si128(r_frame0, 2);
        shifted = _mm_and_si128(shifted, mask_stage2);
        r_frame0 = _mm_xor_si128(shifted, r_frame0);

        shifted = _mm_srli_si128(r_frame0, 1);
        shifted = _mm_and_si128(shifted, mask_stage1);
        r_frame0 = _mm_xor_si128(shifted, r_frame0);

        // store result of chunk.
        _mm_storeu_si128((__m128i*)frame_ptr, r_frame0);
        frame_ptr += 16;
    }
}

#endif /* LV_HAVE_SSSE3 */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_8u_x2_encodeframepolar_8u_u_avx2(unsigned char* frame,
                                                         unsigned char* temp,
                                                         unsigned int frame_size)
{
    if (frame_size < 32) {
        volk_8u_x2_encodeframepolar_8u_generic(frame, temp, frame_size);
        return;
    }

    const unsigned int po2 = log2_of_power_of_2(frame_size);

    unsigned int stage = po2;
    unsigned char* frame_ptr = frame;
    const unsigned char* temp_ptr = temp;

    unsigned int frame_half = frame_size >> 1;
    unsigned int num_branches = 1;
    unsigned int branch;
    unsigned int bit;

    // prepare constants
    const __m256i mask_stage1 = _mm256_set_epi8(0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF);

    const __m128i mask_stage0 = _mm_set_epi8(0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF);
    // get some SIMD registers to play with.
    __m256i r_frame0, r_temp0, shifted;
    __m128i r_temp2, r_frame2, shifted2;
    {
        __m256i r_frame1, r_temp1;
        __m128i r_frame3, r_temp3;
        const __m256i shuffle_separate = _mm256_setr_epi8(0,
                                                          2,
                                                          4,
                                                          6,
                                                          8,
                                                          10,
                                                          12,
                                                          14,
                                                          1,
                                                          3,
                                                          5,
                                                          7,
                                                          9,
                                                          11,
                                                          13,
                                                          15,
                                                          0,
                                                          2,
                                                          4,
                                                          6,
                                                          8,
                                                          10,
                                                          12,
                                                          14,
                                                          1,
                                                          3,
                                                          5,
                                                          7,
                                                          9,
                                                          11,
                                                          13,
                                                          15);
        const __m128i shuffle_separate128 =
            _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);

        while (stage > 4) {
            frame_ptr = frame;
            temp_ptr = temp;

            // for stage = 5 a branch has 32 elements. So upper stages are even bigger.
            for (branch = 0; branch < num_branches; ++branch) {
                for (bit = 0; bit < frame_half; bit += 32) {
                    if ((frame_half - bit) <
                        32) // if only 16 bits remaining in frame, not 32
                    {
                        r_temp2 = _mm_loadu_si128((const __m128i*)temp_ptr);
                        temp_ptr += 16;
                        r_temp3 = _mm_loadu_si128((const __m128i*)temp_ptr);
                        temp_ptr += 16;

                        shifted2 = _mm_srli_si128(r_temp2, 1);
                        shifted2 = _mm_and_si128(shifted2, mask_stage0);
                        r_temp2 = _mm_xor_si128(shifted2, r_temp2);
                        r_temp2 = _mm_shuffle_epi8(r_temp2, shuffle_separate128);

                        shifted2 = _mm_srli_si128(r_temp3, 1);
                        shifted2 = _mm_and_si128(shifted2, mask_stage0);
                        r_temp3 = _mm_xor_si128(shifted2, r_temp3);
                        r_temp3 = _mm_shuffle_epi8(r_temp3, shuffle_separate128);

                        r_frame2 = _mm_unpacklo_epi64(r_temp2, r_temp3);
                        _mm_storeu_si128((__m128i*)frame_ptr, r_frame2);

                        r_frame3 = _mm_unpackhi_epi64(r_temp2, r_temp3);
                        _mm_storeu_si128((__m128i*)(frame_ptr + frame_half), r_frame3);
                        frame_ptr += 16;
                        break;
                    }
                    r_temp0 = _mm256_loadu_si256((const __m256i*)temp_ptr);
                    temp_ptr += 32;
                    r_temp1 = _mm256_loadu_si256((const __m256i*)temp_ptr);
                    temp_ptr += 32;

                    shifted = _mm256_srli_si256(r_temp0, 1); // operate on 128 bit lanes
                    shifted = _mm256_and_si256(shifted, mask_stage1);
                    r_temp0 = _mm256_xor_si256(shifted, r_temp0);
                    r_temp0 = _mm256_shuffle_epi8(r_temp0, shuffle_separate);

                    shifted = _mm256_srli_si256(r_temp1, 1);
                    shifted = _mm256_and_si256(shifted, mask_stage1);
                    r_temp1 = _mm256_xor_si256(shifted, r_temp1);
                    r_temp1 = _mm256_shuffle_epi8(r_temp1, shuffle_separate);

                    r_frame0 = _mm256_unpacklo_epi64(r_temp0, r_temp1);
                    r_temp1 = _mm256_unpackhi_epi64(r_temp0, r_temp1);
                    r_frame0 = _mm256_permute4x64_epi64(r_frame0, 0xd8);
                    r_frame1 = _mm256_permute4x64_epi64(r_temp1, 0xd8);

                    _mm256_storeu_si256((__m256i*)frame_ptr, r_frame0);

                    _mm256_storeu_si256((__m256i*)(frame_ptr + frame_half), r_frame1);
                    frame_ptr += 32;
                }

                frame_ptr += frame_half;
            }
            memcpy(temp, frame, sizeof(unsigned char) * frame_size);

            num_branches = num_branches << 1;
            frame_half = frame_half >> 1;
            stage--;
        }
    }

    // This last part requires at least 32-bit frames.
    // Smaller frames are useless for SIMD optimization anyways. Just choose GENERIC!

    // reset pointers to correct positions.
    frame_ptr = frame;
    temp_ptr = temp;

    // prefetch first chunk
    __VOLK_PREFETCH(temp_ptr);

    const __m256i shuffle_stage4 = _mm256_setr_epi8(0,
                                                    8,
                                                    4,
                                                    12,
                                                    2,
                                                    10,
                                                    6,
                                                    14,
                                                    1,
                                                    9,
                                                    5,
                                                    13,
                                                    3,
                                                    11,
                                                    7,
                                                    15,
                                                    0,
                                                    8,
                                                    4,
                                                    12,
                                                    2,
                                                    10,
                                                    6,
                                                    14,
                                                    1,
                                                    9,
                                                    5,
                                                    13,
                                                    3,
                                                    11,
                                                    7,
                                                    15);
    const __m256i mask_stage4 = _mm256_set_epi8(0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF);
    const __m256i mask_stage3 = _mm256_set_epi8(0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF);
    const __m256i mask_stage2 = _mm256_set_epi8(0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF);

    for (branch = 0; branch < num_branches / 2; ++branch) {
        r_temp0 = _mm256_loadu_si256((const __m256i*)temp_ptr);

        // prefetch next chunk
        temp_ptr += 32;
        __VOLK_PREFETCH(temp_ptr);

        // shuffle once for bit-reversal.
        r_temp0 = _mm256_shuffle_epi8(r_temp0, shuffle_stage4);

        shifted = _mm256_srli_si256(r_temp0, 8); // 128 bit lanes
        shifted = _mm256_and_si256(shifted, mask_stage4);
        r_frame0 = _mm256_xor_si256(shifted, r_temp0);


        shifted = _mm256_srli_si256(r_frame0, 4);
        shifted = _mm256_and_si256(shifted, mask_stage3);
        r_frame0 = _mm256_xor_si256(shifted, r_frame0);

        shifted = _mm256_srli_si256(r_frame0, 2);
        shifted = _mm256_and_si256(shifted, mask_stage2);
        r_frame0 = _mm256_xor_si256(shifted, r_frame0);

        shifted = _mm256_srli_si256(r_frame0, 1);
        shifted = _mm256_and_si256(shifted, mask_stage1);
        r_frame0 = _mm256_xor_si256(shifted, r_frame0);

        // store result of chunk.
        _mm256_storeu_si256((__m256i*)frame_ptr, r_frame0);
        frame_ptr += 32;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_8u_x2_encodeframepolar_8u_u_avx512bw(unsigned char* frame,
                                                               unsigned char* temp,
                                                               unsigned int frame_size)
{
    if (frame_size < 64) {
        volk_8u_x2_encodeframepolar_8u_generic(frame, temp, frame_size);
        return;
    }

    const unsigned int po2 = log2_of_power_of_2(frame_size);
    unsigned int stage = po2;
    unsigned char* frame_ptr = frame;
    const unsigned char* temp_ptr = temp;
    unsigned int frame_half = frame_size >> 1;
    unsigned int num_branches = 1;
    unsigned int branch;
    unsigned int bit;

    const __m512i mask_s1_512 = _mm512_set1_epi16(0x00FF);
    __m512i r_frame0, r_temp0, shifted;

    {
        __m512i r_frame1, r_temp1;
        const __m128i shuffle_sep_128 =
            _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
        const __m512i shuffle_sep_512 = _mm512_broadcast_i32x4(shuffle_sep_128);
        const __m256i shuffle_sep_256 = _mm256_broadcastsi128_si256(shuffle_sep_128);
        const __m128i mask_s1_128 = _mm_set1_epi16(0x00FF);
        const __m256i mask_s1_256 = _mm256_set1_epi16(0x00FF);
        const __m512i perm_deint =
            _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);

        while (stage > 4) {
            frame_ptr = frame;
            temp_ptr = temp;

            for (branch = 0; branch < num_branches; ++branch) {
                for (bit = 0; bit < frame_half; bit += 64) {
                    unsigned int remaining = frame_half - bit;
                    if (remaining < 64) {
                        if (remaining < 32) {
                            /* 16-byte SSE fallback */
                            __m128i rt0 =
                                _mm_loadu_si128((const __m128i*)temp_ptr);
                            temp_ptr += 16;
                            __m128i rt1 =
                                _mm_loadu_si128((const __m128i*)temp_ptr);
                            temp_ptr += 16;
                            __m128i sh = _mm_srli_si128(rt0, 1);
                            sh = _mm_and_si128(sh, mask_s1_128);
                            rt0 = _mm_xor_si128(sh, rt0);
                            rt0 = _mm_shuffle_epi8(rt0, shuffle_sep_128);
                            sh = _mm_srli_si128(rt1, 1);
                            sh = _mm_and_si128(sh, mask_s1_128);
                            rt1 = _mm_xor_si128(sh, rt1);
                            rt1 = _mm_shuffle_epi8(rt1, shuffle_sep_128);
                            _mm_storeu_si128(
                                (__m128i*)frame_ptr,
                                _mm_unpacklo_epi64(rt0, rt1));
                            _mm_storeu_si128(
                                (__m128i*)(frame_ptr + frame_half),
                                _mm_unpackhi_epi64(rt0, rt1));
                            frame_ptr += 16;
                        } else {
                            /* 32-byte AVX2 fallback */
                            __m256i rt0 =
                                _mm256_loadu_si256((const __m256i*)temp_ptr);
                            temp_ptr += 32;
                            __m256i rt1 =
                                _mm256_loadu_si256((const __m256i*)temp_ptr);
                            temp_ptr += 32;
                            __m256i sh = _mm256_srli_si256(rt0, 1);
                            sh = _mm256_and_si256(sh, mask_s1_256);
                            rt0 = _mm256_xor_si256(sh, rt0);
                            rt0 = _mm256_shuffle_epi8(rt0, shuffle_sep_256);
                            sh = _mm256_srli_si256(rt1, 1);
                            sh = _mm256_and_si256(sh, mask_s1_256);
                            rt1 = _mm256_xor_si256(sh, rt1);
                            rt1 = _mm256_shuffle_epi8(rt1, shuffle_sep_256);
                            __m256i rf0 = _mm256_unpacklo_epi64(rt0, rt1);
                            rt1 = _mm256_unpackhi_epi64(rt0, rt1);
                            rf0 = _mm256_permute4x64_epi64(rf0, 0xd8);
                            __m256i rf1 =
                                _mm256_permute4x64_epi64(rt1, 0xd8);
                            _mm256_storeu_si256((__m256i*)frame_ptr, rf0);
                            _mm256_storeu_si256(
                                (__m256i*)(frame_ptr + frame_half), rf1);
                            frame_ptr += 32;
                        }
                        break;
                    }

                    r_temp0 = _mm512_loadu_si512((const __m512i*)temp_ptr);
                    temp_ptr += 64;
                    r_temp1 = _mm512_loadu_si512((const __m512i*)temp_ptr);
                    temp_ptr += 64;

                    shifted = _mm512_bsrli_epi128(r_temp0, 1);
                    shifted = _mm512_and_si512(shifted, mask_s1_512);
                    r_temp0 = _mm512_xor_si512(shifted, r_temp0);
                    r_temp0 = _mm512_shuffle_epi8(r_temp0, shuffle_sep_512);

                    shifted = _mm512_bsrli_epi128(r_temp1, 1);
                    shifted = _mm512_and_si512(shifted, mask_s1_512);
                    r_temp1 = _mm512_xor_si512(shifted, r_temp1);
                    r_temp1 = _mm512_shuffle_epi8(r_temp1, shuffle_sep_512);

                    r_frame0 = _mm512_unpacklo_epi64(r_temp0, r_temp1);
                    r_temp1 = _mm512_unpackhi_epi64(r_temp0, r_temp1);
                    r_frame0 = _mm512_permutexvar_epi64(perm_deint, r_frame0);
                    r_frame1 = _mm512_permutexvar_epi64(perm_deint, r_temp1);

                    _mm512_storeu_si512((__m512i*)frame_ptr, r_frame0);
                    _mm512_storeu_si512(
                        (__m512i*)(frame_ptr + frame_half), r_frame1);
                    frame_ptr += 64;
                }

                frame_ptr += frame_half;
            }
            memcpy(temp, frame, sizeof(unsigned char) * frame_size);

            num_branches = num_branches << 1;
            frame_half = frame_half >> 1;
            stage--;
        }
    }

    /* Lower 4 stages: process 64 bytes (4 sub-frames of 16) at a time */
    frame_ptr = frame;
    temp_ptr = temp;
    __VOLK_PREFETCH(temp_ptr);

    const __m128i shuffle_s4_128 =
        _mm_setr_epi8(0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15);
    const __m512i shuffle_s4 = _mm512_broadcast_i32x4(shuffle_s4_128);
    const __m512i mask_s4 =
        _mm512_set_epi64(0LL, -1LL, 0LL, -1LL, 0LL, -1LL, 0LL, -1LL);
    const __m512i mask_s3 = _mm512_set1_epi64(0x00000000FFFFFFFFLL);
    const __m512i mask_s2 = _mm512_set1_epi32(0x0000FFFF);

    for (branch = 0; branch < num_branches / 4; ++branch) {
        r_temp0 = _mm512_loadu_si512((const __m512i*)temp_ptr);

        temp_ptr += 64;
        __VOLK_PREFETCH(temp_ptr);

        r_temp0 = _mm512_shuffle_epi8(r_temp0, shuffle_s4);

        shifted = _mm512_bsrli_epi128(r_temp0, 8);
        shifted = _mm512_and_si512(shifted, mask_s4);
        r_frame0 = _mm512_xor_si512(shifted, r_temp0);

        shifted = _mm512_bsrli_epi128(r_frame0, 4);
        shifted = _mm512_and_si512(shifted, mask_s3);
        r_frame0 = _mm512_xor_si512(shifted, r_frame0);

        shifted = _mm512_bsrli_epi128(r_frame0, 2);
        shifted = _mm512_and_si512(shifted, mask_s2);
        r_frame0 = _mm512_xor_si512(shifted, r_frame0);

        shifted = _mm512_bsrli_epi128(r_frame0, 1);
        shifted = _mm512_and_si512(shifted, mask_s1_512);
        r_frame0 = _mm512_xor_si512(shifted, r_frame0);

        _mm512_storeu_si512((__m512i*)frame_ptr, r_frame0);
        frame_ptr += 64;
    }
}
#endif /* LV_HAVE_AVX512BW */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_8u_x2_encodeframepolar_8u_rvv(unsigned char* frame,
                                                      unsigned char* temp,
                                                      unsigned int frame_size)
{
    unsigned int stage = log2_of_power_of_2(frame_size);
    unsigned int frame_half = frame_size >> 1;
    unsigned int num_branches = 1;

    while (stage) {
        // encode stage
        if (frame_half < 8) {
            encodepolar_single_stage(frame, temp, num_branches, frame_half);
        } else {
            const unsigned char* in = temp;
            unsigned char* out = frame;
            for (size_t branch = 0; branch < num_branches; ++branch) {
                size_t n = frame_half;
                for (size_t vl; n > 0; n -= vl, in += vl * 2, out += vl) {
                    vl = __riscv_vsetvl_e8m1(n);
                    vuint16m2_t vc = __riscv_vle16_v_u16m2((const uint16_t*)in, vl);
                    vuint8m1_t v1 = __riscv_vnsrl(vc, 0, vl);
                    vuint8m1_t v2 = __riscv_vnsrl(vc, 8, vl);
                    __riscv_vse8(out, __riscv_vxor(v1, v2, vl), vl);
                    __riscv_vse8(out + frame_half, v2, vl);
                }
                out += frame_half;
            }
        }
        memcpy(temp, frame, sizeof(unsigned char) * frame_size);

        // update all the parameters.
        num_branches = num_branches << 1;
        frame_half = frame_half >> 1;
        --stage;
    }
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_8u_x2_encodeframepolar_8u_rvvseg(unsigned char* frame,
                                                         unsigned char* temp,
                                                         unsigned int frame_size)
{
    unsigned int stage = log2_of_power_of_2(frame_size);
    unsigned int frame_half = frame_size >> 1;
    unsigned int num_branches = 1;

    while (stage) {
        // encode stage
        if (frame_half < 8) {
            encodepolar_single_stage(frame, temp, num_branches, frame_half);
        } else {
            const unsigned char* in = temp;
            unsigned char* out = frame;
            for (size_t branch = 0; branch < num_branches; ++branch) {
                size_t n = frame_half;
                for (size_t vl; n > 0; n -= vl, in += vl * 2, out += vl) {
                    vl = __riscv_vsetvl_e8m1(n);
                    vuint8m1x2_t vc = __riscv_vlseg2e8_v_u8m1x2(in, vl);
                    vuint8m1_t v1 = __riscv_vget_u8m1(vc, 0);
                    vuint8m1_t v2 = __riscv_vget_u8m1(vc, 1);
                    __riscv_vse8(out, __riscv_vxor(v1, v2, vl), vl);
                    __riscv_vse8(out + frame_half, v2, vl);
                }
                out += frame_half;
            }
        }
        memcpy(temp, frame, sizeof(unsigned char) * frame_size);

        // update all the parameters.
        num_branches = num_branches << 1;
        frame_half = frame_half >> 1;
        --stage;
    }
}
#endif /* LV_HAVE_RVVSEG */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_8u_x2_encodeframepolar_8u_neon(unsigned char* frame,
                                                        unsigned char* temp,
                                                        unsigned int frame_size)
{
    if (frame_size < 16) {
        volk_8u_x2_encodeframepolar_8u_generic(frame, temp, frame_size);
        return;
    }

    const unsigned int po2 = log2_of_power_of_2(frame_size);

    unsigned int stage = po2;
    unsigned char* frame_ptr = frame;
    const unsigned char* temp_ptr = temp;

    unsigned int frame_half = frame_size >> 1;
    unsigned int num_branches = 1;
    unsigned int branch;
    unsigned int bit;

    /* Upper stages: deinterleave pairs of bytes, XOR even with odd */
    while (stage > 4) {
        frame_ptr = frame;
        temp_ptr = temp;

        for (branch = 0; branch < num_branches; ++branch) {
            for (bit = 0; bit < frame_half; bit += 16) {
                /* Load 32 bytes as interleaved even/odd pairs */
                uint8x16x2_t pairs = vld2q_u8(temp_ptr);
                temp_ptr += 32;

                /* even[i] = temp[2i], odd[i] = temp[2i+1] */
                /* frame[i] = even[i] ^ odd[i], frame[i + half] = odd[i] */
                uint8x16_t xored = veorq_u8(pairs.val[0], pairs.val[1]);
                vst1q_u8(frame_ptr, xored);
                vst1q_u8(frame_ptr + frame_half, pairs.val[1]);
                frame_ptr += 16;
            }

            frame_ptr += frame_half;
        }
        memcpy(temp, frame, sizeof(unsigned char) * frame_size);

        num_branches = num_branches << 1;
        frame_half = frame_half >> 1;
        stage--;
    }

    /* Final 4 stages within each 16-byte block */
    frame_ptr = frame;
    temp_ptr = temp;

    /* Lookup table for the bit-reversal shuffle (same permutation as SSSE3) */
    static const uint8_t shuffle_tbl[16] = {
        0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15
    };
    const uint8x16_t shuffle_idx = vld1q_u8(shuffle_tbl);

    /* Masks for progressive XOR stages */
    static const uint8_t mask4_tbl[16] = {
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    };
    static const uint8_t mask3_tbl[16] = {
        0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00
    };
    static const uint8_t mask2_tbl[16] = {
        0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00,
        0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00
    };
    static const uint8_t mask1_tbl[16] = {
        0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00,
        0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00
    };
    const uint8x16_t mask_stage4 = vld1q_u8(mask4_tbl);
    const uint8x16_t mask_stage3 = vld1q_u8(mask3_tbl);
    const uint8x16_t mask_stage2 = vld1q_u8(mask2_tbl);
    const uint8x16_t mask_stage1 = vld1q_u8(mask1_tbl);

    for (branch = 0; branch < num_branches; ++branch) {
        uint8x16_t r = vld1q_u8(temp_ptr);
        temp_ptr += 16;

        /* Bit-reversal shuffle using vtbl (NEON table lookup) */
        uint8x16_t r_shuffled = vqtbl1q_u8(r, shuffle_idx);

        /* Stage 4: XOR upper 8 bytes into lower 8 bytes */
        uint8x16_t shifted = vandq_u8(vextq_u8(r_shuffled, vdupq_n_u8(0), 8), mask_stage4);
        r_shuffled = veorq_u8(shifted, r_shuffled);

        /* Stage 3: XOR bytes shifted by 4 */
        shifted = vandq_u8(vextq_u8(r_shuffled, vdupq_n_u8(0), 4), mask_stage3);
        r_shuffled = veorq_u8(shifted, r_shuffled);

        /* Stage 2: XOR bytes shifted by 2 */
        shifted = vandq_u8(vextq_u8(r_shuffled, vdupq_n_u8(0), 2), mask_stage2);
        r_shuffled = veorq_u8(shifted, r_shuffled);

        /* Stage 1: XOR bytes shifted by 1 */
        shifted = vandq_u8(vextq_u8(r_shuffled, vdupq_n_u8(0), 1), mask_stage1);
        r_shuffled = veorq_u8(shifted, r_shuffled);

        vst1q_u8(frame_ptr, r_shuffled);
        frame_ptr += 16;
    }
}
#endif /* LV_HAVE_NEON */

#endif /* VOLK_KERNELS_VOLK_VOLK_8U_X2_ENCODEFRAMEPOLAR_8U_U_H_ */

#ifndef VOLK_KERNELS_VOLK_VOLK_8U_X2_ENCODEFRAMEPOLAR_8U_A_H_
#define VOLK_KERNELS_VOLK_VOLK_8U_X2_ENCODEFRAMEPOLAR_8U_A_H_

#ifdef LV_HAVE_SSSE3
#include <tmmintrin.h>

static inline void volk_8u_x2_encodeframepolar_8u_a_ssse3(unsigned char* frame,
                                                          unsigned char* temp,
                                                          unsigned int frame_size)
{
    if (frame_size < 16) {
        volk_8u_x2_encodeframepolar_8u_generic(frame, temp, frame_size);
        return;
    }

    const unsigned int po2 = log2_of_power_of_2(frame_size);

    unsigned int stage = po2;
    unsigned char* frame_ptr = frame;
    const unsigned char* temp_ptr = temp;

    unsigned int frame_half = frame_size >> 1;
    unsigned int num_branches = 1;
    unsigned int branch;
    unsigned int bit;

    // prepare constants
    const __m128i mask_stage1 = _mm_set_epi8(0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF);

    // get some SIMD registers to play with.
    __m128i r_frame0, r_temp0, shifted;

    {
        __m128i r_frame1, r_temp1;
        const __m128i shuffle_separate =
            _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);

        while (stage > 4) {
            frame_ptr = frame;
            temp_ptr = temp;

            // for stage = 5 a branch has 32 elements. So upper stages are even bigger.
            for (branch = 0; branch < num_branches; ++branch) {
                for (bit = 0; bit < frame_half; bit += 16) {
                    r_temp0 = _mm_load_si128((const __m128i*)temp_ptr);
                    temp_ptr += 16;
                    r_temp1 = _mm_load_si128((const __m128i*)temp_ptr);
                    temp_ptr += 16;

                    shifted = _mm_srli_si128(r_temp0, 1);
                    shifted = _mm_and_si128(shifted, mask_stage1);
                    r_temp0 = _mm_xor_si128(shifted, r_temp0);
                    r_temp0 = _mm_shuffle_epi8(r_temp0, shuffle_separate);

                    shifted = _mm_srli_si128(r_temp1, 1);
                    shifted = _mm_and_si128(shifted, mask_stage1);
                    r_temp1 = _mm_xor_si128(shifted, r_temp1);
                    r_temp1 = _mm_shuffle_epi8(r_temp1, shuffle_separate);

                    r_frame0 = _mm_unpacklo_epi64(r_temp0, r_temp1);
                    _mm_store_si128((__m128i*)frame_ptr, r_frame0);

                    r_frame1 = _mm_unpackhi_epi64(r_temp0, r_temp1);
                    _mm_store_si128((__m128i*)(frame_ptr + frame_half), r_frame1);
                    frame_ptr += 16;
                }

                frame_ptr += frame_half;
            }
            memcpy(temp, frame, sizeof(unsigned char) * frame_size);

            num_branches = num_branches << 1;
            frame_half = frame_half >> 1;
            stage--;
        }
    }

    // This last part requires at least 16-bit frames.
    // Smaller frames are useless for SIMD optimization anyways. Just choose GENERIC!

    // reset pointers to correct positions.
    frame_ptr = frame;
    temp_ptr = temp;

    // prefetch first chunk
    __VOLK_PREFETCH(temp_ptr);

    const __m128i shuffle_stage4 =
        _mm_setr_epi8(0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15);
    const __m128i mask_stage4 = _mm_set_epi8(0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF);
    const __m128i mask_stage3 = _mm_set_epi8(0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF);
    const __m128i mask_stage2 = _mm_set_epi8(0x0,
                                             0x0,
                                             0xFF,
                                             0xFF,
                                             0x0,
                                             0x0,
                                             0xFF,
                                             0xFF,
                                             0x0,
                                             0x0,
                                             0xFF,
                                             0xFF,
                                             0x0,
                                             0x0,
                                             0xFF,
                                             0xFF);

    for (branch = 0; branch < num_branches; ++branch) {
        r_temp0 = _mm_load_si128((const __m128i*)temp_ptr);

        // prefetch next chunk
        temp_ptr += 16;
        __VOLK_PREFETCH(temp_ptr);

        // shuffle once for bit-reversal.
        r_temp0 = _mm_shuffle_epi8(r_temp0, shuffle_stage4);

        shifted = _mm_srli_si128(r_temp0, 8);
        shifted = _mm_and_si128(shifted, mask_stage4);
        r_frame0 = _mm_xor_si128(shifted, r_temp0);

        shifted = _mm_srli_si128(r_frame0, 4);
        shifted = _mm_and_si128(shifted, mask_stage3);
        r_frame0 = _mm_xor_si128(shifted, r_frame0);

        shifted = _mm_srli_si128(r_frame0, 2);
        shifted = _mm_and_si128(shifted, mask_stage2);
        r_frame0 = _mm_xor_si128(shifted, r_frame0);

        shifted = _mm_srli_si128(r_frame0, 1);
        shifted = _mm_and_si128(shifted, mask_stage1);
        r_frame0 = _mm_xor_si128(shifted, r_frame0);

        // store result of chunk.
        _mm_store_si128((__m128i*)frame_ptr, r_frame0);
        frame_ptr += 16;
    }
}
#endif /* LV_HAVE_SSSE3 */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_8u_x2_encodeframepolar_8u_a_avx2(unsigned char* frame,
                                                         unsigned char* temp,
                                                         unsigned int frame_size)
{
    if (frame_size < 32) {
        volk_8u_x2_encodeframepolar_8u_generic(frame, temp, frame_size);
        return;
    }

    const unsigned int po2 = log2_of_power_of_2(frame_size);

    unsigned int stage = po2;
    unsigned char* frame_ptr = frame;
    const unsigned char* temp_ptr = temp;

    unsigned int frame_half = frame_size >> 1;
    unsigned int num_branches = 1;
    unsigned int branch;
    unsigned int bit;

    // prepare constants
    const __m256i mask_stage1 = _mm256_set_epi8(0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF);

    const __m128i mask_stage0 = _mm_set_epi8(0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF);
    // get some SIMD registers to play with.
    __m256i r_frame0, r_temp0, shifted;
    __m128i r_temp2, r_frame2, shifted2;
    {
        __m256i r_frame1, r_temp1;
        __m128i r_frame3, r_temp3;
        const __m256i shuffle_separate = _mm256_setr_epi8(0,
                                                          2,
                                                          4,
                                                          6,
                                                          8,
                                                          10,
                                                          12,
                                                          14,
                                                          1,
                                                          3,
                                                          5,
                                                          7,
                                                          9,
                                                          11,
                                                          13,
                                                          15,
                                                          0,
                                                          2,
                                                          4,
                                                          6,
                                                          8,
                                                          10,
                                                          12,
                                                          14,
                                                          1,
                                                          3,
                                                          5,
                                                          7,
                                                          9,
                                                          11,
                                                          13,
                                                          15);
        const __m128i shuffle_separate128 =
            _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);

        while (stage > 4) {
            frame_ptr = frame;
            temp_ptr = temp;

            // for stage = 5 a branch has 32 elements. So upper stages are even bigger.
            for (branch = 0; branch < num_branches; ++branch) {
                for (bit = 0; bit < frame_half; bit += 32) {
                    if ((frame_half - bit) <
                        32) // if only 16 bits remaining in frame, not 32
                    {
                        r_temp2 = _mm_load_si128((const __m128i*)temp_ptr);
                        temp_ptr += 16;
                        r_temp3 = _mm_load_si128((const __m128i*)temp_ptr);
                        temp_ptr += 16;

                        shifted2 = _mm_srli_si128(r_temp2, 1);
                        shifted2 = _mm_and_si128(shifted2, mask_stage0);
                        r_temp2 = _mm_xor_si128(shifted2, r_temp2);
                        r_temp2 = _mm_shuffle_epi8(r_temp2, shuffle_separate128);

                        shifted2 = _mm_srli_si128(r_temp3, 1);
                        shifted2 = _mm_and_si128(shifted2, mask_stage0);
                        r_temp3 = _mm_xor_si128(shifted2, r_temp3);
                        r_temp3 = _mm_shuffle_epi8(r_temp3, shuffle_separate128);

                        r_frame2 = _mm_unpacklo_epi64(r_temp2, r_temp3);
                        _mm_store_si128((__m128i*)frame_ptr, r_frame2);

                        r_frame3 = _mm_unpackhi_epi64(r_temp2, r_temp3);
                        _mm_store_si128((__m128i*)(frame_ptr + frame_half), r_frame3);
                        frame_ptr += 16;
                        break;
                    }
                    r_temp0 = _mm256_load_si256((const __m256i*)temp_ptr);
                    temp_ptr += 32;
                    r_temp1 = _mm256_load_si256((const __m256i*)temp_ptr);
                    temp_ptr += 32;

                    shifted = _mm256_srli_si256(r_temp0, 1); // operate on 128 bit lanes
                    shifted = _mm256_and_si256(shifted, mask_stage1);
                    r_temp0 = _mm256_xor_si256(shifted, r_temp0);
                    r_temp0 = _mm256_shuffle_epi8(r_temp0, shuffle_separate);

                    shifted = _mm256_srli_si256(r_temp1, 1);
                    shifted = _mm256_and_si256(shifted, mask_stage1);
                    r_temp1 = _mm256_xor_si256(shifted, r_temp1);
                    r_temp1 = _mm256_shuffle_epi8(r_temp1, shuffle_separate);

                    r_frame0 = _mm256_unpacklo_epi64(r_temp0, r_temp1);
                    r_temp1 = _mm256_unpackhi_epi64(r_temp0, r_temp1);
                    r_frame0 = _mm256_permute4x64_epi64(r_frame0, 0xd8);
                    r_frame1 = _mm256_permute4x64_epi64(r_temp1, 0xd8);

                    _mm256_store_si256((__m256i*)frame_ptr, r_frame0);

                    _mm256_store_si256((__m256i*)(frame_ptr + frame_half), r_frame1);
                    frame_ptr += 32;
                }

                frame_ptr += frame_half;
            }
            memcpy(temp, frame, sizeof(unsigned char) * frame_size);

            num_branches = num_branches << 1;
            frame_half = frame_half >> 1;
            stage--;
        }
    }

    // This last part requires at least 32-bit frames.
    // Smaller frames are useless for SIMD optimization anyways. Just choose GENERIC!

    // reset pointers to correct positions.
    frame_ptr = frame;
    temp_ptr = temp;

    // prefetch first chunk.
    __VOLK_PREFETCH(temp_ptr);

    const __m256i shuffle_stage4 = _mm256_setr_epi8(0,
                                                    8,
                                                    4,
                                                    12,
                                                    2,
                                                    10,
                                                    6,
                                                    14,
                                                    1,
                                                    9,
                                                    5,
                                                    13,
                                                    3,
                                                    11,
                                                    7,
                                                    15,
                                                    0,
                                                    8,
                                                    4,
                                                    12,
                                                    2,
                                                    10,
                                                    6,
                                                    14,
                                                    1,
                                                    9,
                                                    5,
                                                    13,
                                                    3,
                                                    11,
                                                    7,
                                                    15);
    const __m256i mask_stage4 = _mm256_set_epi8(0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF);
    const __m256i mask_stage3 = _mm256_set_epi8(0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF);
    const __m256i mask_stage2 = _mm256_set_epi8(0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF);

    for (branch = 0; branch < num_branches / 2; ++branch) {
        r_temp0 = _mm256_load_si256((const __m256i*)temp_ptr);

        // prefetch next chunk
        temp_ptr += 32;
        __VOLK_PREFETCH(temp_ptr);

        // shuffle once for bit-reversal.
        r_temp0 = _mm256_shuffle_epi8(r_temp0, shuffle_stage4);

        shifted = _mm256_srli_si256(r_temp0, 8); // 128 bit lanes
        shifted = _mm256_and_si256(shifted, mask_stage4);
        r_frame0 = _mm256_xor_si256(shifted, r_temp0);

        shifted = _mm256_srli_si256(r_frame0, 4);
        shifted = _mm256_and_si256(shifted, mask_stage3);
        r_frame0 = _mm256_xor_si256(shifted, r_frame0);

        shifted = _mm256_srli_si256(r_frame0, 2);
        shifted = _mm256_and_si256(shifted, mask_stage2);
        r_frame0 = _mm256_xor_si256(shifted, r_frame0);

        shifted = _mm256_srli_si256(r_frame0, 1);
        shifted = _mm256_and_si256(shifted, mask_stage1);
        r_frame0 = _mm256_xor_si256(shifted, r_frame0);

        // store result of chunk.
        _mm256_store_si256((__m256i*)frame_ptr, r_frame0);
        frame_ptr += 32;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_8u_x2_encodeframepolar_8u_a_avx512bw(unsigned char* frame,
                                                               unsigned char* temp,
                                                               unsigned int frame_size)
{
    if (frame_size < 64) {
        volk_8u_x2_encodeframepolar_8u_generic(frame, temp, frame_size);
        return;
    }

    const unsigned int po2 = log2_of_power_of_2(frame_size);
    unsigned int stage = po2;
    unsigned char* frame_ptr = frame;
    const unsigned char* temp_ptr = temp;
    unsigned int frame_half = frame_size >> 1;
    unsigned int num_branches = 1;
    unsigned int branch;
    unsigned int bit;

    const __m512i mask_s1_512 = _mm512_set1_epi16(0x00FF);
    __m512i r_frame0, r_temp0, shifted;

    {
        __m512i r_frame1, r_temp1;
        const __m128i shuffle_sep_128 =
            _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
        const __m512i shuffle_sep_512 = _mm512_broadcast_i32x4(shuffle_sep_128);
        const __m256i shuffle_sep_256 = _mm256_broadcastsi128_si256(shuffle_sep_128);
        const __m128i mask_s1_128 = _mm_set1_epi16(0x00FF);
        const __m256i mask_s1_256 = _mm256_set1_epi16(0x00FF);
        const __m512i perm_deint =
            _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);

        while (stage > 4) {
            frame_ptr = frame;
            temp_ptr = temp;

            for (branch = 0; branch < num_branches; ++branch) {
                for (bit = 0; bit < frame_half; bit += 64) {
                    unsigned int remaining = frame_half - bit;
                    if (remaining < 64) {
                        if (remaining < 32) {
                            /* 16-byte SSE fallback */
                            __m128i rt0 =
                                _mm_load_si128((const __m128i*)temp_ptr);
                            temp_ptr += 16;
                            __m128i rt1 =
                                _mm_load_si128((const __m128i*)temp_ptr);
                            temp_ptr += 16;
                            __m128i sh = _mm_srli_si128(rt0, 1);
                            sh = _mm_and_si128(sh, mask_s1_128);
                            rt0 = _mm_xor_si128(sh, rt0);
                            rt0 = _mm_shuffle_epi8(rt0, shuffle_sep_128);
                            sh = _mm_srli_si128(rt1, 1);
                            sh = _mm_and_si128(sh, mask_s1_128);
                            rt1 = _mm_xor_si128(sh, rt1);
                            rt1 = _mm_shuffle_epi8(rt1, shuffle_sep_128);
                            _mm_store_si128(
                                (__m128i*)frame_ptr,
                                _mm_unpacklo_epi64(rt0, rt1));
                            _mm_store_si128(
                                (__m128i*)(frame_ptr + frame_half),
                                _mm_unpackhi_epi64(rt0, rt1));
                            frame_ptr += 16;
                        } else {
                            /* 32-byte AVX2 fallback */
                            __m256i rt0 =
                                _mm256_load_si256((const __m256i*)temp_ptr);
                            temp_ptr += 32;
                            __m256i rt1 =
                                _mm256_load_si256((const __m256i*)temp_ptr);
                            temp_ptr += 32;
                            __m256i sh = _mm256_srli_si256(rt0, 1);
                            sh = _mm256_and_si256(sh, mask_s1_256);
                            rt0 = _mm256_xor_si256(sh, rt0);
                            rt0 = _mm256_shuffle_epi8(rt0, shuffle_sep_256);
                            sh = _mm256_srli_si256(rt1, 1);
                            sh = _mm256_and_si256(sh, mask_s1_256);
                            rt1 = _mm256_xor_si256(sh, rt1);
                            rt1 = _mm256_shuffle_epi8(rt1, shuffle_sep_256);
                            __m256i rf0 = _mm256_unpacklo_epi64(rt0, rt1);
                            rt1 = _mm256_unpackhi_epi64(rt0, rt1);
                            rf0 = _mm256_permute4x64_epi64(rf0, 0xd8);
                            __m256i rf1 =
                                _mm256_permute4x64_epi64(rt1, 0xd8);
                            _mm256_store_si256((__m256i*)frame_ptr, rf0);
                            _mm256_store_si256(
                                (__m256i*)(frame_ptr + frame_half), rf1);
                            frame_ptr += 32;
                        }
                        break;
                    }

                    r_temp0 = _mm512_load_si512((const __m512i*)temp_ptr);
                    temp_ptr += 64;
                    r_temp1 = _mm512_load_si512((const __m512i*)temp_ptr);
                    temp_ptr += 64;

                    shifted = _mm512_bsrli_epi128(r_temp0, 1);
                    shifted = _mm512_and_si512(shifted, mask_s1_512);
                    r_temp0 = _mm512_xor_si512(shifted, r_temp0);
                    r_temp0 = _mm512_shuffle_epi8(r_temp0, shuffle_sep_512);

                    shifted = _mm512_bsrli_epi128(r_temp1, 1);
                    shifted = _mm512_and_si512(shifted, mask_s1_512);
                    r_temp1 = _mm512_xor_si512(shifted, r_temp1);
                    r_temp1 = _mm512_shuffle_epi8(r_temp1, shuffle_sep_512);

                    r_frame0 = _mm512_unpacklo_epi64(r_temp0, r_temp1);
                    r_temp1 = _mm512_unpackhi_epi64(r_temp0, r_temp1);
                    r_frame0 = _mm512_permutexvar_epi64(perm_deint, r_frame0);
                    r_frame1 = _mm512_permutexvar_epi64(perm_deint, r_temp1);

                    _mm512_store_si512((__m512i*)frame_ptr, r_frame0);
                    _mm512_store_si512(
                        (__m512i*)(frame_ptr + frame_half), r_frame1);
                    frame_ptr += 64;
                }

                frame_ptr += frame_half;
            }
            memcpy(temp, frame, sizeof(unsigned char) * frame_size);

            num_branches = num_branches << 1;
            frame_half = frame_half >> 1;
            stage--;
        }
    }

    /* Lower 4 stages: process 64 bytes (4 sub-frames of 16) at a time */
    frame_ptr = frame;
    temp_ptr = temp;
    __VOLK_PREFETCH(temp_ptr);

    const __m128i shuffle_s4_128 =
        _mm_setr_epi8(0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15);
    const __m512i shuffle_s4 = _mm512_broadcast_i32x4(shuffle_s4_128);
    const __m512i mask_s4 =
        _mm512_set_epi64(0LL, -1LL, 0LL, -1LL, 0LL, -1LL, 0LL, -1LL);
    const __m512i mask_s3 = _mm512_set1_epi64(0x00000000FFFFFFFFLL);
    const __m512i mask_s2 = _mm512_set1_epi32(0x0000FFFF);

    for (branch = 0; branch < num_branches / 4; ++branch) {
        r_temp0 = _mm512_load_si512((const __m512i*)temp_ptr);

        temp_ptr += 64;
        __VOLK_PREFETCH(temp_ptr);

        r_temp0 = _mm512_shuffle_epi8(r_temp0, shuffle_s4);

        shifted = _mm512_bsrli_epi128(r_temp0, 8);
        shifted = _mm512_and_si512(shifted, mask_s4);
        r_frame0 = _mm512_xor_si512(shifted, r_temp0);

        shifted = _mm512_bsrli_epi128(r_frame0, 4);
        shifted = _mm512_and_si512(shifted, mask_s3);
        r_frame0 = _mm512_xor_si512(shifted, r_frame0);

        shifted = _mm512_bsrli_epi128(r_frame0, 2);
        shifted = _mm512_and_si512(shifted, mask_s2);
        r_frame0 = _mm512_xor_si512(shifted, r_frame0);

        shifted = _mm512_bsrli_epi128(r_frame0, 1);
        shifted = _mm512_and_si512(shifted, mask_s1_512);
        r_frame0 = _mm512_xor_si512(shifted, r_frame0);

        _mm512_store_si512((__m512i*)frame_ptr, r_frame0);
        frame_ptr += 64;
    }
}
#endif /* LV_HAVE_AVX512BW */

#endif /* VOLK_KERNELS_VOLK_VOLK_8U_X2_ENCODEFRAMEPOLAR_8U_A_H_ */
