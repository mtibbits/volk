/* -*- c++ -*- */
/*
 * Copyright 2015 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_8u_x3_encodepolar_8u_x2
 *
 * \b Overview
 *
 * Encodes a frame using a polar code. Frozen and information bits are
 * interleaved according to a frozen-bit mask, then the Arikan butterfly
 * transform is applied to produce the encoded frame.
 *
 * Polar codes are capacity-achieving channel codes used in 5G NR control
 * channels and other modern communication systems. The frozen-bit mask
 * identifies which bit positions carry channel-reliability-based frozen
 * values versus user information bits, enabling successive cancellation
 * decoding at the receiver.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_8u_x3_encodepolar_8u_x2(unsigned char* frame, unsigned char* temp,
 * const unsigned char* frozen_bit_mask, const unsigned char* frozen_bits,
 * const unsigned char* info_bits, unsigned int frame_size)
 * \endcode
 *
 * \b Inputs
 * \li frozen_bit_mask: 0xFF for frozen-bit positions, 0x00 for info-bit positions
 * (unsigned char).
 * \li frozen_bits: frozen bit values, one bit per byte (unsigned char).
 * \li info_bits: information bit values, one bit per byte (unsigned char).
 * \li frame_size: number of bits in the frame, must be a power of 2.
 *
 * \b Outputs
 * \li frame: polar-encoded frame, one bit per byte (unsigned char).
 * \li temp: scratch buffer, same size as frame (unsigned char).
 *
 * \b Example
 * Encode a 4-bit frame with 2 frozen bits (all zero) and 2 info bits.
 * \code
 * unsigned int N = 4;
 * unsigned int num_frozen = 2;
 * unsigned int num_info = 2;
 * unsigned int alignment = volk_get_alignment();
 *
 * unsigned char* frame =
 *     (unsigned char*)volk_malloc(sizeof(unsigned char) * N, alignment);
 * unsigned char* temp =
 *     (unsigned char*)volk_malloc(sizeof(unsigned char) * N, alignment);
 * unsigned char* frozen_bit_mask =
 *     (unsigned char*)volk_malloc(sizeof(unsigned char) * N, alignment);
 * unsigned char* frozen_bits =
 *     (unsigned char*)volk_malloc(sizeof(unsigned char) * num_frozen, alignment);
 * unsigned char* info_bits =
 *     (unsigned char*)volk_malloc(sizeof(unsigned char) * num_info, alignment);
 *
 * // Positions 0,1 are frozen; positions 2,3 carry info
 * frozen_bit_mask[0] = 0xFF; frozen_bit_mask[1] = 0xFF;
 * frozen_bit_mask[2] = 0x00; frozen_bit_mask[3] = 0x00;
 * frozen_bits[0] = 0; frozen_bits[1] = 0;
 * info_bits[0] = 1; info_bits[1] = 0;
 *
 * // After interleave: {0, 0, 1, 0}
 * // After polar transform: {1, 1, 0, 0}
 *
 * volk_8u_x3_encodepolar_8u_x2(
 *     frame, temp, frozen_bit_mask, frozen_bits, info_bits, N);
 *
 * printf("Expected: {1, 1, 0, 0}\n");
 * printf("Result:   {%d, %d, %d, %d}\n", frame[0], frame[1], frame[2], frame[3]);
 *
 * volk_free(frame);
 * volk_free(temp);
 * volk_free(frozen_bit_mask);
 * volk_free(frozen_bits);
 * volk_free(info_bits);
 * \endcode
 */

#ifndef VOLK_KERNELS_VOLK_VOLK_8U_X3_ENCODEPOLAR_8U_X2_U_H_
#define VOLK_KERNELS_VOLK_VOLK_8U_X3_ENCODEPOLAR_8U_X2_U_H_
#include <stdio.h>
#include <volk/volk_8u_x2_encodeframepolar_8u.h>

static inline void interleave_frozen_and_info_bits(unsigned char* target,
                                                   const unsigned char* frozen_bit_mask,
                                                   const unsigned char* frozen_bits,
                                                   const unsigned char* info_bits,
                                                   const unsigned int frame_size)
{
    unsigned int bit;
    for (bit = 0; bit < frame_size; ++bit) {
        *target++ = *frozen_bit_mask++ ? *frozen_bits++ : *info_bits++;
    }
}

#ifdef LV_HAVE_GENERIC

static inline void
volk_8u_x3_encodepolar_8u_x2_generic(unsigned char* frame,
                                     unsigned char* temp,
                                     const unsigned char* frozen_bit_mask,
                                     const unsigned char* frozen_bits,
                                     const unsigned char* info_bits,
                                     unsigned int frame_size)
{
    // interleave
    interleave_frozen_and_info_bits(
        temp, frozen_bit_mask, frozen_bits, info_bits, frame_size);
    volk_8u_x2_encodeframepolar_8u_generic(frame, temp, frame_size);
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSSE3
#include <tmmintrin.h>

static inline void
volk_8u_x3_encodepolar_8u_x2_u_ssse3(unsigned char* frame,
                                     unsigned char* temp,
                                     const unsigned char* frozen_bit_mask,
                                     const unsigned char* frozen_bits,
                                     const unsigned char* info_bits,
                                     unsigned int frame_size)
{
    // interleave
    interleave_frozen_and_info_bits(
        temp, frozen_bit_mask, frozen_bits, info_bits, frame_size);
    volk_8u_x2_encodeframepolar_8u_u_ssse3(frame, temp, frame_size);
}

#endif /* LV_HAVE_SSSE3 */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>
static inline void
volk_8u_x3_encodepolar_8u_x2_u_avx2(unsigned char* frame,
                                    unsigned char* temp,
                                    const unsigned char* frozen_bit_mask,
                                    const unsigned char* frozen_bits,
                                    const unsigned char* info_bits,
                                    unsigned int frame_size)
{
    interleave_frozen_and_info_bits(
        temp, frozen_bit_mask, frozen_bits, info_bits, frame_size);
    volk_8u_x2_encodeframepolar_8u_u_avx2(frame, temp, frame_size);
}
#endif /* LV_HAVE_AVX2 */

#endif /* VOLK_KERNELS_VOLK_VOLK_8U_X3_ENCODEPOLAR_8U_X2_U_H_ */

#ifndef VOLK_KERNELS_VOLK_VOLK_8U_X3_ENCODEPOLAR_8U_X2_A_H_
#define VOLK_KERNELS_VOLK_VOLK_8U_X3_ENCODEPOLAR_8U_X2_A_H_

#ifdef LV_HAVE_SSSE3
#include <tmmintrin.h>
static inline void
volk_8u_x3_encodepolar_8u_x2_a_ssse3(unsigned char* frame,
                                     unsigned char* temp,
                                     const unsigned char* frozen_bit_mask,
                                     const unsigned char* frozen_bits,
                                     const unsigned char* info_bits,
                                     unsigned int frame_size)
{
    interleave_frozen_and_info_bits(
        temp, frozen_bit_mask, frozen_bits, info_bits, frame_size);
    volk_8u_x2_encodeframepolar_8u_a_ssse3(frame, temp, frame_size);
}
#endif /* LV_HAVE_SSSE3 */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>
static inline void
volk_8u_x3_encodepolar_8u_x2_a_avx2(unsigned char* frame,
                                    unsigned char* temp,
                                    const unsigned char* frozen_bit_mask,
                                    const unsigned char* frozen_bits,
                                    const unsigned char* info_bits,
                                    unsigned int frame_size)
{
    interleave_frozen_and_info_bits(
        temp, frozen_bit_mask, frozen_bits, info_bits, frame_size);
    volk_8u_x2_encodeframepolar_8u_a_avx2(frame, temp, frame_size);
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_RVV
static inline void volk_8u_x3_encodepolar_8u_x2_rvv(unsigned char* frame,
                                                    unsigned char* temp,
                                                    const unsigned char* frozen_bit_mask,
                                                    const unsigned char* frozen_bits,
                                                    const unsigned char* info_bits,
                                                    unsigned int frame_size)
{
    interleave_frozen_and_info_bits(
        temp, frozen_bit_mask, frozen_bits, info_bits, frame_size);
    volk_8u_x2_encodeframepolar_8u_rvv(frame, temp, frame_size);
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_RVVSEG
static inline void
volk_8u_x3_encodepolar_8u_x2_rvvseg(unsigned char* frame,
                                    unsigned char* temp,
                                    const unsigned char* frozen_bit_mask,
                                    const unsigned char* frozen_bits,
                                    const unsigned char* info_bits,
                                    unsigned int frame_size)
{
    interleave_frozen_and_info_bits(
        temp, frozen_bit_mask, frozen_bits, info_bits, frame_size);
    volk_8u_x2_encodeframepolar_8u_rvvseg(frame, temp, frame_size);
}
#endif /* LV_HAVE_RVVSEG */

#endif /* VOLK_KERNELS_VOLK_VOLK_8U_X3_ENCODEPOLAR_8U_X2_A_H_ */
