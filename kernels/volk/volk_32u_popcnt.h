/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32u_popcnt
 *
 * \b Overview
 *
 * Computes the population count (popcnt) of an unsigned 32-bit integer,
 * returning the number of bits set to 1. This is equivalent to computing
 * the Hamming weight of the binary representation.
 *
 * Population count is fundamental in digital communications for computing
 * Hamming distances between codewords in error-correcting codes (e.g. LDPC,
 * turbo codes), syndrome decoding, and bit-error-rate (BER) measurement.
 * It is also used in correlation-based synchronization when comparing
 * binary sequences.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32u_popcnt(uint32_t* ret, const uint32_t value)
 * \endcode
 *
 * \b Inputs
 * \li value: The unsigned 32-bit input value (uint32_t).
 *
 * \b Outputs
 * \li ret: Pointer to the result containing the population count (uint32_t).
 *
 * \b Example
 * Count the number of set bits in an alternating bit pattern.
 * \code
 * uint32_t bitstring = 0x55555555;
 * uint32_t result = 0;
 *
 * // 0x55555555 = 0101...0101 in binary, so 16 of 32 bits are set
 * uint32_t expected = 16;
 *
 * volk_32u_popcnt(&result, bitstring);
 *
 * printf("Expected: %u\n", expected);
 * printf("Result:   %u\n", result);
 * \endcode
 */

#ifndef INCLUDED_VOLK_32u_POPCNT_A16_H
#define INCLUDED_VOLK_32u_POPCNT_A16_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_32u_popcnt_generic(uint32_t* ret, const uint32_t value)
{
    // This is faster than a lookup table
    uint32_t retVal = value;

    retVal = (retVal & 0x55555555) + (retVal >> 1 & 0x55555555);
    retVal = (retVal & 0x33333333) + (retVal >> 2 & 0x33333333);
    retVal = (retVal + (retVal >> 4)) & 0x0F0F0F0F;
    retVal = (retVal + (retVal >> 8));
    retVal = (retVal + (retVal >> 16)) & 0x0000003F;

    *ret = retVal;
}

#endif /*LV_HAVE_GENERIC*/


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32u_popcnt_neon(uint32_t* ret, const uint32_t value)
{
    // Load value into a 64-bit vector (as 8 bytes)
    uint8x8_t input = vreinterpret_u8_u32(vdup_n_u32(value));
    // Count bits in each byte
    uint8x8_t counts = vcnt_u8(input);
    // Sum across all bytes (only first 4 matter for 32-bit value)
    // Use vpaddl to widen and add: 8x8 -> 4x16 -> 2x32 -> 1x64
    uint16x4_t sum16 = vpaddl_u8(counts);
    uint32x2_t sum32 = vpaddl_u16(sum16);
    // Extract the lower 32-bit element which contains the sum of the lower 4 bytes
    *ret = vget_lane_u32(sum32, 0);
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_SSE4_2

#include <nmmintrin.h>

static inline void volk_32u_popcnt_a_sse4_2(uint32_t* ret, const uint32_t value)
{
    *ret = _mm_popcnt_u32(value);
}

#endif /*LV_HAVE_SSE4_2*/

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32u_popcnt_rvv(uint32_t* ret, const uint32_t value)
{
    *ret = __riscv_vcpop(__riscv_vreinterpret_b4(__riscv_vmv_s_x_u64m1(value, 1)), 32);
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVA22V
#include <riscv_bitmanip.h>

static inline void volk_32u_popcnt_rva22(uint32_t* ret, const uint32_t value)
{
    *ret = __riscv_cpop_32(value);
}
#endif /*LV_HAVE_RVA22V*/

#endif /*INCLUDED_VOLK_32u_POPCNT_A16_H*/
