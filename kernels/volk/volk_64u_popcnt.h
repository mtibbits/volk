/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_64u_popcnt
 *
 * \b Overview
 *
 * Computes the population count (popcnt) of a 64-bit unsigned integer,
 * returning the number of bits set to 1. This is equivalent to the
 * Hamming weight of the binary representation.
 *
 * Population count is fundamental in digital communications for computing
 * Hamming distances between codewords, which is central to error detection
 * and correction in channel coding schemes such as block codes and LDPC
 * decoders. It is also used in bit-level operations for spread-spectrum
 * systems and scrambler state analysis.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_64u_popcnt(uint64_t* ret, const uint64_t value)
 * \endcode
 *
 * \b Inputs
 * \li value: The 64-bit input value (uint64_t) whose set bits are counted.
 *
 * \b Outputs
 * \li ret: Pointer to the result (uint64_t) containing the population count.
 *
 * \b Example
 * Count the number of set bits in a known bit pattern.
 * \code
 * unsigned int alignment = volk_get_alignment();
 * uint64_t* ret = (uint64_t*)volk_malloc(sizeof(uint64_t), alignment);
 *
 * // 0xFFFFFFFF has 32 bits set in the lower half
 * uint64_t value = 0xFFFFFFFF;
 * uint64_t expected = 32;
 *
 * volk_64u_popcnt(ret, value);
 *
 * printf("Expected: %lu\n", expected);
 * printf("Result:   %lu\n", *ret);
 *
 * volk_free(ret);
 * \endcode
 */

#ifndef INCLUDED_volk_64u_popcnt_a_H
#define INCLUDED_volk_64u_popcnt_a_H

#include <inttypes.h>
#include <stdio.h>


#ifdef LV_HAVE_GENERIC


static inline void volk_64u_popcnt_generic(uint64_t* ret, const uint64_t value)
{
    // const uint32_t* valueVector = (const uint32_t*)&value;

    // This is faster than a lookup table
    // uint32_t retVal = valueVector[0];
    uint32_t retVal = (uint32_t)(value & 0x00000000FFFFFFFFull);

    retVal = (retVal & 0x55555555) + (retVal >> 1 & 0x55555555);
    retVal = (retVal & 0x33333333) + (retVal >> 2 & 0x33333333);
    retVal = (retVal + (retVal >> 4)) & 0x0F0F0F0F;
    retVal = (retVal + (retVal >> 8));
    retVal = (retVal + (retVal >> 16)) & 0x0000003F;
    uint64_t retVal64 = retVal;

    // retVal = valueVector[1];
    retVal = (uint32_t)((value & 0xFFFFFFFF00000000ull) >> 32);
    retVal = (retVal & 0x55555555) + (retVal >> 1 & 0x55555555);
    retVal = (retVal & 0x33333333) + (retVal >> 2 & 0x33333333);
    retVal = (retVal + (retVal >> 4)) & 0x0F0F0F0F;
    retVal = (retVal + (retVal >> 8));
    retVal = (retVal + (retVal >> 16)) & 0x0000003F;
    retVal64 += retVal;

    *ret = retVal64;
}

#endif /*LV_HAVE_GENERIC*/


#if LV_HAVE_SSE4_2 && LV_HAVE_64

#include <nmmintrin.h>

static inline void volk_64u_popcnt_a_sse4_2(uint64_t* ret, const uint64_t value)
{
    *ret = _mm_popcnt_u64(value);
}

#endif /*LV_HAVE_SSE4_2*/


#if LV_HAVE_NEON
#include <arm_neon.h>
static inline void volk_64u_popcnt_neon(uint64_t* ret, const uint64_t value)
{
    uint8x8_t input_val, count8x8_val;
    uint16x4_t count16x4_val;
    uint32x2_t count32x2_val;
    uint64x1_t count64x1_val;

    input_val = vld1_u8((unsigned char*)&value);
    count8x8_val = vcnt_u8(input_val);
    count16x4_val = vpaddl_u8(count8x8_val);
    count32x2_val = vpaddl_u16(count16x4_val);
    count64x1_val = vpaddl_u32(count32x2_val);
    vst1_u64(ret, count64x1_val);

    //*ret = _mm_popcnt_u64(value);
}
#endif /*LV_HAVE_NEON*/

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_64u_popcnt_neonv8(uint64_t* ret, const uint64_t value)
{
    /* Same as neon, but using cleaner intrinsics available in ARMv8 */
    uint8x8_t input_val = vreinterpret_u8_u64(vcreate_u64(value));
    uint8x8_t count8x8_val = vcnt_u8(input_val);
    *ret = vaddlv_u8(count8x8_val);
}
#endif /*LV_HAVE_NEONV8*/

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_64u_popcnt_rvv(uint64_t* ret, const uint64_t value)
{
    *ret = __riscv_vcpop(__riscv_vreinterpret_b2(__riscv_vmv_s_x_u64m1(value, 1)), 64);
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVA22V
#include <riscv_bitmanip.h>

static inline void volk_64u_popcnt_rva22(uint64_t* ret, const uint64_t value)
{
    *ret = __riscv_cpop_64(value);
}
#endif /*LV_HAVE_RVA22V*/

#endif /*INCLUDED_volk_64u_popcnt_a_H*/
