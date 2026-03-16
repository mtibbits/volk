/* -*- c++ -*- */
/*
 * Copyright 2023 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#ifndef INCLUDED_volk_32f_s32f_clamppuppet_32f_H
#define INCLUDED_volk_32f_s32f_clamppuppet_32f_H

#include <volk/volk_32f_s32f_x2_clamp_32f.h>

#ifdef LV_HAVE_GENERIC
static inline void volk_32f_s32f_clamppuppet_32f_generic(float* out,
                                                         const float* in,
                                                         const float min,
                                                         unsigned int num_points)
{
    volk_32f_s32f_x2_clamp_32f_generic(out, in, min, -min, num_points);
}
#endif

#ifdef LV_HAVE_SSE
static inline void volk_32f_s32f_clamppuppet_32f_a_sse(float* out,
                                                       const float* in,
                                                       const float min,
                                                       unsigned int num_points)
{
    volk_32f_s32f_x2_clamp_32f_a_sse(out, in, min, -min, num_points);
}
#endif

#ifdef LV_HAVE_SSE4_1
static inline void volk_32f_s32f_clamppuppet_32f_a_sse4_1(float* out,
                                                          const float* in,
                                                          const float min,
                                                          unsigned int num_points)
{
    volk_32f_s32f_x2_clamp_32f_a_sse4_1(out, in, min, -min, num_points);
}
#endif

#ifdef LV_HAVE_AVX
static inline void volk_32f_s32f_clamppuppet_32f_a_avx(float* out,
                                                       const float* in,
                                                       const float min,
                                                       unsigned int num_points)
{
    volk_32f_s32f_x2_clamp_32f_a_avx(out, in, min, -min, num_points);
}
#endif

#ifdef LV_HAVE_AVX512F
static inline void volk_32f_s32f_clamppuppet_32f_a_avx512f(float* out,
                                                            const float* in,
                                                            const float min,
                                                            unsigned int num_points)
{
    volk_32f_s32f_x2_clamp_32f_a_avx512f(out, in, min, -min, num_points);
}
#endif

#ifdef LV_HAVE_SSE
static inline void volk_32f_s32f_clamppuppet_32f_u_sse(float* out,
                                                       const float* in,
                                                       const float min,
                                                       unsigned int num_points)
{
    volk_32f_s32f_x2_clamp_32f_u_sse(out, in, min, -min, num_points);
}
#endif

#ifdef LV_HAVE_SSE4_1
static inline void volk_32f_s32f_clamppuppet_32f_u_sse4_1(float* out,
                                                          const float* in,
                                                          const float min,
                                                          unsigned int num_points)
{
    volk_32f_s32f_x2_clamp_32f_u_sse4_1(out, in, min, -min, num_points);
}
#endif

#ifdef LV_HAVE_AVX
static inline void volk_32f_s32f_clamppuppet_32f_u_avx(float* out,
                                                       const float* in,
                                                       const float min,
                                                       unsigned int num_points)
{
    volk_32f_s32f_x2_clamp_32f_u_avx(out, in, min, -min, num_points);
}
#endif

#ifdef LV_HAVE_AVX512F
static inline void volk_32f_s32f_clamppuppet_32f_u_avx512f(float* out,
                                                            const float* in,
                                                            const float min,
                                                            unsigned int num_points)
{
    volk_32f_s32f_x2_clamp_32f_u_avx512f(out, in, min, -min, num_points);
}
#endif

#ifdef LV_HAVE_NEON
static inline void volk_32f_s32f_clamppuppet_32f_neon(float* out,
                                                      const float* in,
                                                      const float min,
                                                      unsigned int num_points)
{
    volk_32f_s32f_x2_clamp_32f_neon(out, in, min, -min, num_points);
}
#endif

#ifdef LV_HAVE_NEONV8
static inline void volk_32f_s32f_clamppuppet_32f_neonv8(float* out,
                                                        const float* in,
                                                        const float min,
                                                        unsigned int num_points)
{
    volk_32f_s32f_x2_clamp_32f_neonv8(out, in, min, -min, num_points);
}
#endif

#ifdef LV_HAVE_RVV
static inline void volk_32f_s32f_clamppuppet_32f_rvv(float* out,
                                                     const float* in,
                                                     const float min,
                                                     unsigned int num_points)
{
    volk_32f_s32f_x2_clamp_32f_rvv(out, in, min, -min, num_points);
}
#endif

#endif /* INCLUDED_volk_32f_s32f_clamppuppet_32f_H */
