/* -*- c++ -*- */
/*
 * Copyright 2026 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "qa_canary_kernel.h"

#include <cstdint> // for uint32_t (input-immutability scribble, #90)

// Each planted impl first copies the in-bounds region [0,num_points) correctly,
// so the only thing distinguishing the negative controls is the write past the
// end -- the canary's guard/two-sentinel check is what flags it. `arch` is
// ignored: every planted kernel has a single "generic" impl.

void volk_32f_canaryok_32f(void* out, void* in, unsigned int num_points, const char*)
{
    float* o = static_cast<float*>(out);
    const float* i = static_cast<const float*>(in);
    for (unsigned int n = 0; n < num_points; ++n) {
        o[n] = i[n];
    }
}

void volk_32f_canaryonepast_32f(void* out, void* in, unsigned int num_points, const char*)
{
    float* o = static_cast<float*>(out);
    const float* i = static_cast<const float*>(in);
    for (unsigned int n = 0; n < num_points; ++n) {
        o[n] = i[n];
    }
    // The planted defect: one element past the end. In the canary's guarded
    // buffer this lands in the trailing sentinel guard (in-allocation) and is
    // caught by the post-run guard check.
    o[num_points] = 1.0f;
}

void volk_32f_canaryunwritten_32f(void* out,
                                  void* in,
                                  unsigned int num_points,
                                  const char*)
{
    float* o = static_cast<float*>(out);
    const float* i = static_cast<const float*>(in);
    // The planted defect: leave the LAST in-bounds element unwritten. `n + 1 <
    // num_points` keeps this safe at num_points == 0 (writes nothing). The gap
    // is invisible to value comparison but caught by the two-sentinel check.
    for (unsigned int n = 0; n + 1 < num_points; ++n) {
        o[n] = i[n];
    }
}

void volk_32f_inputscribble_32f(void* out, void* in, unsigned int num_points, const char*)
{
    float* o = static_cast<float*>(out);
    const float* i = static_cast<const float*>(in);
    for (unsigned int n = 0; n < num_points; ++n) {
        o[n] = i[n]; // copy in -> out FIRST, so out[0] keeps the original value
    }
    // The planted defect (#90): scribble on the INPUT buffer. A correct
    // out-of-place kernel never writes its input; this one does, so the
    // input-immutability check (post-run byte compare vs the pristine pre-image)
    // must flag it. Write a bitwise complement of element 0 rather than a fixed
    // constant: ~x != x for every uint32_t, so the input byte is GUARANTEED to
    // change regardless of the (possibly edge-case-seeded) pristine value -- a
    // fixed constant could coincide with the seed and let the negative control
    // pass silently. Guarded so it is safe at num_points == 0.
    if (num_points > 0) {
        uint32_t* w = static_cast<uint32_t*>(in);
        w[0] = ~w[0];
    }
}

void volk_32f_canaryfarpast_32f(void* out, void* in, unsigned int num_points, const char*)
{
    float* o = static_cast<float*>(out);
    const float* i = static_cast<const float*>(in);
    for (unsigned int n = 0; n < num_points; ++n) {
        o[n] = i[n];
    }
    // Contiguous over-write stepping past the whole guarded allocation into the
    // ASan redzone. Writing every element (rather than a single far store)
    // guarantees the first store that crosses the allocation boundary trips
    // ASan regardless of the exact redzone size. UNDEFINED BEHAVIOUR without
    // ASan -- only invoked under the HARNESS_CANARY_ASAN_DEMO path.
    for (unsigned int k = 0; k <= VOLK_CANARY_FAR_PAST_ELEMENTS; ++k) {
        o[num_points + k] = 1.0f;
    }
}

volk_func_desc_t volk_canary_desc()
{
    static const char* impl_names[] = { "generic" };
    static const int impl_deps[] = { 0 };
    static const bool impl_alignment[] = { false };

    volk_func_desc_t desc;
    desc.impl_names = impl_names;
    desc.impl_deps = impl_deps;
    desc.impl_alignment = impl_alignment;
    desc.n_impls = 1;
    return desc;
}
