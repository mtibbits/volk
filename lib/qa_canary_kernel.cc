/* -*- c++ -*- */
/*
 * Copyright 2026 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "qa_canary_kernel.h"

#include "volk/volk_complex.h" // lv_32fc_t/lv_cmake (planted power pair, #92)
#include <cmath>               // atan2f/powf/cosf/sinf/tanhf (planted defect pairs, #92)
#include <csignal>             // for raise (misaligned-fault portable fallback, #91)
#include <cstdint>             // for uint32_t (input-immutability scribble, #90)
#include <cstring>             // strcmp (tanh pair impl dispatch, #92)
#if defined(__SSE2__)
#include <emmintrin.h> // SSE2 umbrella; provides the SSE1 _mm_load_ps/_mm_storeu_ps
                       // used by the #91 planted misaligned-fault control
#endif

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

void volk_32f_misalignedfault_32f(void* out,
                                  void* in,
                                  unsigned int num_points,
                                  const char*)
{
    float* o = static_cast<float*>(out);
    const float* i = static_cast<const float*>(in);
    // The planted defect (#91): an ALIGNED load inside an impl whose dispatch
    // metadata says unaligned -- the movaps-in-a-_u_-kernel class. Under __SSE2__
    // (any x86-64 build) this is a genuine alignment-checked 16-byte load that
    // raises SIGSEGV iff `in` is not 16-byte aligned; elsewhere we simulate the
    // same hardware fault with raise(SIGSEGV). On ALIGNED input it must run
    // correctly (proves the detector doesn't over-report on the aligned path).
#if defined(__SSE2__)
    if (num_points >= 4) {
        // NOTE: relies on the compiler emitting an alignment-checked load
        // ((v)movaps) for _mm_load_ps, which GCC/clang do at every -O level for
        // this TU. Under LTO/cross-TU inlining a compiler that can PROVE the
        // pointer misaligned may legally lower it to movups and the fault
        // vanishes (negative control catches that as NC LOST, exit 2 -- fail
        // loud). Do not build the harness with LTO.
        __m128 v = _mm_load_ps(i); // movaps: faults on misalignment
        _mm_storeu_ps(o, v);
        for (unsigned int n = 4; n < num_points; ++n) {
            o[n] = i[n];
        }
        return;
    }
#else
    if (reinterpret_cast<uintptr_t>(in) % 16 != 0 && num_points >= 4) {
        raise(SIGSEGV);
    }
#endif
    for (unsigned int n = 0; n < num_points; ++n) {
        o[n] = i[n];
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

// ---- #92 combined negative control planted kernels ----

// The defective scalar power core, verbatim from pre-e29bc7f
// volk_32fc_s32f_power_32fc.h: atan2f(re, im) is SWAPPED (must be (im, re)) and
// the cosine is negated. Kept in this test-only TU permanently so the combined
// negative control survives the live kernel's eventual fix.
static inline lv_32fc_t power_core_defect(const lv_32fc_t exp, const float power)
{
    const float arg = power * atan2f(lv_creal(exp), lv_cimag(exp));
    const float mag =
        powf(lv_creal(exp) * lv_creal(exp) + lv_cimag(exp) * lv_cimag(exp), power / 2);
    return mag * lv_cmake(-cosf(arg), sinf(arg));
}

// The corrected formula (the e29bc7f fix): this is the "reverted
// reintroduction" twin -- reference mode must pass it.
static inline lv_32fc_t power_core_ok(const lv_32fc_t exp, const float power)
{
    const float arg = power * atan2f(lv_cimag(exp), lv_creal(exp));
    const float mag =
        powf(lv_creal(exp) * lv_creal(exp) + lv_cimag(exp) * lv_cimag(exp), power / 2);
    return mag * lv_cmake(cosf(arg), sinf(arg));
}

void volk_32fc_s32f_powerdefect_32fc(
    void* out, void* in, float scalar, unsigned int num_points, const char*)
{
    // single-impl desc; arch is always "generic"
    lv_32fc_t* c = static_cast<lv_32fc_t*>(out);
    const lv_32fc_t* a = static_cast<const lv_32fc_t*>(in);
    for (unsigned int n = 0; n < num_points; ++n) {
        c[n] = power_core_defect(a[n], scalar);
    }
}

void volk_32fc_s32f_powerok_32fc(
    void* out, void* in, float scalar, unsigned int num_points, const char*)
{
    lv_32fc_t* c = static_cast<lv_32fc_t*>(out);
    const lv_32fc_t* a = static_cast<const lv_32fc_t*>(in);
    for (unsigned int n = 0; n < num_points; ++n) {
        c[n] = power_core_ok(a[n], scalar);
    }
}

// The tanh series impl, with and without the pre-e29bc7f stale-pointer defect:
// when |x| crosses the 4.97 clamp, only the output pointer advances in the
// defective version, so every subsequent element reads a stale input value.
static void
tanh_series_impl(float* cPtr, const float* aPtr, unsigned int num_points, bool stale)
{
    for (unsigned int number = 0; number < num_points; number++) {
        if (*aPtr > 4.97f) {
            *cPtr++ = 1;
            if (!stale) {
                aPtr++;
            }
        } else if (*aPtr <= -4.97f) {
            *cPtr++ = -1;
            if (!stale) {
                aPtr++;
            }
        } else {
            float x2 = (*aPtr) * (*aPtr);
            float a = (*aPtr) * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
            float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
            *cPtr++ = a / b;
            aPtr++;
        }
    }
}

static void tanh_nc_dispatch(
    void* out, void* in, unsigned int num_points, const char* arch, bool stale)
{
    float* c = static_cast<float*>(out);
    const float* a = static_cast<const float*>(in);
    if (arch && strcmp(arch, "u_series") == 0) {
        tanh_series_impl(c, a, num_points, stale);
    } else { // "generic": the independent correct baseline
        for (unsigned int n = 0; n < num_points; ++n) {
            c[n] = tanhf(a[n]);
        }
    }
}

void volk_32f_tanhstale_32f(void* out,
                            void* in,
                            unsigned int num_points,
                            const char* arch)
{
    tanh_nc_dispatch(out, in, num_points, arch, true);
}

void volk_32f_tanhok_32f(void* out, void* in, unsigned int num_points, const char* arch)
{
    tanh_nc_dispatch(out, in, num_points, arch, false);
}

volk_func_desc_t volk_power_nc_desc()
{
    // Same synthetic single-impl shape as the #89-#91 planted kernels; the
    // distinct name documents intent at the #92 call sites.
    return volk_canary_desc();
}

volk_func_desc_t volk_tanh_nc_desc()
{
    static const char* impl_names[] = { "generic", "u_series" };
    static const int impl_deps[] = { 0, 0 };
    static const bool impl_alignment[] = { false, false };

    volk_func_desc_t desc;
    desc.impl_names = impl_names;
    desc.impl_deps = impl_deps;
    desc.impl_alignment = impl_alignment;
    desc.n_impls = 2;
    return desc;
}
