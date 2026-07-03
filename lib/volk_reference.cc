/* -*- c++ -*- */
/*
 * Copyright 2026 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "volk_reference.h"

#include <cmath>

// Each oracle reads the kernel's input buffer(s) from `in`, computes the
// kernel's mathematically-defined output in DOUBLE precision, and narrows to the
// kernel's native float/complex output type only at the final store into `out`.
// The double-precision intermediate is what exposes defects all impls share.

// volk_32fc_s32f_power_32fc: c = a^power  (polar: |a|^power * e^{i*power*arg(a)},
// arg(a) = atan2(imag, real)). The shipping kernel swaps the atan2 args and
// negates the real part — this oracle implements the CORRECT definition, so the
// generic impl diverges and reference mode fails it (the #88 negative control).
static void ref_power_32fc(const std::vector<const void*>& in,
                           const std::vector<void*>& out,
                           lv_32fc_t scalar,
                           unsigned int vlen)
{
    const lv_32fc_t* a = static_cast<const lv_32fc_t*>(in[0]);
    lv_32fc_t* c = static_cast<lv_32fc_t*>(out[0]);
    const double power = static_cast<double>(lv_creal(scalar));
    for (unsigned int i = 0; i < vlen; i++) {
        const double re = static_cast<double>(lv_creal(a[i]));
        const double im = static_cast<double>(lv_cimag(a[i]));
        const double mag = std::pow(std::hypot(re, im), power);
        const double ang = power * std::atan2(im, re);
        c[i] = lv_cmake(static_cast<float>(mag * std::cos(ang)),
                        static_cast<float>(mag * std::sin(ang)));
    }
}

// volk_32f_tanh_32f: c = tanh(a). The generic is correct; the SIMD tail handler
// has a stale-read bug, so reference mode fails the SIMD impls (conditionally, at
// non-aligned vlens with a large-magnitude clamp element).
static void ref_tanh_32f(const std::vector<const void*>& in,
                         const std::vector<void*>& out,
                         lv_32fc_t /*scalar*/,
                         unsigned int vlen)
{
    const float* a = static_cast<const float*>(in[0]);
    float* c = static_cast<float*>(out[0]);
    for (unsigned int i = 0; i < vlen; i++) {
        c[i] = static_cast<float>(std::tanh(static_cast<double>(a[i])));
    }
}

// volk_32fc_magnitude_32f: m = sqrt(re^2 + im^2).
static void ref_magnitude_32f(const std::vector<const void*>& in,
                              const std::vector<void*>& out,
                              lv_32fc_t /*scalar*/,
                              unsigned int vlen)
{
    const lv_32fc_t* a = static_cast<const lv_32fc_t*>(in[0]);
    float* m = static_cast<float*>(out[0]);
    for (unsigned int i = 0; i < vlen; i++) {
        const double re = static_cast<double>(lv_creal(a[i]));
        const double im = static_cast<double>(lv_cimag(a[i]));
        m[i] = static_cast<float>(std::sqrt(re * re + im * im));
    }
}

// volk_32fc_s32f_atan2_32f: o = atan2(imag, real) / normalizeFactor.
static void ref_atan2_32f(const std::vector<const void*>& in,
                          const std::vector<void*>& out,
                          lv_32fc_t scalar,
                          unsigned int vlen)
{
    const lv_32fc_t* a = static_cast<const lv_32fc_t*>(in[0]);
    float* o = static_cast<float*>(out[0]);
    const double inv = 1.0 / static_cast<double>(lv_creal(scalar));
    for (unsigned int i = 0; i < vlen; i++) {
        const double re = static_cast<double>(lv_creal(a[i]));
        const double im = static_cast<double>(lv_cimag(a[i]));
        o[i] = static_cast<float>(std::atan2(im, re) * inv);
    }
}

// volk_32f_log2_32f: b = log2(a) with the kernel's NON-IEEE contract
// (0 -> -127, +Inf -> +127, negative|NaN -> NaN), mirroring volk_log2f_non_ieee.
static void ref_log2_32f(const std::vector<const void*>& in,
                         const std::vector<void*>& out,
                         lv_32fc_t /*scalar*/,
                         unsigned int vlen)
{
    const float* a = static_cast<const float*>(in[0]);
    float* b = static_cast<float*>(out[0]);
    for (unsigned int i = 0; i < vlen; i++) {
        const double r = std::log2(static_cast<double>(a[i]));
        if (std::isnan(r)) {
            b[i] = static_cast<float>(r); // negative input or NaN -> NaN
        } else if (std::isinf(r)) {
            b[i] =
                std::copysign(127.0f, static_cast<float>(r)); // 0 -> -127, +Inf -> +127
        } else {
            b[i] = static_cast<float>(r);
        }
    }
}

// volk_32f_x2_dot_prod_32f: result = sum_i input[i]*taps[i]. The first REDUCTION
// oracle: reads two inputs, accumulates in double, writes only element 0 of the
// output (the harness zero-fills both sides, so the untouched tail compares
// equal). Rationale for existence (#118): impl-vs-generic comparison judges the
// WRONG side for reductions — generic's serial float accumulation error grows
// ~linearly in vlen and is 4-6x LARGER than the SIMD partial-sum error, so the
// delta is dominated by the reference. Only a truth oracle fixes the instrument.
static void ref_dot_prod_32f(const std::vector<const void*>& in,
                             const std::vector<void*>& out,
                             lv_32fc_t scalar,
                             unsigned int vlen)
{
    (void)scalar;
    const float* input = static_cast<const float*>(in[0]);
    const float* taps = static_cast<const float*>(in[1]);
    float* result = static_cast<float*>(out[0]);
    double acc = 0.0;
    for (unsigned int i = 0; i < vlen; i++) {
        acc += static_cast<double>(input[i]) * static_cast<double>(taps[i]);
    }
    result[0] = static_cast<float>(acc);
}

static const std::vector<volk_reference_entry> g_registry = {
    // name                          oracle             tol      absolute
    { "volk_32fc_s32f_power_32fc", ref_power_32fc, 1e-4f, false },
    { "volk_32f_tanh_32f", ref_tanh_32f, 1e-2f, false },
    { "volk_32fc_magnitude_32f", ref_magnitude_32f, 1e-1f, false },
    { "volk_32fc_s32f_atan2_32f", ref_atan2_32f, 1e-5f, false },
    // log2 abs tol = 2e-5: the SIMD log2 POLYNOMIAL diverges from true double
    // log2 by up to ~5e-6 (measured), so default qa's 5e-6 (SIMD-vs-generic) is
    // too tight when comparing SIMD-vs-true-double. 2e-5 covers the approximation
    // envelope with margin while still catching gross defects (off by >>1e-4).
    { "volk_32f_log2_32f", ref_log2_32f, 2e-5f, true },
    // dot_prod abs tol = 3e-2 = ceil_1sf(1.5 x max BOTH-SIDES error vs the
    // double oracle at the sweep's max vlen 1000003: generic's serial error
    // reaches 1.41e-2 there (impls 2.4-4.5e-3); generic runs ref mode too, so
    // the bound must cover it. ABSOLUTE because dot products of zero-mean data
    // cross zero (relative-vs-oracle ill-posed at small |result|). Absolute
    // error grows ~linearly in vlen; this bound is anchored at 1000003. (#118)
    { "volk_32f_x2_dot_prod_32f", ref_dot_prod_32f, 3e-2f, true },
};

const std::vector<volk_reference_entry>& volk_reference_registry() { return g_registry; }

const volk_reference_entry* volk_reference_lookup(const std::string& name)
{
    for (const auto& e : g_registry) {
        if (name == e.name) {
            return &e;
        }
    }
    return nullptr;
}
