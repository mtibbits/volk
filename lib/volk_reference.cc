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
// oracle (see the reduction-oracle contract in volk_reference.h and the
// tolerance methodology in docs/kernel_correctness_harness/README.md; full
// rationale: the kernel's "Numerical accuracy" doc-comment, #118).
static void ref_dot_prod_32f(const std::vector<const void*>& in,
                             const std::vector<void*>& out,
                             lv_32fc_t /*scalar*/,
                             unsigned int vlen)
{
    const float* input = static_cast<const float*>(in[0]);
    const float* taps = static_cast<const float*>(in[1]);
    float* result = static_cast<float*>(out[0]);
    double acc = 0.0;
    for (unsigned int i = 0; i < vlen; i++) {
        acc += static_cast<double>(input[i]) * static_cast<double>(taps[i]);
    }
    result[0] = static_cast<float>(acc);
}

// volk_32fc_x2_dot_prod_32fc: result = sum_i input[i]*taps[i] (complex MAC). A
// complex REDUCTION oracle: reads two complex inputs, accumulates real and
// imaginary parts in double, writes only element 0 of the complex output (the
// harness zero-fills both sides, so the untouched tail compares equal). Same
// rationale as the real dot_prod (#118): impl-vs-generic comparison judges the
// wrong side for reductions — the comparator delta is dominated by whichever side
// accumulates in the worse order, so only a double-precision truth oracle judges
// each impl on its own merit. Complex buffers are interleaved re/im floats.
static void ref_dot_prod_32fc(const std::vector<const void*>& in,
                              const std::vector<void*>& out,
                              lv_32fc_t /*scalar*/,
                              unsigned int vlen)
{
    const lv_32fc_t* input = static_cast<const lv_32fc_t*>(in[0]);
    const lv_32fc_t* taps = static_cast<const lv_32fc_t*>(in[1]);
    lv_32fc_t* result = static_cast<lv_32fc_t*>(out[0]);
    double acc_re = 0.0;
    double acc_im = 0.0;
    for (unsigned int i = 0; i < vlen; i++) {
        const double ar = static_cast<double>(lv_creal(input[i]));
        const double ai = static_cast<double>(lv_cimag(input[i]));
        const double br = static_cast<double>(lv_creal(taps[i]));
        const double bi = static_cast<double>(lv_cimag(taps[i]));
        acc_re += ar * br - ai * bi;
        acc_im += ar * bi + ai * br;
    }
    result[0] = lv_cmake(static_cast<float>(acc_re), static_cast<float>(acc_im));
}

// volk_32fc_x2_conjugate_dot_prod_32fc: result = sum_i input[i]*conj(taps[i]) (the
// FIR inner product). A complex REDUCTION oracle: double-precision accumulation of
// input * conj(taps), writing only element 0 of the complex output (the harness
// zero-fills both sides, so the untouched tail compares equal). conj(taps) negates
// the taps' imaginary part: (ar+ai*i)(br-bi*i) = (ar*br+ai*bi) + (ai*br-ar*bi)*i.
// Same rationale as the plain dot_prod (#118/#119): only a double-precision truth
// oracle judges each impl, not the accumulation-order-dependent generic.
static void ref_conjugate_dot_prod_32fc(const std::vector<const void*>& in,
                                        const std::vector<void*>& out,
                                        lv_32fc_t /*scalar*/,
                                        unsigned int vlen)
{
    const lv_32fc_t* input = static_cast<const lv_32fc_t*>(in[0]);
    const lv_32fc_t* taps = static_cast<const lv_32fc_t*>(in[1]);
    lv_32fc_t* result = static_cast<lv_32fc_t*>(out[0]);
    double acc_re = 0.0;
    double acc_im = 0.0;
    for (unsigned int i = 0; i < vlen; i++) {
        const double ar = static_cast<double>(lv_creal(input[i]));
        const double ai = static_cast<double>(lv_cimag(input[i]));
        const double br = static_cast<double>(lv_creal(taps[i]));
        const double bi = static_cast<double>(lv_cimag(taps[i]));
        acc_re += ar * br + ai * bi;
        acc_im += ai * br - ar * bi;
    }
    result[0] = lv_cmake(static_cast<float>(acc_re), static_cast<float>(acc_im));
}

// volk_32fc_32f_dot_prod_32fc: result = sum_i input[i]*taps[i], complex input with
// a REAL tap vector. A complex REDUCTION oracle: double-precision accumulation of
// the complex input scaled by the real tap (no cross terms), writing only element 0
// of the complex output (the harness zero-fills both sides, so the untouched tail
// compares equal). in[1] is a plain float* (real taps). Same rationale as the plain
// dot_prod (#118/#119): only a double-precision truth oracle judges each impl.
static void ref_32fc_32f_dot_prod_32fc(const std::vector<const void*>& in,
                                       const std::vector<void*>& out,
                                       lv_32fc_t /*scalar*/,
                                       unsigned int vlen)
{
    const lv_32fc_t* input = static_cast<const lv_32fc_t*>(in[0]);
    const float* taps = static_cast<const float*>(in[1]);
    lv_32fc_t* result = static_cast<lv_32fc_t*>(out[0]);
    double acc_re = 0.0;
    double acc_im = 0.0;
    for (unsigned int i = 0; i < vlen; i++) {
        const double t = static_cast<double>(taps[i]);
        acc_re += static_cast<double>(lv_creal(input[i])) * t;
        acc_im += static_cast<double>(lv_cimag(input[i])) * t;
    }
    result[0] = lv_cmake(static_cast<float>(acc_re), static_cast<float>(acc_im));
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
    // dot_prod abs tol = 4e-2 = ceil_1sf(2.5 x max BOTH-SIDES error vs the
    // oracle at the sweep's max vlen 1000003 (generic reaches 1.41e-2 there and
    // runs ref mode too; impls <= 4.6e-3). The 2.5x extreme-value margin for
    // scalar reduction metrics, mode and anchor per the methodology in
    // docs/kernel_correctness_harness/README.md. (#118)
    { "volk_32f_x2_dot_prod_32f", ref_dot_prod_32f, 4e-2f, true },
    // dot_prod_32fc abs tol = 4e-2 = ceil_1sf(2.5 x max BOTH-SIDES error vs the
    // oracle at the sweep's max vlen 1000003: generic reaches 1.44e-2, the a_sse3
    // impl 1.47e-2 (the driver), armhf NEON 1.15e-2; every impl incl generic runs
    // ref mode. Metric is the complex-magnitude error (ccompare abs mode). The
    // 2.5x extreme-value margin for scalar reduction metrics, mode and anchor per
    // the reduction-tolerance methodology in
    // docs/kernel_correctness_harness/README.md. ABSOLUTE: complex dot products of
    // zero-mean data cross zero. (#119)
    { "volk_32fc_x2_dot_prod_32fc", ref_dot_prod_32fc, 4e-2f, true },
    // conjugate_dot_prod_32fc abs tol = 6e-2 = ceil_1sf(2.5 x max BOTH-SIDES error
    // vs the oracle at the sweep's max vlen 1000003: generic reaches 2.16e-2 (the
    // driver; SIMD tiers are more accurate, <= 2.0e-2 block ... 6.4e-3 avx512dq).
    // Metric is the complex-magnitude error (ccompare abs mode). 2.5x extreme-value
    // margin per docs/kernel_correctness_harness/README.md §Reduction-tolerance
    // methodology. ABSOLUTE (zero-mean complex crosses zero). Derived on x86 +
    // armv7 NEON; aarch64/rvv fall under the remeasure clause. (#120)
    { "volk_32fc_x2_conjugate_dot_prod_32fc", ref_conjugate_dot_prod_32fc, 6e-2f, true },
    // 32fc_32f_dot_prod abs tol = 4e-2 = ceil_1sf(2.5 x max BOTH-SIDES error vs the
    // oracle at the sweep's max vlen 1000003: generic reaches 1.42e-2 (the driver;
    // SIMD tiers are more accurate, avx <= 4.5e-3, armhf NEON <= 9.4e-3). Metric is
    // the complex-magnitude error (ccompare abs mode). 2.5x extreme-value margin per
    // docs/kernel_correctness_harness/README.md §Reduction-tolerance methodology.
    // ABSOLUTE (zero-mean complex crosses zero). x86 + armv7 NEON; aarch64/rvv fall
    // under the remeasure clause. (#121)
    { "volk_32fc_32f_dot_prod_32fc", ref_32fc_32f_dot_prod_32fc, 4e-2f, true },
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
