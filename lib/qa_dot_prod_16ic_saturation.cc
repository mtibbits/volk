/* -*- c++ -*- */
/*
 * Copyright 2026 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*
 * Saturation-contract regression test for volk_16ic_x2_dot_prod_16ic (#220).
 *
 * Saturating addition is not associative in general, so impls that
 * accumulate in different orders legitimately disagree once a partial sum
 * reaches the int16 rail (the correctness sweep stops judging this kernel
 * above its max_sweep_vlen for that reason). It IS associative on same-sign
 * operands: min(x + y, 32767) over non-negative terms is order-independent.
 * So on streams whose per-element product parts all share one sign, every
 * conforming impl -- any lane count, any reduce order -- must return exactly
 * clamp(sum). On these data shapes the legs catch: any wrapping (mod 2^16)
 * accumulate/reduce/tail step, and accumulators that saturate product
 * SUB-terms before the complex product is formed, in both conventions
 * (subtracted term accumulated negated, legs 3/4; or accumulated positively
 * and subtracted at the end, leg 6). A final unsaturated leg (|parts| <= 72,
 * n <= 455 => |sum| <= 32760) pins exact agreement below the rail.
 */

#include <volk/volk.h>
#include <volk/volk_alloc.hh>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

// Coverage floor: in a build whose machine table carries NEON (resp. RVV)
// machines, on a host that runs them, at least one neon* (rvv*) impl must be
// in the swept list -- otherwise a dispatch/machine-selection regression could
// silently drop the sites this test exists to protect while the sweep stays
// green on generic alone. The build expectation comes from CMake
// (VOLK_SAT16IC_EXPECT_*, from available_machines), so a
// VOLK_STATIC_DISPATCH=generic build does not false-red.
bool neon_floor_armed()
{
#if (defined(__aarch64__) || defined(__ARM_NEON)) && defined(VOLK_SAT16IC_EXPECT_NEON)
    return true;
#else
    return false;
#endif
}

bool rvv_floor_armed()
{
#if defined(__riscv_vector) && defined(VOLK_SAT16IC_EXPECT_RVV)
    return true;
#else
    return false;
#endif
}

// 1..40 covers every tail length for widths 4/8/16 (and the saturation point
// lands before, inside, and after the vector region); 64..1000 span several
// RVV vl chunks at every tested VLEN.
const unsigned kVlens[] = { 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,  13,  14,  15,
                            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,  28,  29,  30,
                            31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 64, 100, 257, 1000 };
// Leg 5 (unsaturated) stops at 455: 72 * 455 = 32760 < 32767.
const unsigned kFlatVlens[] = { 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                                29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 455 };
const unsigned kMaxVlen = 1000; // largest vlen in either list
// Independent expected count: 5 saturating legs x 44 vlens + 41 flat cases.
// Deliberately NOT derived from kVlens -- an edit that silently shrinks the
// sweep fails loudly instead of re-deriving itself green.
const unsigned kExpectedCasesPerImpl = 261;

int16_t clamp16(int64_t v)
{
    return v > 32767 ? int16_t(32767) : (v < -32768 ? int16_t(-32768) : int16_t(v));
}

struct lcg {
    uint32_t s;
    int next() // 0..65535
    {
        s = s * 1103515245u + 12345u;
        return int(s >> 16);
    }
};

// Fill a/b for (leg, n); return the exact int64 sums of the per-element
// product parts. Expected values are computed here, never by calling any impl.
void fill_leg(
    int leg, unsigned n, lv_16sc_t* a, lv_16sc_t* b, int64_t* sum_re, int64_t* sum_im)
{
    lcg g{ 12345u + n * 7u + unsigned(leg) };
    int64_t re = 0, im = 0;
    for (unsigned i = 0; i < n; ++i) {
        int ar, ai, br, bi;
        switch (leg) {
        case 1: // (k^2, 0)
        case 2: // (-k^2, 0)
        {
            const int k = 100 + g.next() % 82; // 181^2 = 32761 fits int16
            ar = k;
            ai = 0;
            br = (leg == 1) ? k : -k;
            bi = 0;
            break;
        }
        case 3: // (0, 2k^2)
        case 4: // (0, -2k^2)
        {
            const int k = 64 + g.next() % 64; // 2 * 127^2 = 32258 fits int16
            ar = ai = k;
            br = bi = (leg == 3) ? k : -k;
            break;
        }
        case 6: // (2k-1, 2k(k-1)): unequal sub-terms k^2 vs (k-1)^2
        {
            const int k = 64 + g.next() % 64;
            ar = br = k;
            ai = bi = k - 1;
            break;
        }
        default: // 5: flat +-6 data, never saturates for n <= 455
        {
            ar = g.next() % 13 - 6;
            ai = g.next() % 13 - 6;
            br = g.next() % 13 - 6;
            bi = g.next() % 13 - 6;
            break;
        }
        }
        a[i] = lv_cmake((int16_t)ar, (int16_t)ai);
        b[i] = lv_cmake((int16_t)br, (int16_t)bi);
        re += int64_t(ar) * br - int64_t(ai) * bi;
        im += int64_t(ar) * bi + int64_t(ai) * br;
    }
    *sum_re = re;
    *sum_im = im;
}

} // namespace

// volk_add_test passes the ctest name as argv[1]; this test takes no input.
int main(int, char**)
{
    const volk_func_desc_t desc = volk_16ic_x2_dot_prod_16ic_get_func_desc();
    const char* name = "volk_16ic_x2_dot_prod_16ic";
    // impl_names from the runtime-selected machine are runnable by
    // construction. The assertion is analytic (== clamp of the exact sum), so
    // no reference impl is needed -- but an empty list would pass vacuously.
    if (desc.n_impls == 0) {
        std::printf("%s: FAIL (empty arch list)\n", name);
        return 1;
    }
    bool has_neon = false, has_rvv = false;
    for (size_t i = 0; i < desc.n_impls; ++i) {
        has_neon |= (std::strstr(desc.impl_names[i], "neon") != NULL);
        has_rvv |= (std::strstr(desc.impl_names[i], "rvv") != NULL);
    }
    if (neon_floor_armed() && !has_neon) {
        std::printf("%s: FAIL (NEON build/host but no neon impl in the arch list "
                    "— the sites #220 protects are not being swept)\n",
                    name);
        return 1;
    }
    if (rvv_floor_armed() && !has_rvv) {
        std::printf("%s: FAIL (RVV build/host but no rvv impl in the arch list "
                    "— the sites #220 protects are not being swept)\n",
                    name);
        return 1;
    }

    const int kLegs[] = { 1, 2, 3, 4, 6, 5 };
    int total_failures = 0;
    bool coverage_ok = true;
    volk::vector<lv_16sc_t> a(kMaxVlen);
    volk::vector<lv_16sc_t> b(kMaxVlen);
    volk::vector<lv_16sc_t> out(1); // aligned: the a_* impls store through it
    for (size_t ii = 0; ii < desc.n_impls; ++ii) {
        const char* impl = desc.impl_names[ii];
        int fails = 0;
        unsigned cases = 0;
        int leg_fails[7] = { 0 };
        for (int leg : kLegs) {
            const bool flat = (leg == 5);
            const unsigned* vlens = flat ? kFlatVlens : kVlens;
            const size_t n_vlens = flat ? sizeof(kFlatVlens) / sizeof(kFlatVlens[0])
                                        : sizeof(kVlens) / sizeof(kVlens[0]);
            for (size_t vi = 0; vi < n_vlens; ++vi) {
                const unsigned n = vlens[vi];
                int64_t sr, si;
                fill_leg(leg, n, a.data(), b.data(), &sr, &si);
                out[0] = lv_cmake((int16_t)0x5A5A, (int16_t)0x5A5A); // poison
                volk_16ic_x2_dot_prod_16ic_manual(
                    out.data(), a.data(), b.data(), n, impl);
                ++cases;
                const int16_t wr = clamp16(sr), wi = clamp16(si);
                if (lv_creal(out[0]) != wr || lv_cimag(out[0]) != wi) {
                    ++leg_fails[leg];
                    if (++fails <= 20) {
                        std::printf("%s %s leg%d n=%u: got (%d,%d) want (%d,%d) "
                                    "exact (%" PRId64 ",%" PRId64 ")\n",
                                    name,
                                    impl,
                                    leg,
                                    n,
                                    lv_creal(out[0]),
                                    lv_cimag(out[0]),
                                    wr,
                                    wi,
                                    sr,
                                    si);
                    }
                }
            }
        }
        std::printf("%s %s: %s (%d failing of %u cases) per-leg %d %d %d %d %d %d\n",
                    name,
                    impl,
                    fails ? "FAIL" : "ok",
                    fails,
                    cases,
                    leg_fails[1],
                    leg_fails[2],
                    leg_fails[3],
                    leg_fails[4],
                    leg_fails[5],
                    leg_fails[6]);
        if (cases != kExpectedCasesPerImpl) {
            std::printf("%s %s: FAIL (coverage: ran %u cases, expected %u)\n",
                        name,
                        impl,
                        cases,
                        kExpectedCasesPerImpl);
            coverage_ok = false;
        }
        total_failures += fails;
    }
    return (total_failures == 0 && coverage_ok) ? 0 : 1;
}
