/* -*- c++ -*- */
/*
 * Copyright 2026 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*
 * Tie-position-sweeping regression test for the complex index kernels (#195).
 *
 * The first-index-wins tie-break is only observable when the FIRST tied
 * element sits in a lane the implementation's horizontal reduce scans LATE.
 * Fixed-position tie edges (test_params_index_fc, the correctness sweep's
 * complex edge vector) place the first tied index in an early-scanned lane,
 * so a scan-order reduce passes them by accident. This test plants an
 * equal-magnitude extremum pair at every position pair (i, j), i < j, for
 * several vlens, and requires every available implementation to return i.
 */

#include <volk/volk.h>
#include <volk/volk_alloc.hh>

#include <cstdio>
#include <cstring>

namespace {

// Coverage floor: on avx2-capable hardware the avx2 impls must actually be in
// the swept list — otherwise a dispatch/machine-table regression could
// silently drop the 16 reduce sites this test exists to protect while the
// sweep stays green on the remaining impls. Keyed to the HARDWARE capability
// (the runtime machine name is not usable: an avx512 box selects e.g.
// avx512f_64_mmx_orc, which does not contain "avx2" yet carries the avx2
// impls). Armed only where the probe exists (GCC/Clang x86); elsewhere the
// per-arch summary lines keep coverage visible.
bool hw_has_avx2()
{
#if (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__))
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

// 8: the issue's repro width (one avx2 block). 32: two avx512 blocks, four
// avx2 blocks, no tail. 37: forces a scalar tail on every SIMD width, so
// vector-region-vs-tail ties are swept too.
const unsigned kVlens[] = { 8, 32, 37 };
// Independent expected count: C(8,2) + C(32,2) + C(37,2). Deliberately NOT
// derived from kVlens — the coverage check below compares the swept total
// against this constant, so an edit that silently shrinks the sweep (or a
// pair-enumeration bug) fails loudly instead of re-deriving itself green.
const unsigned kExpectedPairsPerArch = 1190;

template <typename TargetT, typename ManualF>
int sweep_kernel(const char* kernel_name,
                 volk_func_desc_t desc,
                 ManualF manual,
                 lv_32fc_t baseline,
                 lv_32fc_t extremum)
{
    // impl_names from the runtime-selected machine are runnable by
    // construction. The assertion below is analytic (== first tied index), so
    // no reference impl is required — but an empty list would pass vacuously,
    // and under VOLK_STATIC_DISPATCH the list legitimately shrinks to the
    // pinned machine's impls (no "generic" entry), so emptiness is the only
    // fail-closed condition here.
    if (desc.n_impls == 0) {
        std::printf("%s: FAIL (empty arch list)\n", kernel_name);
        return 1;
    }
    bool list_has_avx2 = false;
    for (size_t i = 0; i < desc.n_impls && !list_has_avx2; ++i)
        list_has_avx2 = (std::strstr(desc.impl_names[i], "avx2") != NULL);
    if (hw_has_avx2() && !list_has_avx2) {
        std::printf("%s: FAIL (avx2-capable host but no avx2 impl in the arch "
                    "list — the sites #195 protects are not being swept)\n",
                    kernel_name);
        return 1;
    }

    int total_failures = 0;
    unsigned total_pairs = 0;
    for (size_t ai = 0; ai < desc.n_impls; ++ai) {
        const char* arch = desc.impl_names[ai];
        unsigned arch_pairs = 0;
        int arch_failures = 0;
        for (unsigned vlen : kVlens) {
            volk::vector<lv_32fc_t> buf(vlen);
            for (unsigned i = 0; i + 1 < vlen; ++i) {
                for (unsigned j = i + 1; j < vlen; ++j) {
                    for (unsigned k = 0; k < vlen; ++k)
                        buf[k] = baseline;
                    buf[i] = extremum;
                    buf[j] = extremum;
                    TargetT target = static_cast<TargetT>(vlen); // impossible
                    manual(&target, buf.data(), vlen, arch);
                    ++arch_pairs;
                    if (static_cast<unsigned>(target) != i) {
                        ++arch_failures;
                        // Cap printed rows PER ARCH so one broken impl
                        // cannot suppress its siblings' evidence; count all.
                        if (arch_failures <= 20)
                            std::printf("%s %s vlen=%u tie(%u,%u): got %u want %u\n",
                                        kernel_name,
                                        arch,
                                        vlen,
                                        i,
                                        j,
                                        static_cast<unsigned>(target),
                                        i);
                    }
                }
            }
        }
        // One summary line per (kernel, arch): the per-impl failure count
        // the born-red evidence keys on.
        std::printf("%s %s: %s (%d failing of %u tie pairs)\n",
                    kernel_name,
                    arch,
                    arch_failures ? "FAIL" : "ok",
                    arch_failures,
                    arch_pairs);
        total_failures += arch_failures;
        total_pairs += arch_pairs;
    }
    if (total_pairs != desc.n_impls * kExpectedPairsPerArch) {
        std::printf("%s: FAIL (coverage: swept %u pairs, expected %zu)\n",
                    kernel_name,
                    total_pairs,
                    desc.n_impls * kExpectedPairsPerArch);
        return total_failures + 1;
    }
    return total_failures;
}

} // namespace

int main(int, char**)
{
    int failures = 0;

    // Magnitudes chosen exactly representable: baseline |z|^2 = 2 or 9,
    // extremum |z|^2 = 9 or 1 -- no float-compare tolerance in play.
    failures += sweep_kernel<uint16_t>("volk_32fc_index_max_16u",
                                       volk_32fc_index_max_16u_get_func_desc(),
                                       volk_32fc_index_max_16u_manual,
                                       lv_cmake(1.0f, 1.0f),
                                       lv_cmake(3.0f, 0.0f));
    failures += sweep_kernel<uint32_t>("volk_32fc_index_max_32u",
                                       volk_32fc_index_max_32u_get_func_desc(),
                                       volk_32fc_index_max_32u_manual,
                                       lv_cmake(1.0f, 1.0f),
                                       lv_cmake(3.0f, 0.0f));
    failures += sweep_kernel<uint16_t>("volk_32fc_index_min_16u",
                                       volk_32fc_index_min_16u_get_func_desc(),
                                       volk_32fc_index_min_16u_manual,
                                       lv_cmake(3.0f, 0.0f),
                                       lv_cmake(1.0f, 0.0f));
    failures += sweep_kernel<uint32_t>("volk_32fc_index_min_32u",
                                       volk_32fc_index_min_32u_get_func_desc(),
                                       volk_32fc_index_min_32u_manual,
                                       lv_cmake(3.0f, 0.0f),
                                       lv_cmake(1.0f, 0.0f));

    std::printf("qa_index_tie_sweep: %d failing checks\n", failures);
    return failures ? 1 : 0;
}
