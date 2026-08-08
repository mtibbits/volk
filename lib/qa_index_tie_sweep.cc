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

// 8: the issue's repro width (one avx2 block). 32: two avx512 blocks, four
// avx2 blocks, no tail. 37: forces a scalar tail on every SIMD width, so
// vector-region-vs-tail ties are swept too.
const unsigned kVlens[] = { 8, 32, 37 };

unsigned pairs_per_arch()
{
    unsigned pairs = 0;
    for (unsigned vlen : kVlens)
        pairs += vlen * (vlen - 1) / 2;
    return pairs;
}

template <typename TargetT, typename ManualF>
int sweep_kernel(const char* kernel_name,
                 volk_func_desc_t desc,
                 ManualF manual,
                 lv_32fc_t baseline,
                 lv_32fc_t extremum)
{
    // impl_names from the runtime-selected machine are runnable by
    // construction.
    bool have_generic = false;
    for (size_t i = 0; i < desc.n_impls && !have_generic; ++i)
        have_generic = (std::strcmp(desc.impl_names[i], "generic") == 0);
    if (!have_generic) {
        std::printf("%s: FAIL (generic missing from arch list, %zu archs)\n",
                    kernel_name,
                    desc.n_impls);
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
    if (total_pairs != desc.n_impls * pairs_per_arch()) {
        std::printf("%s: FAIL (coverage: swept %u pairs, expected %zu)\n",
                    kernel_name,
                    total_pairs,
                    desc.n_impls * pairs_per_arch());
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
