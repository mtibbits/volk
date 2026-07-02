/* -*- c++ -*- */
/*
 * Copyright 2026 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*
 * Kernel correctness harness foundation (mtibbits/volk#87, child of epic #85):
 * tail-remainder sweep + adversarial edges + per-kernel triage report.
 *
 * The standard qa runs each kernel at a single (prime) vlen, which exercises exactly
 * one `num_points mod width` remainder per SIMD width. Tail defects that only trigger
 * at a different remainder slip through. This driver re-runs the *existing*
 * run_volk_tests comparison across a set of vlens that cover every remainder for the
 * SIMD widths VOLK uses, with adversarial edge values injected into the test data, and
 * reports per kernel which vlens failed — the triage list for the per-kernel fix tickets.
 * It reuses run_volk_tests unchanged; default qa is untouched.
 *
 * Child #88 adds an INDEPENDENT double-precision reference: for kernels in the
 * volk_reference registry, every impl (generic included) is compared against a
 * double-precision oracle instead of impl-vs-generic, catching defects ALL impls
 * share (e.g. the swapped atan2 in volk_32fc_s32f_power_32fc). Unregistered kernels
 * fall back to this child's impl-vs-impl comparison; each kernel's mode (ref|impl)
 * is reported. Output canary + ASan (#89), input-immutability (#90), and misaligned
 * `_u` runs (#91) are the remaining epic-#85 children.
 */

#include "kernel_tests.h" // for init_test_list
#include "qa_utils.h"     // for volk_test_case_t, run_volk_tests
#include "volk/volk_complex.h"
#include "volk_reference.h" // for the independent double-precision oracle registry (#88)
#include <volk/volk.h>

#include <cstdio>  // fflush
#include <cstdlib> // getenv
#include <iostream>
#include <string>
#include <vector>

// The stdout-muting helper uses the POSIX fd calls dup/dup2/open/close. MSVC has
// no <unistd.h> and spells these with a leading underscore in <io.h>; it also
// uses "NUL" rather than "/dev/null". Map the names so the call sites stay
// POSIX-shaped on every platform. The object-like macros are defined AFTER all
// standard includes -- macros named open/close/dup/dup2 would otherwise rewrite
// declarations inside subsequently included headers.
#ifdef _MSC_VER
#include <fcntl.h> // _O_WRONLY
#include <io.h>    // _dup, _dup2, _open, _close
#define dup _dup
#define dup2 _dup2
#define open _open
#define close _close
#ifndef STDOUT_FILENO
#define STDOUT_FILENO 1
#endif
#ifndef O_WRONLY
#define O_WRONLY _O_WRONLY
#endif
#define VOLK_DEVNULL "NUL"
#else
#include <fcntl.h>  // open, O_WRONLY
#include <unistd.h> // dup, dup2, close, STDOUT_FILENO
#define VOLK_DEVNULL "/dev/null"
#endif

// Adversarial edge values mixed into the random test data, so impls are exercised at
// the boundaries where edge-case branches (clamps, saturation, zero/sign handling) live
// — not just the smooth interior the default random data covers. Kept moderate so they
// stress branches without provoking universal float overflow.
static const std::vector<float> kFloatEdges = { 0.0f,  -0.0f, 1.0f,  -1.0f, 4.97f, 5.0f,
                                                -5.0f, 6.0f,  -6.0f, 8.0f,  -8.0f };
static const std::vector<lv_32fc_t> kComplexEdges = {
    lv_32fc_t(0, 0),  lv_32fc_t(5, 5),  lv_32fc_t(-5, -5),
    lv_32fc_t(6, -6), lv_32fc_t(-8, 8), lv_32fc_t(1, 0)
};

// Run a single (kernel, vlen) with stdout (fmt::print + std::cout) muted at the fd
// level, so the sweep emits only our per-kernel report. If `ref` is non-null the
// kernel is checked against the independent double-precision oracle (#88, every impl
// vs truth); otherwise it falls back to the impl-vs-impl comparison. Returns true on
// FAILURE.
static bool quiet_run(volk_test_case_t& tc,
                      const volk_reference_entry* ref,
                      float tol,
                      lv_32fc_t scalar,
                      int iter,
                      unsigned int v)
{
    std::vector<volk_test_results_t> results;
    const bool verbose = (std::getenv("HARNESS_VERBOSE") != nullptr);
    std::fflush(stdout);
    std::cout.flush();
    int saved = -1, devnull = -1;
    if (!verbose) {
        saved = dup(STDOUT_FILENO);
        devnull = open(VOLK_DEVNULL, O_WRONLY);
        // Only redirect if BOTH fds are valid; otherwise we could mute stdout
        // with no way to restore it (a failed dup leaves saved == -1).
        if (saved >= 0 && devnull >= 0)
            dup2(devnull, STDOUT_FILENO);
    }
    bool fail = false;
    try {
        if (ref) {
            fail = run_volk_reference_test(tc.desc(),
                                           tc.kernel_ptr(),
                                           tc.name(),
                                           *ref,
                                           scalar,
                                           v /*vlen*/,
                                           &results,
                                           kFloatEdges,
                                           kComplexEdges);
        } else {
            fail = run_volk_tests(tc.desc(),
                                  tc.kernel_ptr(),
                                  tc.name(),
                                  tol,
                                  scalar,
                                  v /*vlen*/,
                                  iter,
                                  &results,
                                  tc.puppet_master_name(),
                                  false /*absolute_mode*/,
                                  false /*benchmark_mode*/,
                                  kFloatEdges,
                                  kComplexEdges);
        }
    } catch (...) {
        fail = true; // an exception at this vlen is itself a defect to report
    }
    std::fflush(stdout);
    std::cout.flush();
    if (!verbose) {
        // Restore only if we actually have the saved fd; never dup2/close(-1).
        if (saved >= 0) {
            dup2(saved, STDOUT_FILENO);
            close(saved);
        }
        if (devnull >= 0)
            close(devnull);
    }
    return fail;
}

int main(int argc, char* argv[])
{
    const float tol = 1e-6f;
    const lv_32fc_t scalar = 327.0;
    const int iter = 1;

    volk_test_params_t base(tol, scalar, 131071, iter, false, "");
    std::vector<volk_test_case_t> test_cases = init_test_list(base);

    // Small sizes 1..40 cover every `n mod width` for widths {2,4,8,16,32} with several
    // k; the two large primes preserve coverage at scale (and the historical 131071).
    std::vector<unsigned int> vlens;
    for (unsigned int v = 1; v <= 40; ++v)
        vlens.push_back(v);
    vlens.push_back(131071);
    vlens.push_back(1000003);

    const std::string filter = (argc > 1) ? std::string(argv[1]) : std::string();

    // Optional machine-readable triage report (one CSV row per kernel). The failed_vlens
    // column is space-separated so it cannot clash with the CSV comma delimiter.
    FILE* report = nullptr;
    if (const char* rp = std::getenv("HARNESS_REPORT")) {
        report = std::fopen(rp, "w");
        if (report)
            std::fprintf(report, "kernel,mode,result,failed_vlens\n");
        else
            std::cerr << "HARNESS_REPORT: cannot open '" << rp
                      << "' — human output only\n";
    }

    std::cout << "# kernel-correctness remainder sweep: vlens 1..40, 131071, 1000003\n";
    std::cout.flush();

    int tested = 0, failed = 0;
    // Negative-control tracking: power_seen = power_32fc was in the (filtered) sweep
    // at all; power_ref_tested = it ran in reference mode. The two together detect a
    // dropped/renamed registration (the likelier regression), not just a broken oracle.
    bool power_seen = false, power_ref_tested = false;
    for (auto& tc : test_cases) {
        if (!filter.empty() && tc.name() != filter)
            continue;
        ++tested;
        // Registered kernels run against the independent double reference (#88);
        // the rest fall back to the impl-vs-impl comparison.
        const volk_reference_entry* ref = volk_reference_lookup(tc.name());
        const char* mode = ref ? "ref" : "impl";
        // Reference mode needs the kernel's OWN scalar (e.g. power exponent 2.5, not
        // the driver default 327 — which would overflow |x|^327 to inf and mask the
        // defect). Impl mode keeps the driver default to preserve #87's sweep.
        const lv_32fc_t kscalar = ref ? tc.test_parameters().scalar() : scalar;
        std::vector<unsigned int> bad;
        for (unsigned int v : vlens) {
            if (quiet_run(tc, ref, tol, kscalar, iter, v))
                bad.push_back(v);
        }
        if (tc.name() == "volk_32fc_s32f_power_32fc") {
            power_seen = true;
            power_ref_tested = (ref != nullptr);
        }
        if (bad.empty()) {
            std::cout << "ok    [" << mode << "] " << tc.name() << "\n";
        } else {
            ++failed;
            std::cout << "FAIL  [" << mode << "] " << tc.name() << "  vlens:";
            for (unsigned int v : bad)
                std::cout << " " << v;
            std::cout << "\n";
        }
        if (report) {
            std::string vs;
            for (unsigned int v : bad) {
                vs += std::to_string(v);
                vs += ' ';
            }
            std::fprintf(report,
                         "%s,%s,%s,%s\n",
                         tc.name().c_str(),
                         mode,
                         bad.empty() ? "ok" : "FAIL",
                         vs.c_str());
        }
        std::cout.flush();
    }
    if (report)
        std::fclose(report);
    std::cerr << "\ncorrectness remainder sweep: " << failed << " / " << tested
              << " kernels failed\n";

    // #88 coverage guard: volk_32fc_s32f_power_32fc is the harness's motivating
    // escaped defect (swapped atan2f; it has no SIMD impl, so the impl-vs-impl
    // comparison can never see it — the double oracle is its only detector).
    // Deliberately state-independent: whether the live kernel passes or fails
    // reference mode is NOT asserted here — a kernel regression already fails
    // the sweep above (exit 1), and the detector itself is proven continuously
    // by the planted-twin combined negative control (HARNESS_COMBINED_NC).
    // The one failure nothing else catches is the reference REGISTRATION
    // silently disappearing (dropped/renamed) — only that is guarded.
    if (power_seen && !power_ref_tested) {
        std::cerr << "NEGATIVE CONTROL LOST: volk_32fc_s32f_power_32fc ran but is NOT "
                     "reference-registered (dropped/renamed registration) — a power "
                     "defect of the escaped swapped-atan2 class would go undetected.\n";
        return 2;
    }

    return failed > 0 ? 1 : 0;
}
