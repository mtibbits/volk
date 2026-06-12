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

#include "kernel_tests.h"     // for init_test_list
#include "qa_canary_kernel.h" // for the planted output-canary negative controls (#89)
#include "qa_utils.h"         // for volk_test_case_t, run_volk_tests
#include "volk/volk_complex.h"
#include "volk_reference.h" // for the independent double-precision oracle registry (#88)
#include <volk/volk.h>

#include <cstdio>  // fflush
#include <cstdlib> // getenv
#include <iostream>
#include <map> // per-impl failed-vlen accumulation (#92 triage report)
#include <set> // per-impl seen-set (#92 triage report)
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

// Compile-time AddressSanitizer detection: the far-past ASan demo deliberately
// writes past the guarded allocation, which is undefined behaviour unless ASan
// is bracketing it, so it must only run in an ASan build.
#if defined(__SANITIZE_ADDRESS__)
#define VOLK_BUILT_WITH_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define VOLK_BUILT_WITH_ASAN 1
#endif
#endif
#ifndef VOLK_BUILT_WITH_ASAN
#define VOLK_BUILT_WITH_ASAN 0
#endif

// Adversarial edge values mixed into the random test data, so impls are exercised at
// the boundaries where edge-case branches (clamps, saturation, zero/sign handling) live
// — not just the smooth interior the default random data covers. Kept moderate so they
// stress branches without provoking universal float overflow.
// NOTE (#92): the combined negative control derives its edge list from kFloatEdges
// (minus 4.97f). An added edge in roughly [4.5, 4.97] — past where the planted tanh
// Padé series degrades beyond 1e-2 but not into the > 4.97 clamp — would flip the
// tanh-ok revert-proof red with a misattributed "sweep over-reports" diagnosis.
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
// FAILURE. The edge vectors default to the sweep's globals; the combined negative
// control (#92) overrides them to prove its edges-off blind-spot assertions.
// impls_seen / impl_fails (optional) collect per-impl triage detail (#92): every
// impl the run exercised, and the vlens at which each impl failed. An exception
// produces a kernel-level failure with no impl attribution -- the caller reports
// such vlens on a synthetic "-" impl row.
static bool
quiet_run(volk_test_case_t& tc,
          const volk_reference_entry* ref,
          float tol,
          lv_32fc_t scalar,
          bool absolute_mode,
          int iter,
          unsigned int v,
          const std::vector<float>& fedges = kFloatEdges,
          const std::vector<lv_32fc_t>& cedges = kComplexEdges,
          std::set<std::string>* impls_seen = nullptr,
          std::map<std::string, std::vector<unsigned int>>* impl_fails = nullptr)
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
                                           fedges,
                                           cedges);
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
                                  absolute_mode,
                                  false /*benchmark_mode*/,
                                  fedges,
                                  cedges);
        }
    } catch (...) {
        fail = true; // an exception at this vlen is itself a defect to report
    }
    // #92 triage detail: both run_volk_tests and run_volk_reference_test fill
    // results.back().results with one entry per impl (generic included), each
    // carrying the impl's pass flag for this (kernel, vlen) run.
    if ((impls_seen || impl_fails) && !results.empty()) {
        for (const auto& kv : results.back().results) {
            if (impls_seen)
                impls_seen->insert(kv.first);
            if (impl_fails && !kv.second.pass)
                (*impl_fails)[kv.first].push_back(v);
        }
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

// #89: run the output-buffer canary for one (kernel, vlen) with stdout muted
// (mirrors quiet_run). Returns the split summary so the driver can treat a buffer
// over/under-write (always a defect) differently from an in-bounds unwritten
// element (expected for reduction/index kernels). On an exception, the guarded
// path could not complete -- reported as a guard violation to surface it.
static volk_canary_summary quiet_canary_run(volk_test_case_t& tc, unsigned int v)
{
    std::vector<volk_test_results_t> results;
    const bool verbose = (std::getenv("HARNESS_VERBOSE") != nullptr);
    std::fflush(stdout);
    std::cout.flush();
    int saved = -1, devnull = -1;
    if (!verbose) {
        saved = dup(STDOUT_FILENO);
        devnull = open(VOLK_DEVNULL, O_WRONLY);
        if (saved >= 0 && devnull >= 0)
            dup2(devnull, STDOUT_FILENO);
    }
    volk_canary_summary summary;
    try {
        summary = run_volk_canary_test(tc.desc(),
                                       tc.kernel_ptr(),
                                       tc.name(),
                                       tc.test_parameters().scalar(),
                                       v /*vlen*/,
                                       &results,
                                       kFloatEdges,
                                       kComplexEdges);
    } catch (...) {
        summary.guard_violation = true;
    }
    std::fflush(stdout);
    std::cout.flush();
    if (!verbose) {
        if (saved >= 0) {
            dup2(saved, STDOUT_FILENO);
            close(saved);
        }
        if (devnull >= 0)
            close(devnull);
    }
    return summary;
}

// #90: run the input-immutability check for one (kernel, vlen) with stdout muted
// (mirrors quiet_canary_run). On an exception the run did not complete, so we cannot
// conclude anything about input immutability -- report it skipped (applied stays
// false) with a note, rather than mislabeling a crash as a mutation.
static volk_immutability_summary quiet_immutability_run(volk_test_case_t& tc,
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
        if (saved >= 0 && devnull >= 0)
            dup2(devnull, STDOUT_FILENO);
    }
    volk_immutability_summary summary;
    bool threw = false;
    try {
        summary = run_volk_immutability_test(tc.desc(),
                                             tc.kernel_ptr(),
                                             tc.name(),
                                             tc.test_parameters().scalar(),
                                             v /*vlen*/,
                                             &results,
                                             kFloatEdges,
                                             kComplexEdges);
    } catch (...) {
        threw = true;
        summary = volk_immutability_summary(); // applied=false, mutated=false -> skip
    }
    std::fflush(stdout);
    std::cout.flush();
    if (!verbose) {
        if (saved >= 0) {
            dup2(saved, STDOUT_FILENO);
            close(saved);
        }
        if (devnull >= 0)
            close(devnull);
    }
    if (threw)
        std::cerr << "note  [immutable] " << tc.name() << " threw at vlen " << v
                  << " (skipped; crashes are out of #90 scope)\n";
    return summary;
}

// #91: run the misaligned check for one (kernel, vlen) with stdout muted (mirrors
// quiet_immutability_run). Hardware faults are handled INSIDE
// run_volk_misaligned_test by the signal path (-> summary.crashed); a C++
// exception here is non-signal harness plumbing, out of #91 scope -- report it
// skipped (applied stays false) with a note rather than mislabeling it.
static volk_misaligned_summary quiet_misaligned_run(volk_test_case_t& tc, unsigned int v)
{
    std::vector<volk_test_results_t> results;
    const bool verbose = (std::getenv("HARNESS_VERBOSE") != nullptr);
    std::fflush(stdout);
    std::cout.flush();
    int saved = -1, devnull = -1;
    if (!verbose) {
        saved = dup(STDOUT_FILENO);
        devnull = open(VOLK_DEVNULL, O_WRONLY);
        if (saved >= 0 && devnull >= 0)
            dup2(devnull, STDOUT_FILENO);
    }
    volk_misaligned_summary summary;
    bool threw = false;
    try {
        summary = run_volk_misaligned_test(tc.desc(),
                                           tc.kernel_ptr(),
                                           tc.name(),
                                           tc.test_parameters().scalar(),
                                           tc.test_parameters().tol(),
                                           tc.test_parameters().absolute_mode(),
                                           v /*vlen*/,
                                           &results,
                                           kFloatEdges,
                                           kComplexEdges);
    } catch (...) {
        threw = true;
        summary = volk_misaligned_summary(); // applied=false -> skip
    }
    std::fflush(stdout);
    std::cout.flush();
    if (!verbose) {
        if (saved >= 0) {
            dup2(saved, STDOUT_FILENO);
            close(saved);
        }
        if (devnull >= 0)
            close(devnull);
    }
    if (threw)
        std::cerr << "note  [misaligned] " << tc.name() << " threw at vlen " << v
                  << " (skipped; non-signal exceptions are out of #91 scope)\n";
    return summary;
}

// #92: format a vlen list as the space-separated failed_vlens CSV field.
static std::string vlens_str(const std::vector<unsigned int>& vl)
{
    std::string s;
    for (unsigned int v : vl) {
        s += std::to_string(v);
        s += ' ';
    }
    return s;
}

// #92: vlens in `all` not attributed to any impl in `by_impl` (exception-class
// failures) -- reported on a synthetic "-" impl row so they stay visible.
static std::vector<unsigned int>
unattributed_vlens(const std::vector<unsigned int>& all,
                   const std::map<std::string, std::vector<unsigned int>>& by_impl)
{
    std::set<unsigned int> attributed;
    for (const auto& kv : by_impl)
        attributed.insert(kv.second.begin(), kv.second.end());
    std::vector<unsigned int> out;
    for (unsigned int v : all) {
        if (attributed.find(v) == attributed.end())
            out.push_back(v);
    }
    return out;
}

// #92: one offender category for emit_impl_rows -- the impl->failed-vlens map
// for one defect class, and the result string its rows carry.
struct impl_row_category {
    const std::map<std::string, std::vector<unsigned int>>* fails;
    const char* status;
};

// #92: the shared per-impl CSV row emitter (one row per exercised impl).
// Categories are in priority order: an impl's row takes the status of the
// FIRST category that contains it, and concatenates the vlens of every
// same-status category that also contains it (e.g. misaligned crash+diverge
// both "FAIL"); impls in no category emit an `ok` row. Keeping this in ONE
// place keeps the four mode blocks' rows from drifting apart.
static void emit_impl_rows(FILE* report,
                           const std::string& kernel,
                           const char* mode,
                           const std::set<std::string>& impls_seen,
                           const std::vector<impl_row_category>& categories)
{
    for (const std::string& impl : impls_seen) {
        const char* status = nullptr;
        std::string vs;
        for (const impl_row_category& cat : categories) {
            const auto it = cat.fails->find(impl);
            if (it == cat.fails->end())
                continue;
            if (!status)
                status = cat.status;
            if (std::string(status) == cat.status)
                vs += vlens_str(it->second);
        }
        std::fprintf(report,
                     "%s,%s,%s,%s,%s\n",
                     kernel.c_str(),
                     impl.c_str(),
                     mode,
                     status ? status : "ok",
                     vs.c_str());
    }
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

    // Optional machine-readable triage report (#87, per-impl rows since #92):
    // one CSV row per kernel x impl -- `kernel,impl,mode,result,failed_vlens`.
    // Rows with impl `-` are kernel-level: skips, and failures that could not
    // be attributed to a specific impl (e.g. an exception during a run). The
    // failed_vlens column is space-separated so it cannot clash with the CSV
    // comma delimiter.
    FILE* report = nullptr;
    if (const char* rp = std::getenv("HARNESS_REPORT")) {
        report = std::fopen(rp, "w");
        if (report)
            std::fprintf(report, "kernel,impl,mode,result,failed_vlens\n");
        else
            std::cerr << "HARNESS_REPORT: cannot open '" << rp
                      << "' — human output only\n";
    }

    // #89 output-buffer canary mode (opt-in via HARNESS_CANARY). This is a
    // distinct mode: it replaces the impl-vs-impl / reference sweep with the
    // guarded-buffer over/under-write + unwritten check. The default qa suite
    // and the toggle-off correctness sweep are entirely unaffected.
    const bool canary_mode = (std::getenv("HARNESS_CANARY") != nullptr);
    if (canary_mode) {
        // The far-past ASan demonstration: deliberately over-run the guarded
        // own-malloc'd buffer far enough to hit the ASan redzone, proving the
        // guarded path's buffers are ASan-bracketed (acceptance #89-2). This is
        // undefined behaviour without ASan, so it is gated to ASan builds.
        if (std::getenv("HARNESS_CANARY_ASAN_DEMO") != nullptr) {
            if (report)
                std::fclose(report);
#if VOLK_BUILT_WITH_ASAN
            std::cerr << "HARNESS_CANARY_ASAN_DEMO: over-running the guarded buffer "
                         "via volk_32f_canaryfarpast_32f; expect an ASan "
                         "heap-buffer-overflow abort.\n";
            std::vector<volk_test_results_t> demo;
            run_volk_canary_test(volk_canary_desc(),
                                 (void (*)())(&volk_32f_canaryfarpast_32f),
                                 "volk_32f_canaryfarpast_32f",
                                 scalar,
                                 127u,
                                 &demo,
                                 kFloatEdges,
                                 kComplexEdges);
            std::cerr << "WARNING: ASan did not abort on the far-past over-run.\n";
            return 2;
#else
            std::cerr << "HARNESS_CANARY_ASAN_DEMO requested but this binary was NOT "
                         "built with AddressSanitizer; refusing to run the far-past "
                         "planted kernel (undefined behaviour without ASan). Rebuild "
                         "under build-asan/.\n";
            return 2;
#endif
        }

        // Hard negative control (any build), exercising the SAME run_volk_canary_test
        // path real kernels use. Three planted kernels prove BOTH detectors:
        //   ok        -> no violation             (canary does not over-report)
        //   one-past  -> GUARD violation          (the over-run detector)
        //   unwritten -> UNWRITTEN finding         (the in-bounds-gap detector)
        // Any deviation means the canary is broken — fail hard (exit 2), per #88.
        const unsigned int nc_vlen = 127;
        std::vector<volk_test_results_t> nc;
        const volk_canary_summary ok_sum =
            run_volk_canary_test(volk_canary_desc(),
                                 (void (*)())(&volk_32f_canaryok_32f),
                                 "volk_32f_canaryok_32f",
                                 scalar,
                                 nc_vlen,
                                 &nc,
                                 kFloatEdges,
                                 kComplexEdges);
        nc.clear();
        const volk_canary_summary one_past_sum =
            run_volk_canary_test(volk_canary_desc(),
                                 (void (*)())(&volk_32f_canaryonepast_32f),
                                 "volk_32f_canaryonepast_32f",
                                 scalar,
                                 nc_vlen,
                                 &nc,
                                 kFloatEdges,
                                 kComplexEdges);
        nc.clear();
        const volk_canary_summary unwritten_sum =
            run_volk_canary_test(volk_canary_desc(),
                                 (void (*)())(&volk_32f_canaryunwritten_32f),
                                 "volk_32f_canaryunwritten_32f",
                                 scalar,
                                 nc_vlen,
                                 &nc,
                                 kFloatEdges,
                                 kComplexEdges);
        nc.clear();
        const char* nc_err = nullptr;
        if (ok_sum.guard_violation || ok_sum.unwritten) {
            nc_err = "the correct planted kernel volk_32f_canaryok_32f was flagged — "
                     "the canary over-reports";
        } else if (!one_past_sum.guard_violation) {
            nc_err = "the planted one-past kernel volk_32f_canaryonepast_32f did not "
                     "trip the guard — the canary is blind to output over-runs";
        } else if (!unwritten_sum.unwritten) {
            nc_err = "the planted kernel volk_32f_canaryunwritten_32f did not trip the "
                     "unwritten check — the canary is blind to in-bounds gaps";
        }
        if (nc_err) {
            std::cerr << "NEGATIVE CONTROL LOST: " << nc_err << ". Aborting.\n";
            if (report)
                std::fclose(report);
            return 2;
        }
        std::cerr << "canary negative control OK: ok-kernel clean, one-past trips the "
                     "guard, unwritten-kernel trips the unwritten check.\n";

        // Best-effort per-kernel canary sweep over the real kernels.
        //   FAIL    -> a GUARD violation (write past the end / before index 0): a
        //              real buffer over/under-run, the issue's core blind spot.
        //   partial -> an in-bounds element left unwritten with NO guard violation:
        //              expected for reduction/index/accumulator kernels (fixed-size
        //              scalar output, not num_points elements), which the signature
        //              cannot distinguish from a real map under-write — surfaced for
        //              triage, not counted as a failure.
        // Puppets are skipped: their output-buffer semantics differ from a plain map.
        std::cout << "# output-canary sweep: vlens 1..40, 131071, 1000003\n";
        std::cout.flush();
        int c_tested = 0, c_failed = 0, c_partial = 0;
        for (auto& tc : test_cases) {
            if (!filter.empty() && tc.name() != filter)
                continue;
            if (tc.puppet_master_name() != "NULL") {
                std::cout << "skip  [canary] " << tc.name() << " (puppet)\n";
                if (report)
                    std::fprintf(report, "%s,-,canary,skip,\n", tc.name().c_str());
                std::cout.flush();
                continue;
            }
            std::vector<unsigned int> over_run;
            std::vector<unsigned int> partial;
            bool any_applied = false;
            std::set<std::string> impls_seen;
            std::map<std::string, std::vector<unsigned int>> guard_fails;
            std::map<std::string, std::vector<unsigned int>> unwritten_fails;
            for (unsigned int v : vlens) {
                const volk_canary_summary s = quiet_canary_run(tc, v);
                if (s.applied)
                    any_applied = true;
                if (s.guard_violation)
                    over_run.push_back(v);
                else if (s.unwritten)
                    partial.push_back(v);
                for (const std::string& impl : s.checked_impls)
                    impls_seen.insert(impl);
                for (const std::string& impl : s.guard_impls)
                    guard_fails[impl].push_back(v);
                for (const std::string& impl : s.unwritten_impls)
                    unwritten_fails[impl].push_back(v);
            }
            // A kernel the canary could not guard (no output buffer / unsupported
            // signature) is SKIPPED, not reported ok — an unguarded kernel must not
            // masquerade as covered. But a guard violation (incl. an exception, which
            // quiet_canary_run reports as guard_violation) is a real defect and must
            // FAIL even when no impl was successfully guarded: skip only when nothing
            // was observed at all (fail closed).
            if (!any_applied && over_run.empty() && partial.empty()) {
                std::cout << "skip  [canary] " << tc.name()
                          << " (no guardable output buffer)\n";
                if (report)
                    std::fprintf(report, "%s,-,canary,skip,\n", tc.name().c_str());
                std::cout.flush();
                continue;
            }
            ++c_tested;
            const std::vector<unsigned int>* vl;
            if (!over_run.empty()) {
                ++c_failed;
                vl = &over_run;
                std::cout << "FAIL  [canary] " << tc.name() << "  over-run vlens:";
            } else if (!partial.empty()) {
                ++c_partial;
                vl = &partial;
                std::cout << "part  [canary] " << tc.name()
                          << "  unwritten (reduction?) vlens:";
            } else {
                vl = nullptr;
                std::cout << "ok    [canary] " << tc.name();
            }
            if (vl) {
                for (unsigned int v : *vl)
                    std::cout << " " << v;
            }
            std::cout << "\n";
            if (report) {
                // Guard violation > unwritten (#92).
                emit_impl_rows(
                    report,
                    tc.name(),
                    "canary",
                    impls_seen,
                    { { &guard_fails, "FAIL" }, { &unwritten_fails, "partial" } });
                // Guard-violation vlens with no impl attribution (an exception
                // inside the guarded run) stay visible on a "-" row.
                const std::vector<unsigned int> unattr =
                    unattributed_vlens(over_run, guard_fails);
                if (!unattr.empty()) {
                    std::fprintf(report,
                                 "%s,-,canary,FAIL,%s\n",
                                 tc.name().c_str(),
                                 vlens_str(unattr).c_str());
                }
            }
            std::cout.flush();
        }
        if (report)
            std::fclose(report);
        std::cerr << "\noutput-canary sweep: " << c_failed << " / " << c_tested
                  << " kernels over-ran the output buffer; " << c_partial
                  << " left in-bounds elements unwritten (reduction/index kernels — "
                     "triage)\n";
        return c_failed > 0 ? 1 : 0;
    }

    // #90 input-immutability mode (opt-in via HARNESS_IMMUTABLE). Like the #89
    // canary it is a distinct mode that replaces the correctness sweep; the default
    // qa suite and the toggle-off correctness sweep are entirely unaffected.
    const bool immutable_mode = (std::getenv("HARNESS_IMMUTABLE") != nullptr);
    if (immutable_mode) {
        // Hard negative control (any build), exercising the SAME
        // run_volk_immutability_test path real kernels use. Two planted kernels:
        //   ok        -> input untouched         (must NOT be flagged)
        //   scribble  -> writes in[0]            (must trip the mutation check)
        // Any deviation means the detector is broken -- fail hard (exit 2), per #88.
        const unsigned int nc_vlen = 127;
        std::vector<volk_test_results_t> nc;
        const volk_immutability_summary ok_sum =
            run_volk_immutability_test(volk_canary_desc(),
                                       (void (*)())(&volk_32f_canaryok_32f),
                                       "volk_32f_canaryok_32f",
                                       scalar,
                                       nc_vlen,
                                       &nc,
                                       kFloatEdges,
                                       kComplexEdges);
        nc.clear();
        const volk_immutability_summary scribble_sum =
            run_volk_immutability_test(volk_canary_desc(),
                                       (void (*)())(&volk_32f_inputscribble_32f),
                                       "volk_32f_inputscribble_32f",
                                       scalar,
                                       nc_vlen,
                                       &nc,
                                       kFloatEdges,
                                       kComplexEdges);
        nc.clear();
        const char* nc_err = nullptr;
        if (ok_sum.mutated) {
            nc_err = "the correct planted kernel volk_32f_canaryok_32f was flagged — "
                     "the immutability check over-reports";
        } else if (!scribble_sum.mutated) {
            nc_err = "the planted kernel volk_32f_inputscribble_32f did not trip the "
                     "immutability check — the detector is blind to input writes";
        }
        if (nc_err) {
            std::cerr << "NEGATIVE CONTROL LOST: " << nc_err << ". Aborting.\n";
            if (report)
                std::fclose(report);
            return 2;
        }
        std::cerr << "immutability negative control OK: ok-kernel clean, scribble "
                     "trips the input-immutability check.\n";

        // Best-effort per-kernel immutability sweep over the real kernels.
        //   FAIL -> the kernel wrote a byte of an input buffer (always a defect).
        //   skip -> in-place / unsupported signature / no impl observed (fail closed).
        // Puppets are skipped: their buffer plumbing differs from a plain kernel call.
        std::cout << "# input-immutability sweep: vlens 1..40, 131071, 1000003\n";
        std::cout.flush();
        int m_tested = 0, m_failed = 0;
        for (auto& tc : test_cases) {
            if (!filter.empty() && tc.name() != filter)
                continue;
            if (tc.puppet_master_name() != "NULL") {
                std::cout << "skip  [immutable] " << tc.name() << " (puppet)\n";
                if (report)
                    std::fprintf(report, "%s,-,immutable,skip,\n", tc.name().c_str());
                std::cout.flush();
                continue;
            }
            std::vector<unsigned int> mutated;
            bool any_applied = false;
            std::set<std::string> impls_seen;
            std::map<std::string, std::vector<unsigned int>> mutated_fails;
            for (unsigned int v : vlens) {
                const volk_immutability_summary s = quiet_immutability_run(tc, v);
                if (s.applied)
                    any_applied = true;
                if (s.mutated)
                    mutated.push_back(v);
                for (const std::string& impl : s.checked_impls)
                    impls_seen.insert(impl);
                for (const std::string& impl : s.mutated_impls)
                    mutated_fails[impl].push_back(v);
            }
            // A kernel no impl could check (in-place / unsupported) is SKIPPED, not ok
            // -- an unchecked kernel must not masquerade as covered. Fail closed: skip
            // only when nothing was observed at all.
            if (!any_applied && mutated.empty()) {
                std::cout << "skip  [immutable] " << tc.name()
                          << " (no protectable input buffer)\n";
                if (report)
                    std::fprintf(report, "%s,-,immutable,skip,\n", tc.name().c_str());
                std::cout.flush();
                continue;
            }
            ++m_tested;
            if (!mutated.empty()) {
                ++m_failed;
                std::cout << "FAIL  [immutable] " << tc.name()
                          << "  mutated-input vlens:";
                for (unsigned int v : mutated)
                    std::cout << " " << v;
            } else {
                std::cout << "ok    [immutable] " << tc.name();
            }
            std::cout << "\n";
            if (report) {
                // No unattributed "-" row here: quiet_immutability_run maps
                // exceptions to skip (applied=false, mutated=false), so every
                // mutated vlen is always attributed to an impl.
                emit_impl_rows(report,
                               tc.name(),
                               "immutable",
                               impls_seen,
                               { { &mutated_fails, "FAIL" } });
            }
            std::cout.flush();
        }
        if (report)
            std::fclose(report);
        std::cerr << "\ninput-immutability sweep: " << m_failed << " / " << m_tested
                  << " kernels wrote a byte of an input buffer\n";
        return m_failed > 0 ? 1 : 0;
    }

    // #91 misaligned-run mode (opt-in via HARNESS_MISALIGNED). Distinct mode like
    // the #89/#90 siblings; default qa and the toggle-off sweep are untouched.
    // NOTE (ASan): this mode installs its own SIGSEGV handler inside
    // run_volk_misaligned_test; under AddressSanitizer run with
    // ASAN_OPTIONS=handle_segv=0:handle_sigbus=0:handle_sigill=0:allow_user_segv_handler=1
    // or ASan's handler wins and aborts the run on the first planted fault.
    const bool misaligned_mode = (std::getenv("HARNESS_MISALIGNED") != nullptr);
    if (misaligned_mode) {
        // Hard negative control: ok-kernel must pass misaligned (no over-report);
        // the planted aligned-load kernel must produce a RECORDED crash while this
        // process continues to this very check (fault isolation works). Any
        // deviation = broken detector, exit 2.
        const unsigned int nc_vlen = 127;
        std::vector<volk_test_results_t> nc;
        const volk_misaligned_summary ok_sum =
            run_volk_misaligned_test(volk_canary_desc(),
                                     (void (*)())(&volk_32f_canaryok_32f),
                                     "volk_32f_canaryok_32f",
                                     scalar,
                                     tol,
                                     false /*absolute_mode*/,
                                     nc_vlen,
                                     &nc,
                                     kFloatEdges,
                                     kComplexEdges);
        nc.clear();
        const volk_misaligned_summary fault_sum =
            run_volk_misaligned_test(volk_canary_desc(),
                                     (void (*)())(&volk_32f_misalignedfault_32f),
                                     "volk_32f_misalignedfault_32f",
                                     scalar,
                                     tol,
                                     false /*absolute_mode*/,
                                     nc_vlen,
                                     &nc,
                                     kFloatEdges,
                                     kComplexEdges);
        nc.clear();
        const char* nc_err = nullptr;
        if (ok_sum.crashed || ok_sum.diverged) {
            nc_err = "the correct planted kernel volk_32f_canaryok_32f was flagged "
                     "on misaligned buffers — the detector over-reports";
        } else if (!fault_sum.applied) {
            nc_err = "the misaligned mode could not run at all (unsupported "
                     "platform — POSIX signals required — or unsupported "
                     "signature/alignment); refusing to report coverage";
        } else if (!fault_sum.crashed) {
            nc_err = "the planted aligned-load kernel volk_32f_misalignedfault_32f "
                     "did not produce a recorded crash — fault isolation is broken "
                     "or the misalignment never reached the impl";
        }
        if (nc_err) {
            std::cerr << "NEGATIVE CONTROL LOST: " << nc_err << ". Aborting.\n";
            if (report)
                std::fclose(report);
            return 2;
        }
        std::cerr << "misaligned negative control OK: ok-kernel clean, aligned-load "
                     "kernel crash recorded and the run continued.\n";

        // Per-kernel misaligned sweep over the real kernels (unaligned impls only).
        //   FAIL -> an impl crashed on misaligned buffers or diverged from the
        //           the SAME impl's aligned run (both always defects).
        //   skip -> puppet / unsupported signature / nothing observed (fail closed).
        // Puppets are skipped -- load-bearing here: it is what keeps the longjmp
        // safety argument airtight (no impl with internal allocation runs under the
        // signal guard) and keeps #96's conv_k7 out of this mode.
        std::cout << "# misaligned sweep: vlens 1..40, 131071, 1000003\n";
        std::cout.flush();
        int a_tested = 0, a_failed = 0, a_crashed = 0, a_diverged = 0;
        for (auto& tc : test_cases) {
            if (!filter.empty() && tc.name() != filter)
                continue;
            if (tc.puppet_master_name() != "NULL") {
                std::cout << "skip  [misaligned] " << tc.name() << " (puppet)\n";
                if (report)
                    std::fprintf(report, "%s,-,misaligned,skip,\n", tc.name().c_str());
                std::cout.flush();
                continue;
            }
            std::vector<unsigned int> crash_vlens;
            std::vector<unsigned int> diverge_vlens;
            bool any_applied = false;
            std::set<std::string> impls_seen;
            std::map<std::string, std::vector<unsigned int>> crash_fails;
            std::map<std::string, std::vector<unsigned int>> diverge_fails;
            for (unsigned int v : vlens) {
                const volk_misaligned_summary s = quiet_misaligned_run(tc, v);
                if (s.applied)
                    any_applied = true;
                if (s.crashed)
                    crash_vlens.push_back(v);
                if (s.diverged)
                    diverge_vlens.push_back(v);
                for (const std::string& impl : s.checked_impls)
                    impls_seen.insert(impl);
                for (const std::string& impl : s.crashed_impls)
                    crash_fails[impl].push_back(v);
                for (const std::string& impl : s.diverged_impls)
                    diverge_fails[impl].push_back(v);
            }
            if (!any_applied && crash_vlens.empty() && diverge_vlens.empty()) {
                std::cout << "skip  [misaligned] " << tc.name()
                          << " (no checkable unaligned impl)\n";
                if (report)
                    std::fprintf(report, "%s,-,misaligned,skip,\n", tc.name().c_str());
                std::cout.flush();
                continue;
            }
            ++a_tested;
            if (!crash_vlens.empty() || !diverge_vlens.empty()) {
                ++a_failed;
                if (!crash_vlens.empty())
                    ++a_crashed;
                if (!diverge_vlens.empty())
                    ++a_diverged;
                std::cout << "FAIL  [misaligned] " << tc.name();
                if (!crash_vlens.empty()) {
                    std::cout << "  crash vlens:";
                    for (unsigned int v : crash_vlens)
                        std::cout << " " << v;
                }
                if (!diverge_vlens.empty()) {
                    std::cout << "  diverge vlens:";
                    for (unsigned int v : diverge_vlens)
                        std::cout << " " << v;
                }
            } else {
                std::cout << "ok    [misaligned] " << tc.name();
            }
            std::cout << "\n";
            if (report) {
                // Crash and diverge rows are both FAIL; an impl hit by both
                // concatenates its vlens (#92). No "-" row: crashes are
                // recorded inside run_volk_misaligned_test with impl
                // attribution, and non-signal exceptions are skips.
                emit_impl_rows(report,
                               tc.name(),
                               "misaligned",
                               impls_seen,
                               { { &crash_fails, "FAIL" }, { &diverge_fails, "FAIL" } });
            }
            std::cout.flush();
        }
        if (report)
            std::fclose(report);
        std::cerr << "\nmisaligned sweep: " << a_crashed << " kernels crashed, "
                  << a_diverged << " diverged, of " << a_tested
                  << " tested (unaligned impls only)\n";
        return a_failed > 0 ? 1 : 0;
    }

    // #92 combined negative control (opt-in via HARNESS_COMBINED_NC; wired as
    // ctest qa_harness_negative_control). Re-runs the two escaped defects that
    // motivated epic #85 as PLANTED COPIES (the live kernels get fixed by the
    // per-kernel campaign; these copies do not) and asserts each is caught by
    // the capability built to catch it — and ONLY for that reason:
    //   power: flagged by the independent reference (#88); invisible to
    //          impl-vs-impl (single impl — the original blind spot).
    //   tanh:  flagged by the tail/edge sweep (#87); invisible on smooth data
    //          (the original blind spot — random data is uniform [-1,1] and
    //          never reaches the 4.97 clamp).
    // The corrected twins must pass, which is the continuously-enforced form of
    // "reverting the reintroduction makes the harness pass". Any deviation =>
    // exit 2. NOTE: volk_add_test passes the ctest name as argv[1]; this mode
    // ignores argv entirely.
    if (std::getenv("HARNESS_COMBINED_NC") != nullptr) {
        if (report)
            std::fclose(report);
        // The sweep's own edge set, minus the 4.97f clamp boundary (ambiguous
        // for the planted tanh pair: `> 4.97` excludes it from the clamp while
        // the series polynomial degrades near it). Deriving from kFloatEdges
        // keeps the control coupled to the edges the live sweep actually
        // injects; 5/6/8 remain firmly inside the |x|>4.97 clamp.
        std::vector<float> ncEdges;
        for (float e : kFloatEdges) {
            if (e != 4.97f)
                ncEdges.push_back(e);
        }
        const std::vector<float> noFloatEdges;
        const std::vector<lv_32fc_t> noComplexEdges;
        // 2..40 covers every remainder for widths {2,4,8,16,32} (power's signal
        // needs vlen >= 2: at vlen 1 the only element is the (0,0) edge, equal
        // under both formulas); 127 is a prime covering odd remainders at a
        // non-trivial size.
        std::vector<unsigned int> nc_vlens;
        for (unsigned int v = 2; v <= 40; ++v)
            nc_vlens.push_back(v);
        nc_vlens.push_back(127);

        const volk_reference_entry* pref =
            volk_reference_lookup("volk_32fc_s32f_power_32fc");
        if (!pref) {
            std::cerr << "NEGATIVE CONTROL LOST: volk_32fc_s32f_power_32fc is not "
                         "reference-registered (dropped/renamed registration).\n";
            return 2;
        }
        const lv_32fc_t power_scalar = lv_cmake(2.5f, 0.0f);
        volk_test_params_t nc_params(1e-2f, power_scalar, 127, 1, false, "");

        volk_test_case_t power_defect_tc(volk_power_nc_desc(),
                                         (void (*)())(&volk_32fc_s32f_powerdefect_32fc),
                                         "volk_32fc_s32f_powerdefect_32fc",
                                         nc_params);
        volk_test_case_t power_ok_tc(volk_power_nc_desc(),
                                     (void (*)())(&volk_32fc_s32f_powerok_32fc),
                                     "volk_32fc_s32f_powerok_32fc",
                                     nc_params);
        volk_test_case_t tanh_stale_tc(volk_tanh_nc_desc(),
                                       (void (*)())(&volk_32f_tanhstale_32f),
                                       "volk_32f_tanhstale_32f",
                                       nc_params);
        volk_test_case_t tanh_ok_tc(volk_tanh_nc_desc(),
                                    (void (*)())(&volk_32f_tanhok_32f),
                                    "volk_32f_tanhok_32f",
                                    nc_params);

        // Run one planted case across the NC vlens; "flagged" = any vlen FAILed.
        // In reference runs the tol argument is unused (the registry entry's tol
        // governs inside run_volk_reference_test), so those pass 0.0f.
        auto sweep_nc = [&](volk_test_case_t& tc,
                            const volk_reference_entry* ref,
                            float tol_,
                            const std::vector<float>& fe,
                            const std::vector<lv_32fc_t>& ce) {
            bool flagged = false;
            for (unsigned int v : nc_vlens)
                flagged |= quiet_run(
                    tc, ref, tol_, power_scalar, false /*absolute_mode*/, 1, v, fe, ce);
            return flagged;
        };

        struct nc_check {
            const char* what;
            bool flagged;
            bool must_flag;
            const char* lost_msg;
        };
        const nc_check checks[] = {
            { "power-defect via independent reference (#88)",
              sweep_nc(power_defect_tc, pref, 0.0f, ncEdges, kComplexEdges),
              true,
              "the planted swapped-atan2 power defect passed reference mode — the "
              "independent-reference capability is blind to the defect class that "
              "motivated it" },
            // Enforced mechanism for the blind-spot proof: with a single-impl
            // desc, setup_test_data's "need >= 2 archs" guard makes
            // run_volk_tests return false without executing the kernel —
            // exactly how historical qa skipped single-impl kernels. The
            // check is not vacuous: it pins that behavior, and flags any
            // future change that makes impl-vs-impl claim coverage here.
            { "power-defect via impl-vs-impl (blind-spot proof)",
              sweep_nc(power_defect_tc, nullptr, 1e-4f, ncEdges, kComplexEdges),
              false,
              "impl-vs-impl flagged the single-impl power defect — the blind-spot "
              "premise no longer holds; re-examine the control (and whether #88 is "
              "still the capability that catches it)" },
            { "power-ok via independent reference (revert proof)",
              sweep_nc(power_ok_tc, pref, 0.0f, ncEdges, kComplexEdges),
              false,
              "the CORRECTED power twin was flagged by reference mode — the "
              "reference over-reports (or its tolerance regressed)" },
            { "tanh-stale via tail/edge sweep (#87)",
              sweep_nc(tanh_stale_tc, nullptr, 1e-2f, ncEdges, kComplexEdges),
              true,
              "the planted stale-input-pointer tanh defect passed the tail/edge "
              "sweep — the edge-injection capability is blind to the defect class "
              "that motivated it" },
            { "tanh-stale without edges (reason-specificity proof)",
              sweep_nc(tanh_stale_tc, nullptr, 1e-2f, noFloatEdges, noComplexEdges),
              false,
              "the stale tanh defect was flagged WITHOUT edge injection — detection "
              "is happening for a different reason than the edge capability; the "
              "control no longer proves what it claims" },
            { "tanh-ok via tail/edge sweep (revert proof)",
              sweep_nc(tanh_ok_tc, nullptr, 1e-2f, ncEdges, kComplexEdges),
              false,
              "the CORRECTED tanh twin was flagged by the sweep — the sweep "
              "over-reports" },
        };
        bool lost = false;
        for (const nc_check& c : checks) {
            const bool good = (c.flagged == c.must_flag);
            std::cout << (good ? "ok    " : "LOST  ") << "[combined-nc] " << c.what
                      << "\n";
            if (!good) {
                std::cerr << "NEGATIVE CONTROL LOST: " << c.lost_msg << "\n";
                lost = true;
            }
        }
        if (lost)
            return 2;
        std::cerr << "combined negative control OK: power flagged by the independent "
                     "reference only; tanh flagged by the tail/edge sweep only; "
                     "corrected twins pass.\n";
        return 0;
    }

    std::cout << "# kernel-correctness remainder sweep: vlens 1..40, 131071, 1000003\n";
    std::cout.flush();

    int tested = 0, failed = 0;
    // Negative-control tracking: power_seen = power_32fc was in the (filtered) sweep
    // at all; power_ref_tested = it ran in reference mode. The two together detect a
    // dropped/renamed registration (the likelier regression), not just a broken oracle.
    bool power_seen = false, power_ref_tested = false, power_ref_failed = false;
    for (auto& tc : test_cases) {
        if (!filter.empty() && tc.name() != filter)
            continue;
        ++tested;
        // Registered kernels run against the independent double reference (#88);
        // the rest fall back to the impl-vs-impl comparison.
        const volk_reference_entry* ref = volk_reference_lookup(tc.name());
        const char* mode = ref ? "ref" : "impl";
        // The sweep judges each kernel against its OWN registered QA contract
        // (#106): scalar always (driver-default 327 turns power into inf and
        // clamp into an inverted range), tol/absolute_mode in impl mode only —
        // ref mode ignores them (the oracle applies the registry entry's bound).
        volk_test_params_t kp = tc.test_parameters(); // accessors are non-const
        const lv_32fc_t kscalar = kp.scalar();
        const float ktol = ref ? tol : kp.tol();
        const bool kabs = ref ? false : kp.absolute_mode();
        std::vector<unsigned int> bad;
        std::set<std::string> impls_seen;
        std::map<std::string, std::vector<unsigned int>> impl_fails;
        for (unsigned int v : vlens) {
            // Per-impl collection is only consumed by the CSV report; skip the
            // per-(kernel,vlen) map building when no report was requested.
            if (quiet_run(tc,
                          ref,
                          ktol,
                          kscalar,
                          kabs,
                          iter,
                          v,
                          kFloatEdges,
                          kComplexEdges,
                          report ? &impls_seen : nullptr,
                          report ? &impl_fails : nullptr))
                bad.push_back(v);
        }
        if (tc.name() == "volk_32fc_s32f_power_32fc") {
            power_seen = true;
            // The control's signal needs a vlen >= 2: at vlen 1 the only element is
            // kComplexEdges[0] = (0,0), for which both the swapped-atan2 kernel and the
            // oracle yield 0, so the defect is invisible. The sweep includes 2..40.
            power_ref_tested = (ref != nullptr);
            power_ref_failed = ref && !bad.empty();
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
            // `mode` is ref|impl per the kernel's reference registration (#92).
            emit_impl_rows(
                report, tc.name(), mode, impls_seen, { { &impl_fails, "FAIL" } });
            // Failures with no impl attribution (an exception during the run,
            // or nothing observed at all) stay visible on a "-" row.
            const std::vector<unsigned int> unattr = unattributed_vlens(bad, impl_fails);
            if (!unattr.empty()) {
                std::fprintf(report,
                             "%s,-,%s,FAIL,%s\n",
                             tc.name().c_str(),
                             mode,
                             vlens_str(unattr).c_str());
            }
        }
        std::cout.flush();
    }
    if (report)
        std::fclose(report);
    std::cerr << "\ncorrectness remainder sweep: " << failed << " / " << tested
              << " kernels failed\n";

    // #88 negative control: the live swapped-atan2 defect in volk_32fc_s32f_power_32fc
    // makes it wrong for every input, but it has no SIMD impl so the impl-vs-impl
    // harness reports `ok`. In reference mode the double oracle MUST fail it. If it
    // passes, the reference machinery is broken — fail hard (exit 2). Bug-gated: when
    // the e29bc7f power fix lands on this branch, flip this to expect-pass.
    if (power_seen && !power_ref_tested) {
        std::cerr << "NEGATIVE CONTROL LOST: volk_32fc_s32f_power_32fc ran but is NOT "
                     "reference-registered (dropped/renamed registration) — the "
                     "swapped-atan2 defect would go undetected.\n";
        return 2;
    }
    if (power_ref_tested && !power_ref_failed) {
        std::cerr << "NEGATIVE CONTROL LOST: volk_32fc_s32f_power_32fc passed in "
                     "reference mode but its swapped-atan2 defect is present — the "
                     "reference harness is broken.\n";
        return 2;
    }

    return failed > 0 ? 1 : 0;
}
