/* -*- c++ -*- */
/*
 * Copyright 2025 Matt Tibbits
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*
 * volk_ulp_profile — Measure floating-point accuracy (ULP error) of each
 * VOLK kernel implementation relative to the generic (scalar) reference.
 *
 * ULP = "Unit in the Last Place".  Two IEEE-754 floats that are adjacent
 * representable values differ by 1 ULP.  This tool computes the ULP distance
 * between each SIMD implementation's output and the generic output for every
 * element, then reports max / mean / RMS statistics per implementation.
 *
 * For integer-output kernels the tool reports absolute difference instead.
 *
 * Usage:
 *   volk_ulp_profile [-R kernel_substr] [-v vlen] [-j output.json]
 */

#include <volk/volk.h>
#include <volk/volk_malloc.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <fmt/core.h>

#include "kernel_tests.h"
#include "qa_utils.h"
#include "volk/volk_complex.h"
#include "volk_option_helpers.h"

static constexpr const char* kGenericImplName = "generic";

// ULP distance between two IEEE-754 floats of type Float (float or double).
// UInt must be the unsigned integer type of the same width. Returns
// UINT64_MAX for NaN/Inf mismatches.
//
// Dawson's algorithm: reinterpret each float as its bit pattern, then map
// the sign-magnitude IEEE 754 representation onto a linear unsigned range
// where adjacent floats have adjacent integer values and both zeros map
// to the same integer. The absolute difference is the ULP distance.
template <typename Float, typename UInt>
static uint64_t ulp_distance(Float a, Float b)
{
    static_assert(sizeof(Float) == sizeof(UInt), "type width mismatch");

    if (std::isnan(a) && std::isnan(b)) {
        return 0;
    }
    if (std::isnan(a) || std::isnan(b)) {
        return UINT64_MAX;
    }
    if (std::isinf(a) && std::isinf(b)) {
        return (a == b) ? 0 : UINT64_MAX;
    }
    if (std::isinf(a) || std::isinf(b)) {
        return UINT64_MAX;
    }

    constexpr UInt sign_bit = UInt(1) << (sizeof(UInt) * 8 - 1);
    UInt ua, ub;
    std::memcpy(&ua, &a, sizeof(ua));
    std::memcpy(&ub, &b, sizeof(ub));

    if (ua & sign_bit) {
        ua = ~ua + UInt(1);
    } else {
        ua += sign_bit;
    }

    if (ub & sign_bit) {
        ub = ~ub + UInt(1);
    } else {
        ub += sign_bit;
    }

    return (ua > ub) ? (ua - ub) : (ub - ua);
}

// -----------------------------------------------------------------------
// Per-implementation ULP statistics
// -----------------------------------------------------------------------
struct ulp_stats_t {
    std::string arch;
    uint64_t max_ulp = 0;
    double sum_ulp = 0.0;
    double sum_ulp_sq = 0.0;
    uint64_t count = 0;
    bool has_nan_mismatch = false;
    bool is_integer = false; // true → report as abs diff, not ULP

    double mean_ulp() const { return count ? sum_ulp / count : 0.0; }
    double rms_ulp() const { return count ? std::sqrt(sum_ulp_sq / count) : 0.0; }
};

template <typename Float, typename UInt>
static void accumulate_ulp(const Float* expected,
                           const Float* actual,
                           unsigned int n,
                           ulp_stats_t& stats)
{
    for (unsigned int i = 0; i < n; i++) {
        uint64_t d = ulp_distance<Float, UInt>(expected[i], actual[i]);
        if (d == UINT64_MAX) {
            stats.has_nan_mismatch = true;
            continue;
        }
        if (d > stats.max_ulp) {
            stats.max_ulp = d;
        }
        stats.sum_ulp += static_cast<double>(d);
        stats.sum_ulp_sq += static_cast<double>(d) * static_cast<double>(d);
        stats.count++;
    }
}

// Accumulate absolute-difference stats for integer arrays
template <typename T>
static void accumulate_int_diff(const T* expected,
                                const T* actual,
                                unsigned int n,
                                ulp_stats_t& stats)
{
    for (unsigned int i = 0; i < n; i++) {
        uint64_t d = static_cast<uint64_t>(std::abs(static_cast<int64_t>(expected[i]) -
                                                    static_cast<int64_t>(actual[i])));
        if (d > stats.max_ulp) {
            stats.max_ulp = d;
        }
        stats.sum_ulp += static_cast<double>(d);
        stats.sum_ulp_sq += static_cast<double>(d) * static_cast<double>(d);
        stats.count++;
    }
    stats.is_integer = true;
}

struct kernel_ulp_result_t {
    std::string kernel_name;
    std::vector<ulp_stats_t> arch_stats;
};

static kernel_ulp_result_t run_ulp_analysis(volk_test_case_t& test_case,
                                            unsigned int vlen)
{
    kernel_ulp_result_t result;
    result.kernel_name = test_case.name();

    volk_func_desc_t desc = test_case.desc();
    void (*manual_func)() = test_case.kernel_ptr();
    volk_test_params_t params = test_case.test_parameters();

    std::vector<std::string> arch_list = get_arch_list(desc);

    if (arch_list.size() < 2) {
        fmt::print("  {} — only 1 impl, skipping\n", result.kernel_name);
        return result;
    }

    std::vector<volk_type_t> inputsig, outputsig;
    try {
        get_signatures_from_name(inputsig, outputsig, test_case.name());
    } catch (const std::exception& e) {
        fmt::print(stderr,
                   "  {} — cannot parse signature ({}), skipping\n",
                   result.kernel_name,
                   e.what());
        return result;
    } catch (...) {
        fmt::print(stderr,
                   "  {} — cannot parse signature (unknown error), skipping\n",
                   result.kernel_name);
        return result;
    }

    std::vector<volk_type_t> inputsc;
    for (size_t i = 0; i < inputsig.size(); i++) {
        if (inputsig[i].is_scalar) {
            inputsc.push_back(inputsig[i]);
            inputsig.erase(inputsig.begin() + i);
            i--;
        }
    }

    volk_qa_aligned_mem_pool pool;

    std::vector<void*> inbuffs;
    for (size_t i = 0; i < inputsig.size(); i++) {
        inbuffs.push_back(
            pool.get_new(vlen * inputsig[i].size * (inputsig[i].is_complex ? 2 : 1)));
    }
    for (size_t i = 0; i < inbuffs.size(); i++) {
        load_random_data(inbuffs[i],
                         inputsig[i],
                         vlen,
                         params.float_edge_cases(),
                         params.complex_edge_cases());
    }

    std::vector<volk_type_t> both_sigs;
    both_sigs.insert(both_sigs.end(), outputsig.begin(), outputsig.end());
    both_sigs.insert(both_sigs.end(), inputsig.begin(), inputsig.end());

    std::vector<std::vector<void*>> test_data;
    for (size_t i = 0; i < arch_list.size(); i++) {
        std::vector<void*> arch_buffs;
        for (size_t j = 0; j < outputsig.size(); j++) {
            arch_buffs.push_back(pool.get_new(vlen * outputsig[j].size *
                                              (outputsig[j].is_complex ? 2 : 1)));
        }
        for (size_t j = 0; j < inputsig.size(); j++) {
            void* buf =
                pool.get_new(vlen * inputsig[j].size * (inputsig[j].is_complex ? 2 : 1));
            std::memcpy(buf,
                        inbuffs[j],
                        vlen * inputsig[j].size * (inputsig[j].is_complex ? 2 : 1));
            arch_buffs.push_back(buf);
        }
        test_data.push_back(arch_buffs);
    }

    lv_32fc_t scalar = params.scalar();
    for (size_t i = 0; i < arch_list.size(); i++) {
        switch (both_sigs.size()) {
        case 1:
            if (inputsc.size() == 0) {
                ((volk_fn_1arg)manual_func)(test_data[i][0], vlen, arch_list[i].c_str());
            } else if (inputsc.size() == 1 && inputsc[0].is_float) {
                if (inputsc[0].is_complex) {
                    ((volk_fn_1arg_s32fc)manual_func)(
                        test_data[i][0], &scalar, vlen, arch_list[i].c_str());
                } else {
                    ((volk_fn_1arg_s32f)manual_func)(
                        test_data[i][0], scalar.real(), vlen, arch_list[i].c_str());
                }
            }
            break;
        case 2:
            if (inputsc.size() == 0) {
                ((volk_fn_2arg)manual_func)(
                    test_data[i][0], test_data[i][1], vlen, arch_list[i].c_str());
            } else if (inputsc.size() == 1 && inputsc[0].is_float) {
                if (inputsc[0].is_complex) {
                    ((volk_fn_2arg_s32fc)manual_func)(test_data[i][0],
                                                      test_data[i][1],
                                                      &scalar,
                                                      vlen,
                                                      arch_list[i].c_str());
                } else {
                    ((volk_fn_2arg_s32f)manual_func)(test_data[i][0],
                                                     test_data[i][1],
                                                     scalar.real(),
                                                     vlen,
                                                     arch_list[i].c_str());
                }
            }
            break;
        case 3:
            if (inputsc.size() == 0) {
                ((volk_fn_3arg)manual_func)(test_data[i][0],
                                            test_data[i][1],
                                            test_data[i][2],
                                            vlen,
                                            arch_list[i].c_str());
            } else if (inputsc.size() == 1 && inputsc[0].is_float) {
                if (inputsc[0].is_complex) {
                    ((volk_fn_3arg_s32fc)manual_func)(test_data[i][0],
                                                      test_data[i][1],
                                                      test_data[i][2],
                                                      &scalar,
                                                      vlen,
                                                      arch_list[i].c_str());
                } else {
                    ((volk_fn_3arg_s32f)manual_func)(test_data[i][0],
                                                     test_data[i][1],
                                                     test_data[i][2],
                                                     scalar.real(),
                                                     vlen,
                                                     arch_list[i].c_str());
                }
            }
            break;
        case 4:
            ((volk_fn_4arg)manual_func)(test_data[i][0],
                                        test_data[i][1],
                                        test_data[i][2],
                                        test_data[i][3],
                                        vlen,
                                        arch_list[i].c_str());
            break;
        default:
            fmt::print(stderr,
                       "volk_ulp_profile: unsupported arg count {} for kernel {}, "
                       "skipping\n",
                       both_sigs.size(),
                       test_case.name());
            return result;
        }
    }

    size_t generic_idx = 0;
    for (size_t i = 0; i < arch_list.size(); i++) {
        if (arch_list[i] == kGenericImplName) {
            generic_idx = i;
            break;
        }
    }

    for (size_t i = 0; i < arch_list.size(); i++) {
        if (i == generic_idx) {
            continue;
        }

        ulp_stats_t stats;
        stats.arch = arch_list[i];

        for (size_t j = 0; j < outputsig.size(); j++) {
            unsigned int elem_count = vlen * (outputsig[j].is_complex ? 2 : 1);

            if (outputsig[j].is_float) {
                if (outputsig[j].size == 8) {
                    accumulate_ulp<double, uint64_t>(
                        static_cast<const double*>(test_data[generic_idx][j]),
                        static_cast<const double*>(test_data[i][j]),
                        elem_count,
                        stats);
                } else {
                    accumulate_ulp<float, uint32_t>(
                        static_cast<const float*>(test_data[generic_idx][j]),
                        static_cast<const float*>(test_data[i][j]),
                        elem_count,
                        stats);
                }
            } else {
                switch (outputsig[j].size) {
                case 8:
                    if (outputsig[j].is_signed) {
                        accumulate_int_diff(
                            static_cast<const int64_t*>(test_data[generic_idx][j]),
                            static_cast<const int64_t*>(test_data[i][j]),
                            elem_count,
                            stats);
                    } else {
                        accumulate_int_diff(
                            static_cast<const uint64_t*>(test_data[generic_idx][j]),
                            static_cast<const uint64_t*>(test_data[i][j]),
                            elem_count,
                            stats);
                    }
                    break;
                case 4:
                    if (outputsig[j].is_signed) {
                        accumulate_int_diff(
                            static_cast<const int32_t*>(test_data[generic_idx][j]),
                            static_cast<const int32_t*>(test_data[i][j]),
                            elem_count,
                            stats);
                    } else {
                        accumulate_int_diff(
                            static_cast<const uint32_t*>(test_data[generic_idx][j]),
                            static_cast<const uint32_t*>(test_data[i][j]),
                            elem_count,
                            stats);
                    }
                    break;
                case 2:
                    if (outputsig[j].is_signed) {
                        accumulate_int_diff(
                            static_cast<const int16_t*>(test_data[generic_idx][j]),
                            static_cast<const int16_t*>(test_data[i][j]),
                            elem_count,
                            stats);
                    } else {
                        accumulate_int_diff(
                            static_cast<const uint16_t*>(test_data[generic_idx][j]),
                            static_cast<const uint16_t*>(test_data[i][j]),
                            elem_count,
                            stats);
                    }
                    break;
                case 1:
                    if (outputsig[j].is_signed) {
                        accumulate_int_diff(
                            static_cast<const int8_t*>(test_data[generic_idx][j]),
                            static_cast<const int8_t*>(test_data[i][j]),
                            elem_count,
                            stats);
                    } else {
                        accumulate_int_diff(
                            static_cast<const uint8_t*>(test_data[generic_idx][j]),
                            static_cast<const uint8_t*>(test_data[i][j]),
                            elem_count,
                            stats);
                    }
                    break;
                }
            }
        }

        result.arch_stats.push_back(stats);
    }

    return result;
}

// -----------------------------------------------------------------------
// JSON output
// -----------------------------------------------------------------------
static bool write_json(const std::string& filename,
                       const std::vector<kernel_ulp_result_t>& all_results)
{
    std::ofstream f(filename);
    if (!f.is_open()) {
        fmt::print(stderr, "volk_ulp_profile: cannot open {} for writing\n", filename);
        return false;
    }

    f << "{\n  \"ulp_results\": [\n";
    for (size_t k = 0; k < all_results.size(); k++) {
        const auto& kr = all_results[k];
        f << "    {\n";
        f << "      \"kernel\": \"" << kr.kernel_name << "\",\n";
        f << "      \"implementations\": [\n";
        for (size_t a = 0; a < kr.arch_stats.size(); a++) {
            const auto& s = kr.arch_stats[a];
            f << "        {\n";
            f << "          \"arch\": \"" << s.arch << "\",\n";
            f << "          \"metric\": \"" << (s.is_integer ? "abs_diff" : "ulp")
              << "\",\n";
            f << "          \"max\": " << s.max_ulp << ",\n";
            f << "          \"mean\": " << s.mean_ulp() << ",\n";
            f << "          \"rms\": " << s.rms_ulp() << ",\n";
            f << "          \"count\": " << s.count << ",\n";
            f << "          \"nan_mismatch\": " << (s.has_nan_mismatch ? "true" : "false")
              << "\n";
            f << "        }";
            if (a + 1 < kr.arch_stats.size()) {
                f << ",";
            }
            f << "\n";
        }
        f << "      ]\n";
        f << "    }";
        if (k + 1 < all_results.size()) {
            f << ",";
        }
        f << "\n";
    }
    f << "  ]\n}\n";
    f.close();
    fmt::print("JSON results written to {}\n", filename);
    return true;
}

// -----------------------------------------------------------------------
// main
// -----------------------------------------------------------------------

volk_test_params_t ulp_test_params(1e-6f, 327.f, 131071, 1, false, "");

std::vector<std::string> ulp_kernel_patterns;
void ulp_set_substr(std::string val) { ulp_kernel_patterns.push_back(val); }
void ulp_set_vlen(int val) { ulp_test_params.set_vlen((unsigned int)val); }
std::string ulp_json_filename;
void ulp_set_json(std::string val) { ulp_json_filename = val; }

int main(int argc, char* argv[])
{
    option_list options("volk_ulp_profile");
    options.add(option_t("tests-substr",
                         "R",
                         "Run tests matching substring (can be repeated)",
                         ulp_set_substr));
    options.add(option_t("vlen", "v", "Set vector length", ulp_set_vlen));
    options.add(option_t("json", "j", "Write results to JSON file", ulp_set_json));
    options.parse(argc, argv);

    if (options.present("help")) {
        return 0;
    }

    unsigned int vlen = ulp_test_params.vlen();

    // Initialize kernel test list
    std::vector<volk_test_case_t> test_cases = init_test_list(ulp_test_params);

    fmt::print("volk_ulp_profile: measuring ULP error for {} kernels (vlen={})\n",
               test_cases.size(),
               vlen);
    fmt::print("{:=<90}\n\n", "");

    std::vector<kernel_ulp_result_t> all_results;

    // Table column widths
    constexpr int w_arch = 28;
    constexpr int w_metric = 8;
    constexpr int w_max = 12;
    constexpr int w_mean = 12;
    constexpr int w_rms = 12;

    for (auto& tc : test_cases) {
        // Filter by pattern
        bool match = ulp_kernel_patterns.empty();
        for (const auto& pat : ulp_kernel_patterns) {
            if (tc.name().find(pat) != std::string::npos) {
                match = true;
                break;
            }
        }
        if (!match) {
            continue;
        }

        fmt::print("{}\n", tc.name());

        kernel_ulp_result_t kr;
        try {
            kr = run_ulp_analysis(tc, vlen);
        } catch (const std::exception& e) {
            fmt::print(stderr, "  ERROR: {}\n\n", e.what());
            continue;
        } catch (const std::string& e) {
            fmt::print(stderr, "  ERROR: {}\n\n", e);
            continue;
        } catch (const char* e) {
            fmt::print(stderr, "  ERROR: {}\n\n", e);
            continue;
        } catch (...) {
            fmt::print(stderr, "  ERROR: unknown exception\n\n");
            continue;
        }

        if (kr.arch_stats.empty()) {
            fmt::print("\n");
            continue;
        }

        fmt::print("  {:<{}} {:>{}} {:>{}} {:>{}} {:>{}}\n",
                   "arch",
                   w_arch,
                   "metric",
                   w_metric,
                   "max",
                   w_max,
                   "mean",
                   w_mean,
                   "rms",
                   w_rms);
        fmt::print("  {:-<{}}-{:-<{}}-{:-<{}}-{:-<{}}-{:-<{}}\n",
                   "",
                   w_arch,
                   "",
                   w_metric,
                   "",
                   w_max,
                   "",
                   w_mean,
                   "",
                   w_rms);

        auto sorted = kr.arch_stats;
        std::sort(
            sorted.begin(), sorted.end(), [](const ulp_stats_t& a, const ulp_stats_t& b) {
                return a.max_ulp > b.max_ulp;
            });

        for (const auto& s : sorted) {
            std::string metric = s.is_integer ? "abs_diff" : "ulp";
            std::string nan_flag = s.has_nan_mismatch ? " [NaN!]" : "";

            fmt::print("  {:<{}} {:>{}} {:>{}} {:>12.1f} {:>12.1f}{}\n",
                       s.arch,
                       w_arch,
                       metric,
                       w_metric,
                       s.max_ulp,
                       w_max,
                       s.mean_ulp(),
                       s.rms_ulp(),
                       nan_flag);
        }
        fmt::print("\n");

        all_results.push_back(kr);
    }

    // Summary: rank kernels by worst-case max ULP
    if (all_results.size() > 1) {
        fmt::print("{:=<90}\n", "");
        fmt::print("SUMMARY — Kernels ranked by worst-case max ULP (highest first)\n");
        fmt::print("{:-<90}\n", "");

        struct summary_entry {
            std::string kernel;
            std::string worst_arch;
            uint64_t max_ulp;
            bool is_integer;
        };

        std::vector<summary_entry> summary;
        for (const auto& kr : all_results) {
            summary_entry se;
            se.kernel = kr.kernel_name;
            se.max_ulp = 0;
            se.is_integer = false;
            for (const auto& s : kr.arch_stats) {
                if (s.max_ulp > se.max_ulp) {
                    se.max_ulp = s.max_ulp;
                    se.worst_arch = s.arch;
                    se.is_integer = s.is_integer;
                }
            }
            summary.push_back(se);
        }

        std::sort(summary.begin(),
                  summary.end(),
                  [](const summary_entry& a, const summary_entry& b) {
                      return a.max_ulp > b.max_ulp;
                  });

        fmt::print(
            "{:<50} {:>10} {:>8} {}\n", "kernel", "max_ulp", "metric", "worst_arch");
        fmt::print("{:-<50}-{:-<10}-{:-<8}-{:-<20}\n", "", "", "", "");

        for (const auto& se : summary) {
            fmt::print("{:<50} {:>10} {:>8} {}\n",
                       se.kernel,
                       se.max_ulp,
                       se.is_integer ? "abs_diff" : "ulp",
                       se.worst_arch);
        }
    }

    if (!ulp_json_filename.empty()) {
        if (!write_json(ulp_json_filename, all_results)) {
            return 1;
        }
    }

    return 0;
}
