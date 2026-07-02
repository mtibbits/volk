/* -*- c++ -*- */
/*
 * Copyright 2011 - 2020, 2022 Free Software Foundation, Inc.
 * Copyright 2025 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "qa_utils.h"
#include "volk_reference.h" // for the independent double-precision oracle registry (#88)
#include <volk/volk.h>

#include <volk/volk.h>        // for volk_func_desc_t
#include <volk/volk_malloc.h> // for volk_free, volk_m...

#include <assert.h>    // for assert
#include <stdint.h>    // for uint16_t, uint64_t
#include <sys/time.h>  // for CLOCKS_PER_SEC
#include <sys/types.h> // for int16_t, int32_t
#include <cassert>     // for assert (#90 immutability index invariant)
#include <chrono>
#include <cmath>    // for sqrt, fabs, abs
#include <csetjmp>  // for sigsetjmp/siglongjmp (#91 misaligned fault isolation)
#include <csignal>  // for sigaction, SIGSEGV/SIGBUS/SIGILL (#91)
#include <cstdint>  // for uint8_t, uintptr_t (#89 canary)
#include <cstdlib>  // for malloc, free (#89 canary)
#include <cstring>  // for memcpy, memset
#include <ctime>    // for clock
#include <iostream> // for cerr
#include <limits>   // for numeric_limits
#include <map>      // for map, map<>::mappe...
#include <memory>   // for unique_ptr (#89 canary)
#include <random>
#include <stdexcept> // for runtime_error (#88)
#include <vector>    // for vector, _Bit_refe...

#include <fmt/format.h>

// Warmup time for CPU frequency scaling (ms)
static double g_warmup_ms = 2000.0;
static bool g_warmup_done = false;

double volk_test_get_warmup_ms() { return g_warmup_ms; }
void volk_test_set_warmup_ms(double ms) { g_warmup_ms = ms; }
void volk_test_reset_warmup() { g_warmup_done = false; }

template <typename T>
void random_floats(void* buf, unsigned int n, std::default_random_engine& rnd_engine)
{
    T* array = static_cast<T*>(buf);
    std::uniform_real_distribution<T> uniform_dist(T(-1), T(1));
    for (unsigned int i = 0; i < n; i++) {
        array[i] = uniform_dist(rnd_engine);
    }
}

void load_random_data(void* data,
                      volk_type_t type,
                      unsigned int n,
                      const std::vector<float>& float_edge_cases,
                      const std::vector<lv_32fc_t>& complex_edge_cases)
{
    std::random_device rnd_device;
    std::default_random_engine rnd_engine(rnd_device());

    unsigned int edge_case_count = 0;

    // Inject complex edge cases for complex float types
    if (type.is_float && type.is_complex && !complex_edge_cases.empty()) {
        edge_case_count = std::min((unsigned int)complex_edge_cases.size(), n);
        if (type.size == 8) {
            lv_64fc_t* array = static_cast<lv_64fc_t*>(data);
            for (unsigned int i = 0; i < edge_case_count; i++) {
                array[i] = lv_cmake((double)lv_creal(complex_edge_cases[i]),
                                    (double)lv_cimag(complex_edge_cases[i]));
            }
        } else {
            lv_32fc_t* array = static_cast<lv_32fc_t*>(data);
            for (unsigned int i = 0; i < edge_case_count; i++) {
                array[i] = complex_edge_cases[i];
            }
        }
    }
    // Inject float edge cases for non-complex float types
    else if (type.is_float && !type.is_complex && !float_edge_cases.empty()) {
        edge_case_count = std::min((unsigned int)float_edge_cases.size(), n);
        if (type.size == 8) {
            double* array = static_cast<double*>(data);
            for (unsigned int i = 0; i < edge_case_count; i++) {
                array[i] = static_cast<double>(float_edge_cases[i]);
            }
        } else {
            float* array = static_cast<float*>(data);
            for (unsigned int i = 0; i < edge_case_count; i++) {
                array[i] = float_edge_cases[i];
            }
        }
    }

    unsigned int remaining_n = n - edge_case_count;
    if (type.is_complex)
        remaining_n *= 2;

    if (type.is_float) {
        if (type.size == 8) {
            double* array = static_cast<double*>(data);
            random_floats<double>(array + edge_case_count * (type.is_complex ? 2 : 1),
                                  remaining_n,
                                  rnd_engine);
        } else {
            float* array = static_cast<float*>(data);
            random_floats<float>(array + edge_case_count * (type.is_complex ? 2 : 1),
                                 remaining_n,
                                 rnd_engine);
        }
    } else {
        if (type.is_complex)
            n *= 2;
        switch (type.size) {
        case 8:
            if (type.is_signed) {
                std::uniform_int_distribution<int64_t> uniform_dist(
                    std::numeric_limits<int64_t>::min(),
                    std::numeric_limits<int64_t>::max());
                for (unsigned int i = 0; i < n; i++)
                    ((int64_t*)data)[i] = uniform_dist(rnd_engine);
            } else {
                std::uniform_int_distribution<uint64_t> uniform_dist(
                    std::numeric_limits<uint64_t>::min(),
                    std::numeric_limits<uint64_t>::max());
                for (unsigned int i = 0; i < n; i++)
                    ((uint64_t*)data)[i] = uniform_dist(rnd_engine);
            }
            break;
        case 4:
            if (type.is_signed) {
                std::uniform_int_distribution<int32_t> uniform_dist(
                    std::numeric_limits<int32_t>::min(),
                    std::numeric_limits<int32_t>::max());
                for (unsigned int i = 0; i < n; i++)
                    ((int32_t*)data)[i] = uniform_dist(rnd_engine);
            } else {
                std::uniform_int_distribution<uint32_t> uniform_dist(
                    std::numeric_limits<uint32_t>::min(),
                    std::numeric_limits<uint32_t>::max());
                for (unsigned int i = 0; i < n; i++)
                    ((uint32_t*)data)[i] = uniform_dist(rnd_engine);
            }
            break;
        case 2:
            if (type.is_signed) {
                std::uniform_int_distribution<int16_t> uniform_dist(-6, 6);
                for (unsigned int i = 0; i < n; i++)
                    ((int16_t*)data)[i] = uniform_dist(rnd_engine);
            } else {
                std::uniform_int_distribution<uint16_t> uniform_dist(
                    std::numeric_limits<uint16_t>::min(),
                    std::numeric_limits<uint16_t>::max());
                for (unsigned int i = 0; i < n; i++)
                    ((uint16_t*)data)[i] = uniform_dist(rnd_engine);
            }
            break;
        case 1:
            if (type.is_signed) {
                std::uniform_int_distribution<int16_t> uniform_dist(
                    std::numeric_limits<int8_t>::min(),
                    std::numeric_limits<int8_t>::max());
                for (unsigned int i = 0; i < n; i++)
                    ((int8_t*)data)[i] = uniform_dist(rnd_engine);
            } else {
                std::uniform_int_distribution<uint16_t> uniform_dist(
                    std::numeric_limits<uint8_t>::min(),
                    std::numeric_limits<uint8_t>::max());
                for (unsigned int i = 0; i < n; i++)
                    ((uint8_t*)data)[i] = uniform_dist(rnd_engine);
            }
            break;
        default:
            throw "load_random_data: no support for data size > 8 or < 1"; // no
                                                                           // shenanigans
                                                                           // here
        }
    }
}

static std::vector<std::string> get_arch_list(volk_func_desc_t desc)
{
    std::vector<std::string> archlist;

    for (size_t i = 0; i < desc.n_impls; i++) {
        archlist.push_back(std::string(desc.impl_names[i]));
    }

    return archlist;
}

template <typename T>
T volk_lexical_cast(const std::string& str)
{
    for (unsigned int c_index = 0; c_index < str.size(); ++c_index) {
        if (str.at(c_index) < '0' || str.at(c_index) > '9') {
            throw "not all numbers!";
        }
    }
    T var;
    std::istringstream iss;
    iss.str(str);
    iss >> var;
    // deal with any error bits that may have been set on the stream
    return var;
}

volk_type_t volk_type_from_string(std::string name)
{
    volk_type_t type;
    type.is_float = false;
    type.is_scalar = false;
    type.is_complex = false;
    type.is_signed = false;
    type.size = 0;
    type.str = name;

    if (name.size() < 2) {
        throw std::string("name too short to be a datatype");
    }

    // is it a scalar?
    if (name[0] == 's') {
        type.is_scalar = true;
        name = name.substr(1, name.size() - 1);
    }

    // get the data size
    size_t last_size_pos = name.find_last_of("0123456789");
    if (last_size_pos == std::string::npos) {
        throw std::string("no size spec in type ").append(name);
    }
    // will throw if malformed
    int size = volk_lexical_cast<int>(name.substr(0, last_size_pos + 1));

    assert(((size % 8) == 0) && (size <= 64) && (size != 0));
    type.size = size / 8; // in bytes

    for (size_t i = last_size_pos + 1; i < name.size(); i++) {
        switch (name[i]) {
        case 'f':
            type.is_float = true;
            break;
        case 'i':
            type.is_signed = true;
            break;
        case 'c':
            type.is_complex = true;
            break;
        case 'u':
            type.is_signed = false;
            break;
        default:
            throw std::string("Error: no such type: '") + name[i] + "'";
        }
    }

    return type;
}

std::vector<std::string> split_signature(const std::string& protokernel_signature)
{
    std::vector<std::string> signature_tokens;
    std::string token;
    for (unsigned int loc = 0; loc < protokernel_signature.size(); ++loc) {
        if (protokernel_signature.at(loc) == '_') {
            // this is a break
            signature_tokens.push_back(token);
            token = "";
        } else {
            token.push_back(protokernel_signature.at(loc));
        }
    }
    // Get the last one to the end of the string
    signature_tokens.push_back(token);
    return signature_tokens;
}

static void get_signatures_from_name(std::vector<volk_type_t>& inputsig,
                                     std::vector<volk_type_t>& outputsig,
                                     std::string name)
{

    std::vector<std::string> toked = split_signature(name);

    assert(toked[0] == "volk");
    toked.erase(toked.begin());

    // ok. we're assuming a string in the form
    //(sig)_(multiplier-opt)_..._(name)_(sig)_(multiplier-opt)_..._(alignment)

    enum { SIDE_INPUT, SIDE_NAME, SIDE_OUTPUT } side = SIDE_INPUT;
    std::string fn_name;
    volk_type_t type;
    for (unsigned int token_index = 0; token_index < toked.size(); ++token_index) {
        std::string token = toked[token_index];
        try {
            type = volk_type_from_string(token);
            if (side == SIDE_NAME)
                side = SIDE_OUTPUT; // if this is the first one after the name...

            if (side == SIDE_INPUT)
                inputsig.push_back(type);
            else
                outputsig.push_back(type);
        } catch (...) {
            if (token[0] == 'x' && (token.size() > 1) &&
                (token[1] > '0' && token[1] < '9')) { // it's a multiplier
                if (side == SIDE_INPUT)
                    assert(inputsig.size() > 0);
                else
                    assert(outputsig.size() > 0);
                int multiplier = volk_lexical_cast<int>(
                    token.substr(1, token.size() - 1)); // will throw if invalid
                for (int i = 1; i < multiplier; i++) {
                    if (side == SIDE_INPUT)
                        inputsig.push_back(inputsig.back());
                    else
                        outputsig.push_back(outputsig.back());
                }
            } else if (side ==
                       SIDE_INPUT) { // it's the function name, at least it better be
                side = SIDE_NAME;
                fn_name.append("_");
                fn_name.append(token);
            } else if (side == SIDE_OUTPUT) {
                if (token != toked.back())
                    throw; // the last token in the name is the alignment
            }
        }
    }
    // we don't need an output signature (some fn's operate on the input data, "in
    // place"), but we do need at least one input!
    assert(inputsig.size() != 0);
}

inline void run_cast_test1(volk_fn_1arg func,
                           std::vector<void*>& buffs,
                           unsigned int vlen,
                           unsigned int iter,
                           std::string arch)
{
    while (iter--)
        func(buffs[0], vlen, arch.c_str());
}

inline void run_cast_test2(volk_fn_2arg func,
                           std::vector<void*>& buffs,
                           unsigned int vlen,
                           unsigned int iter,
                           std::string arch)
{
    while (iter--)
        func(buffs[0], buffs[1], vlen, arch.c_str());
}

inline void run_cast_test3(volk_fn_3arg func,
                           std::vector<void*>& buffs,
                           unsigned int vlen,
                           unsigned int iter,
                           std::string arch)
{
    while (iter--)
        func(buffs[0], buffs[1], buffs[2], vlen, arch.c_str());
}

inline void run_cast_test4(volk_fn_4arg func,
                           std::vector<void*>& buffs,
                           unsigned int vlen,
                           unsigned int iter,
                           std::string arch)
{
    while (iter--)
        func(buffs[0], buffs[1], buffs[2], buffs[3], vlen, arch.c_str());
}

inline void run_cast_test1_s32f(volk_fn_1arg_s32f func,
                                std::vector<void*>& buffs,
                                float scalar,
                                unsigned int vlen,
                                unsigned int iter,
                                std::string arch)
{
    while (iter--)
        func(buffs[0], scalar, vlen, arch.c_str());
}

inline void run_cast_test2_s32f(volk_fn_2arg_s32f func,
                                std::vector<void*>& buffs,
                                float scalar,
                                unsigned int vlen,
                                unsigned int iter,
                                std::string arch)
{
    while (iter--)
        func(buffs[0], buffs[1], scalar, vlen, arch.c_str());
}

inline void run_cast_test3_s32f(volk_fn_3arg_s32f func,
                                std::vector<void*>& buffs,
                                float scalar,
                                unsigned int vlen,
                                unsigned int iter,
                                std::string arch)
{
    while (iter--)
        func(buffs[0], buffs[1], buffs[2], scalar, vlen, arch.c_str());
}

inline void run_cast_test1_s32fc(volk_fn_1arg_s32fc func,
                                 std::vector<void*>& buffs,
                                 lv_32fc_t scalar,
                                 unsigned int vlen,
                                 unsigned int iter,
                                 std::string arch)
{
    while (iter--)
        func(buffs[0], &scalar, vlen, arch.c_str());
}

inline void run_cast_test2_s32fc(volk_fn_2arg_s32fc func,
                                 std::vector<void*>& buffs,
                                 lv_32fc_t scalar,
                                 unsigned int vlen,
                                 unsigned int iter,
                                 std::string arch)
{
    while (iter--)
        func(buffs[0], buffs[1], &scalar, vlen, arch.c_str());
}

inline void run_cast_test3_s32fc(volk_fn_3arg_s32fc func,
                                 std::vector<void*>& buffs,
                                 lv_32fc_t scalar,
                                 unsigned int vlen,
                                 unsigned int iter,
                                 std::string arch)
{
    while (iter--)
        func(buffs[0], buffs[1], buffs[2], &scalar, vlen, arch.c_str());
}

template <class t>
bool fcompare(t* expected,
              t* actual,
              unsigned int vlen,
              float tol,
              bool absolute_mode,
              std::vector<unsigned int>& fail_indices,
              double& max_err)
{
    bool fail = false;
    max_err = 0.0;
    for (unsigned int i = 0; i < vlen; i++) {
        t exp_val = expected[i];
        t act_val = actual[i];

        // Check for special values (NaN, inf)
        bool exp_special = std::isnan(exp_val) || std::isinf(exp_val);
        bool act_special = std::isnan(act_val) || std::isinf(act_val);

        bool this_fail = false;
        double cmp_err = 0.0; // The error metric compared against tol
        if (exp_special || act_special) {
            // For NaN: both must be NaN (NaN != NaN, so use isnan)
            // For inf: both must be same signed infinity
            bool values_match =
                (std::isnan(exp_val) && std::isnan(act_val)) || (exp_val == act_val);
            if (!values_match) {
                this_fail = true;
                cmp_err = std::numeric_limits<double>::infinity();
            }
        } else if (absolute_mode) {
            cmp_err = fabs(exp_val - act_val);
            if (cmp_err > tol) {
                this_fail = true;
            }
        } else {
            // for very small numbers we'll see round off errors due to limited
            // precision. So a special test case...
            if (fabs(exp_val) < 1e-30) {
                cmp_err = fabs(act_val);
                if (cmp_err > tol) {
                    this_fail = true;
                }
            }
            // the primary test is the percent different greater than given tol
            else {
                cmp_err = fabs(exp_val - act_val) / fabs(exp_val);
                if (cmp_err > tol) {
                    this_fail = true;
                }
            }
        }
        if (cmp_err > max_err) {
            max_err = cmp_err;
        }
        if (this_fail) {
            fail = true;
            fail_indices.push_back(i);
        }
    }

    return fail;
}

template <class t>
bool ccompare(t* expected,
              t* actual,
              unsigned int vlen,
              float tol,
              bool absolute_mode,
              std::vector<unsigned int>& fail_indices,
              double& max_err)
{
    bool fail = false;
    max_err = 0.0;
    for (unsigned int i = 0; i < 2 * vlen; i += 2) {
        t exp_re = expected[i];
        t exp_im = expected[i + 1];
        t act_re = actual[i];
        t act_im = actual[i + 1];

        // Check for special values (NaN, inf) and verify they match
        bool exp_has_special = std::isnan(exp_re) || std::isnan(exp_im) ||
                               std::isinf(exp_re) || std::isinf(exp_im);
        bool act_has_special = std::isnan(act_re) || std::isnan(act_im) ||
                               std::isinf(act_re) || std::isinf(act_im);

        bool this_fail = false;
        double cmp_err = 0.0; // The error metric compared against tol
        if (exp_has_special || act_has_special) {
            // For NaN: both must be NaN (NaN != NaN, so use isnan)
            // For inf: both must be same signed infinity
            bool real_match =
                (std::isnan(exp_re) && std::isnan(act_re)) || (exp_re == act_re);
            bool imag_match =
                (std::isnan(exp_im) && std::isnan(act_im)) || (exp_im == act_im);

            if (!real_match || !imag_match) {
                this_fail = true;
                cmp_err = std::numeric_limits<double>::infinity();
            }
        } else {
            t diff[2] = { exp_re - act_re, exp_im - act_im };
            t err = std::sqrt(diff[0] * diff[0] + diff[1] * diff[1]);
            t norm = std::sqrt(exp_re * exp_re + exp_im * exp_im);

            if (absolute_mode) {
                cmp_err = err;
                if (cmp_err > tol) {
                    this_fail = true;
                }
            } else {
                // for very small numbers we'll see round off errors due to limited
                // precision. So a special test case...
                if (norm < 1e-30) {
                    cmp_err = err;
                    if (cmp_err > tol) {
                        this_fail = true;
                    }
                }
                // the primary test is the percent different greater than given tol
                else {
                    cmp_err = err / norm;
                    if (cmp_err > tol) {
                        this_fail = true;
                    }
                }
            }
        }
        if (cmp_err > max_err) {
            max_err = cmp_err;
        }
        if (this_fail) {
            fail = true;
            fail_indices.push_back(i / 2);
        }
    }

    return fail;
}

template <class t>
bool icompare(t* expected,
              t* actual,
              unsigned int vlen,
              unsigned int tol,
              std::vector<unsigned int>& fail_indices,
              double& max_err)
{
    bool fail = false;
    max_err = 0.0;
    for (unsigned int i = 0; i < vlen; i++) {
        t exp_val = expected[i];
        t act_val = actual[i];
        uint64_t abs_err = (uint64_t)abs(int64_t(exp_val) - int64_t(act_val));
        if ((double)abs_err > max_err) {
            max_err = (double)abs_err;
        }
        if (abs_err > tol) {
            fail = true;
            fail_indices.push_back(i);
        }
    }

    return fail;
}

// Print error table for failed comparisons
// Shows: index, input(s), expected, actual, rel_error, tol
void print_error_table(const std::vector<unsigned int>& fail_indices,
                       const std::vector<void*>& inputs,
                       const std::vector<volk_type_t>& input_sigs,
                       void* expected,
                       void* actual,
                       const volk_type_t& output_sig,
                       float tol,
                       int max_errors = 10)
{
    if (fail_indices.empty())
        return;

    // Print header
    fmt::print("{:>7}", "index");
    for (size_t k = 0; k < input_sigs.size(); k++) {
        fmt::print(" | {:>10}", fmt::format("in{}", k));
    }
    fmt::print(
        " | {:>10} | {:>10} | {:>9} | {:>9}\n", "expected", "actual", "rel_err", "tol");

    // Print separator
    fmt::print("{:-<7}", "");
    for (size_t k = 0; k < input_sigs.size(); k++) {
        fmt::print("-+-{:-<10}", "");
    }
    fmt::print("-+-{:-<10}-+-{:-<10}-+-{:-<9}-+-{:-<9}\n", "", "", "", "");

    int print_count = 0;
    for (unsigned int idx : fail_indices) {
        if (print_count++ >= max_errors) {
            fmt::print("... and {} more errors\n", fail_indices.size() - max_errors);
            break;
        }

        fmt::print("{:>7}", idx);

        // Print input values
        for (size_t k = 0; k < input_sigs.size(); k++) {
            if (input_sigs[k].is_float) {
                double val = (input_sigs[k].size == 8) ? ((double*)inputs[k])[idx]
                                                       : ((float*)inputs[k])[idx];
                fmt::print(" | {:>10.4f}", val);
            } else {
                int64_t val = 0;
                switch (input_sigs[k].size) {
                case 8:
                    val = input_sigs[k].is_signed ? ((int64_t*)inputs[k])[idx]
                                                  : (int64_t)((uint64_t*)inputs[k])[idx];
                    break;
                case 4:
                    val = input_sigs[k].is_signed ? ((int32_t*)inputs[k])[idx]
                                                  : (int64_t)((uint32_t*)inputs[k])[idx];
                    break;
                case 2:
                    val = input_sigs[k].is_signed ? ((int16_t*)inputs[k])[idx]
                                                  : (int64_t)((uint16_t*)inputs[k])[idx];
                    break;
                case 1:
                    val = input_sigs[k].is_signed ? ((int8_t*)inputs[k])[idx]
                                                  : (int64_t)((uint8_t*)inputs[k])[idx];
                    break;
                }
                fmt::print(" | {:>10}", val);
            }
        }

        // Get expected and actual values, compute relative error
        double exp_val = 0, act_val = 0, rel_err = 0;
        if (output_sig.is_float) {
            if (output_sig.size == 8) {
                exp_val = ((double*)expected)[idx];
                act_val = ((double*)actual)[idx];
            } else {
                exp_val = ((float*)expected)[idx];
                act_val = ((float*)actual)[idx];
            }
            double abs_err = fabs(exp_val - act_val);
            rel_err = (fabs(exp_val) > 1e-30) ? abs_err / fabs(exp_val) : abs_err;
            fmt::print(" | {:>10.4f} | {:>10.4f}", exp_val, act_val);
        } else {
            int64_t exp_i = 0, act_i = 0;
            switch (output_sig.size) {
            case 8:
                exp_i = output_sig.is_signed ? ((int64_t*)expected)[idx]
                                             : (int64_t)((uint64_t*)expected)[idx];
                act_i = output_sig.is_signed ? ((int64_t*)actual)[idx]
                                             : (int64_t)((uint64_t*)actual)[idx];
                break;
            case 4:
                exp_i = output_sig.is_signed ? ((int32_t*)expected)[idx]
                                             : (int64_t)((uint32_t*)expected)[idx];
                act_i = output_sig.is_signed ? ((int32_t*)actual)[idx]
                                             : (int64_t)((uint32_t*)actual)[idx];
                break;
            case 2:
                exp_i = output_sig.is_signed ? ((int16_t*)expected)[idx]
                                             : (int64_t)((uint16_t*)expected)[idx];
                act_i = output_sig.is_signed ? ((int16_t*)actual)[idx]
                                             : (int64_t)((uint16_t*)actual)[idx];
                break;
            case 1:
                exp_i = output_sig.is_signed ? ((int8_t*)expected)[idx]
                                             : (int64_t)((uint8_t*)expected)[idx];
                act_i = output_sig.is_signed ? ((int8_t*)actual)[idx]
                                             : (int64_t)((uint8_t*)actual)[idx];
                break;
            }
            fmt::print(" | {:>10} | {:>10}", exp_i, act_i);
            double abs_err = (double)abs(exp_i - act_i);
            rel_err = (exp_i != 0) ? abs_err / fabs((double)exp_i) : abs_err;
        }

        fmt::print(" | {:>9.1e} | {:>9.1e}\n", rel_err, (double)tol);
    }
}

// Structure to hold failure info for deferred printing
struct fail_info_t {
    std::string arch_name;
    std::vector<unsigned int> fail_indices;
    size_t output_idx;
    size_t arch_index;
    double max_err;
};

class volk_qa_aligned_mem_pool
{
public:
    void* get_new(size_t size)
    {
        size_t alignment = volk_get_alignment();
        void* ptr = volk_malloc(size, alignment);
        memset(ptr, 0x00, size);
        _mems.push_back(ptr);
        return ptr;
    }
    ~volk_qa_aligned_mem_pool()
    {
        for (unsigned int ii = 0; ii < _mems.size(); ++ii) {
            volk_free(_mems[ii]);
        }
    }

private:
    std::vector<void*> _mems;
};

bool run_volk_tests(volk_func_desc_t desc,
                    void (*manual_func)(),
                    std::string name,
                    volk_test_params_t test_params,
                    std::vector<volk_test_results_t>* results,
                    std::string puppet_master_name)
{
    return run_volk_tests(desc,
                          manual_func,
                          name,
                          test_params.tol(),
                          test_params.scalar(),
                          test_params.vlen(),
                          test_params.iter(),
                          results,
                          puppet_master_name,
                          test_params.absolute_mode(),
                          test_params.benchmark_mode(),
                          test_params.float_edge_cases(),
                          test_params.complex_edge_cases());
}

// Shared setup for run_volk_tests and run_volk_reference_test (#88): build the
// arch list, parse the kernel signature, generate ONE set of random inputs, and
// build the per-arch buffer copies. `vlen` is the already-twiddled buffer length
// (the caller owns the twiddle). Emits the same diagnostics and returns
// ok=false on <2 archs (non-benchmark) or an unparseable signature. `mem_pool`
// must outlive the returned buffers. This is pure setup (no kernel execution and
// no comparison), so run_volk_tests' observable behaviour is unchanged.
struct qa_test_data {
    std::vector<std::string> arch_list;
    std::map<std::string, size_t> arch_to_orig_idx;
    std::vector<volk_type_t> inputsig;
    std::vector<volk_type_t> outputsig;
    std::vector<volk_type_t> inputsc;
    std::vector<volk_type_t> both_sigs;
    std::vector<void*> inbuffs;
    std::vector<std::vector<void*>> test_data;
    bool ok = true;
};

static qa_test_data setup_test_data(volk_func_desc_t desc,
                                    const std::string& name,
                                    unsigned int vlen,
                                    bool benchmark_mode,
                                    const std::vector<float>& float_edge_cases,
                                    const std::vector<lv_32fc_t>& complex_edge_cases,
                                    volk_qa_aligned_mem_pool& mem_pool)
{
    qa_test_data d;

    // first let's get a list of available architectures for the test
    d.arch_list = get_arch_list(desc);

    // Build map from arch name to original index (for impl_alignment lookup)
    for (size_t i = 0; i < d.arch_list.size(); i++) {
        d.arch_to_orig_idx[d.arch_list[i]] = i;
    }

    // Reorder arch_list to put generic implementations first for consistent output
    // Priority: "generic" first, then other generic_* variants, then everything else
    std::vector<std::string> plain_generic;
    std::vector<std::string> other_generic_impls;
    std::vector<std::string> other_impls;
    for (const auto& arch : d.arch_list) {
        if (arch == "generic") {
            plain_generic.push_back(arch);
        } else if (arch.find("generic") == 0) { // starts with "generic"
            other_generic_impls.push_back(arch);
        } else {
            other_impls.push_back(arch);
        }
    }
    d.arch_list.clear();
    d.arch_list.insert(d.arch_list.end(), plain_generic.begin(), plain_generic.end());
    d.arch_list.insert(
        d.arch_list.end(), other_generic_impls.begin(), other_generic_impls.end());
    d.arch_list.insert(d.arch_list.end(), other_impls.begin(), other_impls.end());

    if ((!benchmark_mode) && (d.arch_list.size() < 2)) {
        std::cerr << "no architectures to test" << std::endl;
        d.ok = false;
        return d;
    }

    // now we have to get a function signature by parsing the name
    try {
        get_signatures_from_name(d.inputsig, d.outputsig, name);
    } catch (std::exception& error) {
        std::cerr << "Error: unable to get function signature from kernel name"
                  << std::endl;
        std::cerr << "  - " << name << std::endl;
        d.ok = false;
        return d;
    }

    // pull the input scalars into their own vector
    for (size_t i = 0; i < d.inputsig.size(); i++) {
        if (d.inputsig[i].is_scalar) {
            d.inputsc.push_back(d.inputsig[i]);
            d.inputsig.erase(d.inputsig.begin() + i);
            i -= 1;
        }
    }
    for (unsigned int inputsig_index = 0; inputsig_index < d.inputsig.size();
         ++inputsig_index) {
        volk_type_t sig = d.inputsig[inputsig_index];
        if (!sig.is_scalar) // we don't make buffers for scalars
            d.inbuffs.push_back(
                mem_pool.get_new(vlen * sig.size * (sig.is_complex ? 2 : 1)));
    }
    for (size_t i = 0; i < d.inbuffs.size(); i++) {
        load_random_data(
            d.inbuffs[i], d.inputsig[i], vlen, float_edge_cases, complex_edge_cases);
    }

    // ok let's make a vector of vector of void buffers, which holds the input/output
    // vectors for each arch
    for (size_t i = 0; i < d.arch_list.size(); i++) {
        std::vector<void*> arch_buffs;
        for (size_t j = 0; j < d.outputsig.size(); j++) {
            arch_buffs.push_back(mem_pool.get_new(vlen * d.outputsig[j].size *
                                                  (d.outputsig[j].is_complex ? 2 : 1)));
        }
        for (size_t j = 0; j < d.inputsig.size(); j++) {
            void* arch_inbuff = mem_pool.get_new(vlen * d.inputsig[j].size *
                                                 (d.inputsig[j].is_complex ? 2 : 1));
            memcpy(arch_inbuff,
                   d.inbuffs[j],
                   vlen * d.inputsig[j].size * (d.inputsig[j].is_complex ? 2 : 1));
            arch_buffs.push_back(arch_inbuff);
        }
        d.test_data.push_back(arch_buffs);
    }

    d.both_sigs.insert(d.both_sigs.end(), d.outputsig.begin(), d.outputsig.end());
    d.both_sigs.insert(d.both_sigs.end(), d.inputsig.begin(), d.inputsig.end());

    return d;
}

bool run_volk_tests(volk_func_desc_t desc,
                    void (*manual_func)(),
                    std::string name,
                    float tol,
                    lv_32fc_t scalar,
                    unsigned int vlen,
                    unsigned int iter,
                    std::vector<volk_test_results_t>* results,
                    std::string puppet_master_name,
                    bool absolute_mode,
                    bool benchmark_mode,
                    const std::vector<float>& float_edge_cases,
                    const std::vector<lv_32fc_t>& complex_edge_cases)
{
    // Initialize this entry in results vector
    results->push_back(volk_test_results_t());
    results->back().name = name;
    results->back().vlen = vlen;
    results->back().iter = iter;
    fmt::print(
        "\nRUN_VOLK_TESTS: {}(vlen={}, iter={}, tol={:.0e})\n", name, vlen, iter, tol);

    // vlen_twiddle will increase vlen for malloc and data generation
    // but kernels will still be called with the user provided vlen.
    // This is useful for causing errors in kernels that do bad reads
    const unsigned int vlen_twiddle = 5;
    vlen = vlen + vlen_twiddle;

    const float tol_f = tol;
    const unsigned int tol_i = static_cast<unsigned int>(tol);

    // something that can hang onto memory and cleanup when this function exits
    volk_qa_aligned_mem_pool mem_pool;

    // Build arch list, parse the signature, generate inputs, and per-arch buffer
    // copies (shared with run_volk_reference_test; #88). `vlen` is twiddled here.
    qa_test_data test_setup = setup_test_data(
        desc, name, vlen, benchmark_mode, float_edge_cases, complex_edge_cases, mem_pool);
    if (!test_setup.ok) {
        return false;
    }
    std::vector<std::string>& arch_list = test_setup.arch_list;
    std::map<std::string, size_t>& arch_to_orig_idx = test_setup.arch_to_orig_idx;
    std::vector<volk_type_t>& inputsig = test_setup.inputsig;
    std::vector<volk_type_t>& outputsig = test_setup.outputsig;
    std::vector<volk_type_t>& inputsc = test_setup.inputsc;
    std::vector<volk_type_t>& both_sigs = test_setup.both_sigs;
    std::vector<void*>& inbuffs = test_setup.inbuffs;
    std::vector<std::vector<void*>>& test_data = test_setup.test_data;

    // now run the test
    vlen = vlen - vlen_twiddle;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::vector<double> profile_times;

    // Warmup to let CPU reach full turbo frequency (only for first kernel)
    const double warmup_target_ms = g_warmup_done ? 0.0 : volk_test_get_warmup_ms();
    {
        // Run a quick test to estimate time per iteration
        start = std::chrono::system_clock::now();
        switch (both_sigs.size()) {
        case 1:
            if (inputsc.size() == 0) {
                run_cast_test1(
                    (volk_fn_1arg)(manual_func), test_data[0], vlen, iter, "generic");
            } else if (inputsc.size() == 1 && inputsc[0].is_float) {
                if (inputsc[0].is_complex) {
                    run_cast_test1_s32fc((volk_fn_1arg_s32fc)(manual_func),
                                         test_data[0],
                                         scalar,
                                         vlen,
                                         iter,
                                         "generic");
                } else {
                    run_cast_test1_s32f((volk_fn_1arg_s32f)(manual_func),
                                        test_data[0],
                                        scalar.real(),
                                        vlen,
                                        iter,
                                        "generic");
                }
            }
            break;
        case 2:
            if (inputsc.size() == 0) {
                run_cast_test2(
                    (volk_fn_2arg)(manual_func), test_data[0], vlen, iter, "generic");
            } else if (inputsc.size() == 1 && inputsc[0].is_float) {
                if (inputsc[0].is_complex) {
                    run_cast_test2_s32fc((volk_fn_2arg_s32fc)(manual_func),
                                         test_data[0],
                                         scalar,
                                         vlen,
                                         iter,
                                         "generic");
                } else {
                    run_cast_test2_s32f((volk_fn_2arg_s32f)(manual_func),
                                        test_data[0],
                                        scalar.real(),
                                        vlen,
                                        iter,
                                        "generic");
                }
            }
            break;
        case 3:
            if (inputsc.size() == 0) {
                run_cast_test3(
                    (volk_fn_3arg)(manual_func), test_data[0], vlen, iter, "generic");
            } else if (inputsc.size() == 1 && inputsc[0].is_float) {
                if (inputsc[0].is_complex) {
                    run_cast_test3_s32fc((volk_fn_3arg_s32fc)(manual_func),
                                         test_data[0],
                                         scalar,
                                         vlen,
                                         iter,
                                         "generic");
                } else {
                    run_cast_test3_s32f((volk_fn_3arg_s32f)(manual_func),
                                        test_data[0],
                                        scalar.real(),
                                        vlen,
                                        iter,
                                        "generic");
                }
            }
            break;
        case 4:
            run_cast_test4(
                (volk_fn_4arg)(manual_func), test_data[0], vlen, iter, "generic");
            break;
        default:
            break;
        }
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double test_time_ms = 1000.0 * elapsed.count();

        // If we haven't reached 500ms yet, calculate how many more iterations we need
        if (test_time_ms < warmup_target_ms) {
            double remaining_ms = warmup_target_ms - test_time_ms;
            unsigned int warmup_iterations =
                (unsigned int)((remaining_ms / test_time_ms) * iter);
            if (warmup_iterations > 0) {
                // Run additional warmup iterations
                switch (both_sigs.size()) {
                case 1:
                    if (inputsc.size() == 0) {
                        run_cast_test1((volk_fn_1arg)(manual_func),
                                       test_data[0],
                                       vlen,
                                       warmup_iterations,
                                       "generic");
                    } else if (inputsc.size() == 1 && inputsc[0].is_float) {
                        if (inputsc[0].is_complex) {
                            run_cast_test1_s32fc((volk_fn_1arg_s32fc)(manual_func),
                                                 test_data[0],
                                                 scalar,
                                                 vlen,
                                                 warmup_iterations,
                                                 "generic");
                        } else {
                            run_cast_test1_s32f((volk_fn_1arg_s32f)(manual_func),
                                                test_data[0],
                                                scalar.real(),
                                                vlen,
                                                warmup_iterations,
                                                "generic");
                        }
                    }
                    break;
                case 2:
                    if (inputsc.size() == 0) {
                        run_cast_test2((volk_fn_2arg)(manual_func),
                                       test_data[0],
                                       vlen,
                                       warmup_iterations,
                                       "generic");
                    } else if (inputsc.size() == 1 && inputsc[0].is_float) {
                        if (inputsc[0].is_complex) {
                            run_cast_test2_s32fc((volk_fn_2arg_s32fc)(manual_func),
                                                 test_data[0],
                                                 scalar,
                                                 vlen,
                                                 warmup_iterations,
                                                 "generic");
                        } else {
                            run_cast_test2_s32f((volk_fn_2arg_s32f)(manual_func),
                                                test_data[0],
                                                scalar.real(),
                                                vlen,
                                                warmup_iterations,
                                                "generic");
                        }
                    }
                    break;
                case 3:
                    if (inputsc.size() == 0) {
                        run_cast_test3((volk_fn_3arg)(manual_func),
                                       test_data[0],
                                       vlen,
                                       warmup_iterations,
                                       "generic");
                    } else if (inputsc.size() == 1 && inputsc[0].is_float) {
                        if (inputsc[0].is_complex) {
                            run_cast_test3_s32fc((volk_fn_3arg_s32fc)(manual_func),
                                                 test_data[0],
                                                 scalar,
                                                 vlen,
                                                 warmup_iterations,
                                                 "generic");
                        } else {
                            run_cast_test3_s32f((volk_fn_3arg_s32f)(manual_func),
                                                test_data[0],
                                                scalar.real(),
                                                vlen,
                                                warmup_iterations,
                                                "generic");
                        }
                    }
                    break;
                case 4:
                    run_cast_test4((volk_fn_4arg)(manual_func),
                                   test_data[0],
                                   vlen,
                                   warmup_iterations,
                                   "generic");
                    break;
                default:
                    break;
                }
            }
        }
        g_warmup_done = true;
    }

    // Reset all test buffers after warmup
    for (size_t i = 0; i < arch_list.size(); i++) {
        for (size_t j = 0; j < outputsig.size(); j++) {
            memset(test_data[i][j],
                   0,
                   vlen * outputsig[j].size * (outputsig[j].is_complex ? 2 : 1));
        }
        // Reload input buffers from original data
        for (size_t j = 0; j < inputsig.size(); j++) {
            memcpy(test_data[i][outputsig.size() + j],
                   inbuffs[j],
                   vlen * inputsig[j].size * (inputsig[j].is_complex ? 2 : 1));
        }
    }

    for (size_t i = 0; i < arch_list.size(); i++) {
        start = std::chrono::system_clock::now();

        switch (both_sigs.size()) {
        case 1:
            if (inputsc.size() == 0) {
                run_cast_test1(
                    (volk_fn_1arg)(manual_func), test_data[i], vlen, iter, arch_list[i]);
            } else if (inputsc.size() == 1 && inputsc[0].is_float) {
                if (inputsc[0].is_complex) {
                    run_cast_test1_s32fc((volk_fn_1arg_s32fc)(manual_func),
                                         test_data[i],
                                         scalar,
                                         vlen,
                                         iter,
                                         arch_list[i]);
                } else {
                    run_cast_test1_s32f((volk_fn_1arg_s32f)(manual_func),
                                        test_data[i],
                                        scalar.real(),
                                        vlen,
                                        iter,
                                        arch_list[i]);
                }
            } else
                throw "unsupported 1 arg function >1 scalars";
            break;
        case 2:
            if (inputsc.size() == 0) {
                run_cast_test2(
                    (volk_fn_2arg)(manual_func), test_data[i], vlen, iter, arch_list[i]);
            } else if (inputsc.size() == 1 && inputsc[0].is_float) {
                if (inputsc[0].is_complex) {
                    run_cast_test2_s32fc((volk_fn_2arg_s32fc)(manual_func),
                                         test_data[i],
                                         scalar,
                                         vlen,
                                         iter,
                                         arch_list[i]);
                } else {
                    run_cast_test2_s32f((volk_fn_2arg_s32f)(manual_func),
                                        test_data[i],
                                        scalar.real(),
                                        vlen,
                                        iter,
                                        arch_list[i]);
                }
            } else
                throw "unsupported 2 arg function >1 scalars";
            break;
        case 3:
            if (inputsc.size() == 0) {
                run_cast_test3(
                    (volk_fn_3arg)(manual_func), test_data[i], vlen, iter, arch_list[i]);
            } else if (inputsc.size() == 1 && inputsc[0].is_float) {
                if (inputsc[0].is_complex) {
                    run_cast_test3_s32fc((volk_fn_3arg_s32fc)(manual_func),
                                         test_data[i],
                                         scalar,
                                         vlen,
                                         iter,
                                         arch_list[i]);
                } else {
                    run_cast_test3_s32f((volk_fn_3arg_s32f)(manual_func),
                                        test_data[i],
                                        scalar.real(),
                                        vlen,
                                        iter,
                                        arch_list[i]);
                }
            } else
                throw "unsupported 3 arg function >1 scalars";
            break;
        case 4:
            run_cast_test4(
                (volk_fn_4arg)(manual_func), test_data[i], vlen, iter, arch_list[i]);
            break;
        default:
            throw "no function handler for this signature";
            break;
        }

        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        double arch_time = 1000.0 * elapsed_seconds.count();

        volk_test_time_t result;
        result.name = arch_list[i];
        result.time = arch_time;
        result.units = "ms";
        result.pass = true;
        results->back().results[result.name] = result;

        profile_times.push_back(arch_time);
    }

    // and now compare each output to the generic output
    // first we have to know which output is the generic one, they aren't in order...
    size_t generic_offset = 0;
    for (size_t i = 0; i < arch_list.size(); i++) {
        if (arch_list[i] == "generic") {
            generic_offset = i;
        }
    }

    // Just in case a kernel wrote to OOB memory, use the twiddled vlen
    vlen = vlen + vlen_twiddle;
    bool fail;
    bool fail_global = false;
    std::vector<bool> arch_results;

    // Collect input buffers for error reporting (inputs are after outputs in test_data)
    std::vector<void*> input_buffs;
    for (size_t k = outputsig.size(); k < both_sigs.size(); k++) {
        input_buffs.push_back(test_data[generic_offset][k]);
    }

    // Collect failures for deferred printing (after timing summary)
    std::vector<fail_info_t> failures;

    // Track max error per architecture (absolute or relative depending on mode)
    std::vector<double> arch_max_err(arch_list.size(), 0.0);

    for (size_t i = 0; i < arch_list.size(); i++) {
        fail = false;
        if (i != generic_offset) {
            for (size_t j = 0; j < outputsig.size(); j++) {
                std::vector<unsigned int> fail_indices;
                double max_err = 0.0;
                if (both_sigs[j].is_float) {
                    if (both_sigs[j].size == 8) {
                        if (both_sigs[j].is_complex) {
                            fail = ccompare((double*)test_data[generic_offset][j],
                                            (double*)test_data[i][j],
                                            vlen,
                                            tol_f,
                                            absolute_mode,
                                            fail_indices,
                                            max_err);
                        } else {
                            fail = fcompare((double*)test_data[generic_offset][j],
                                            (double*)test_data[i][j],
                                            vlen,
                                            tol_f,
                                            absolute_mode,
                                            fail_indices,
                                            max_err);
                        }
                    } else {
                        if (both_sigs[j].is_complex) {
                            fail = ccompare((float*)test_data[generic_offset][j],
                                            (float*)test_data[i][j],
                                            vlen,
                                            tol_f,
                                            absolute_mode,
                                            fail_indices,
                                            max_err);
                        } else {
                            fail = fcompare((float*)test_data[generic_offset][j],
                                            (float*)test_data[i][j],
                                            vlen,
                                            tol_f,
                                            absolute_mode,
                                            fail_indices,
                                            max_err);
                        }
                    }
                } else {
                    // i could replace this whole switch statement with a memcmp if i
                    // wasn't interested in printing the outputs where they differ
                    switch (both_sigs[j].size) {
                    case 8:
                        if (both_sigs[j].is_signed) {
                            fail = icompare((int64_t*)test_data[generic_offset][j],
                                            (int64_t*)test_data[i][j],
                                            vlen * (both_sigs[j].is_complex ? 2 : 1),
                                            tol_i,
                                            fail_indices,
                                            max_err);
                        } else {
                            fail = icompare((uint64_t*)test_data[generic_offset][j],
                                            (uint64_t*)test_data[i][j],
                                            vlen * (both_sigs[j].is_complex ? 2 : 1),
                                            tol_i,
                                            fail_indices,
                                            max_err);
                        }
                        break;
                    case 4:
                        if (both_sigs[j].is_complex) {
                            if (both_sigs[j].is_signed) {
                                fail = icompare((int16_t*)test_data[generic_offset][j],
                                                (int16_t*)test_data[i][j],
                                                vlen * (both_sigs[j].is_complex ? 2 : 1),
                                                tol_i,
                                                fail_indices,
                                                max_err);
                            } else {
                                fail = icompare((uint16_t*)test_data[generic_offset][j],
                                                (uint16_t*)test_data[i][j],
                                                vlen * (both_sigs[j].is_complex ? 2 : 1),
                                                tol_i,
                                                fail_indices,
                                                max_err);
                            }
                        } else {
                            if (both_sigs[j].is_signed) {
                                fail = icompare((int32_t*)test_data[generic_offset][j],
                                                (int32_t*)test_data[i][j],
                                                vlen * (both_sigs[j].is_complex ? 2 : 1),
                                                tol_i,
                                                fail_indices,
                                                max_err);
                            } else {
                                fail = icompare((uint32_t*)test_data[generic_offset][j],
                                                (uint32_t*)test_data[i][j],
                                                vlen * (both_sigs[j].is_complex ? 2 : 1),
                                                tol_i,
                                                fail_indices,
                                                max_err);
                            }
                        }
                        break;
                    case 2:
                        if (both_sigs[j].is_signed) {
                            fail = icompare((int16_t*)test_data[generic_offset][j],
                                            (int16_t*)test_data[i][j],
                                            vlen * (both_sigs[j].is_complex ? 2 : 1),
                                            tol_i,
                                            fail_indices,
                                            max_err);
                        } else {
                            fail = icompare((uint16_t*)test_data[generic_offset][j],
                                            (uint16_t*)test_data[i][j],
                                            vlen * (both_sigs[j].is_complex ? 2 : 1),
                                            tol_i,
                                            fail_indices,
                                            max_err);
                        }
                        break;
                    case 1:
                        if (both_sigs[j].is_signed) {
                            fail = icompare((int8_t*)test_data[generic_offset][j],
                                            (int8_t*)test_data[i][j],
                                            vlen * (both_sigs[j].is_complex ? 2 : 1),
                                            tol_i,
                                            fail_indices,
                                            max_err);
                        } else {
                            fail = icompare((uint8_t*)test_data[generic_offset][j],
                                            (uint8_t*)test_data[i][j],
                                            vlen * (both_sigs[j].is_complex ? 2 : 1),
                                            tol_i,
                                            fail_indices,
                                            max_err);
                        }
                        break;
                    default:
                        fail = 1;
                    }
                }
                // Track max error for this arch across all outputs
                if (max_err > arch_max_err[i]) {
                    arch_max_err[i] = max_err;
                }
                if (fail) {
                    volk_test_time_t* result = &results->back().results[arch_list[i]];
                    result->pass = false;
                    fail_global = true;
                    // Store failure info for later printing
                    fail_info_t fi;
                    fi.max_err = max_err;
                    fi.arch_name = arch_list[i];
                    fi.fail_indices = fail_indices;
                    fi.output_idx = j;
                    fi.arch_index = i;
                    failures.push_back(fi);
                }
            }
        }
        arch_results.push_back(!fail);
    }

    double best_time_a = std::numeric_limits<double>::max();
    double best_time_u = std::numeric_limits<double>::max();
    std::string best_arch_a = "generic";
    std::string best_arch_u = "generic";
    for (size_t i = 0; i < arch_list.size(); i++) {
        // Look up alignment using original index (before reordering)
        size_t orig_idx = arch_to_orig_idx[arch_list[i]];
        bool requires_alignment = desc.impl_alignment[orig_idx];

        if ((profile_times[i] < best_time_u) && arch_results[i] && !requires_alignment) {
            best_time_u = profile_times[i];
            best_arch_u = arch_list[i];
        }
        if ((profile_times[i] < best_time_a) && arch_results[i]) {
            best_time_a = profile_times[i];
            best_arch_a = arch_list[i];
        }
    }

    // Unaligned implementations (alignment == 0) work on any memory alignment.
    // If an unaligned impl is faster than all aligned impls, use it for both.
    if (best_time_u < best_time_a) {
        best_time_a = best_time_u;
        best_arch_a = best_arch_u;
    }

    // Calculate total data transferred (bytes read + written) for throughput display
    size_t bytes_per_call = 0;
    for (size_t j = 0; j < outputsig.size(); j++) {
        bytes_per_call += outputsig[j].size * (outputsig[j].is_complex ? 2 : 1) * vlen;
    }
    for (size_t j = 0; j < inputsig.size(); j++) {
        bytes_per_call += inputsig[j].size * (inputsig[j].is_complex ? 2 : 1) * vlen;
    }
    double total_mb = (bytes_per_call * iter) / 1e6; // Total megabytes transferred

    // Get generic timing for speedup calculation
    double generic_time = 0.0;
    for (size_t i = 0; i < arch_list.size(); i++) {
        if (arch_list[i] == "generic") {
            generic_time = profile_times[i];
            break;
        }
    }

    // Column widths for results table
    constexpr int w_arch = 26;
    constexpr int w_time = 14;
    constexpr int w_tput = 14;
    constexpr int w_speedup = 8;
    constexpr int w_err = 10;

    // Column header depends on error mode
    // Integer outputs always use absolute comparison, so show "max_abs" for them
    bool has_int_output = false;
    for (const auto& sig : outputsig) {
        if (!sig.is_float) {
            has_int_output = true;
            break;
        }
    }
    const char* err_col = (absolute_mode || has_int_output) ? "max_abs" : "max_rel";

    // Helper for adaptive decimal places based on magnitude
    auto format_time = [](double ms) -> std::string {
        return fmt::format("{:.2f} ms", ms);
    };

    auto format_throughput = [](double mbps) -> std::string {
        return fmt::format("{:.1f} MB/s", mbps);
    };

    // Print table header
    fmt::print("{:<{}} | {:>{}} | {:>{}} | {:>{}} | {:>{}} |\n",
               "arch",
               w_arch,
               "time",
               w_time,
               "throughput",
               w_tput,
               "speedup",
               w_speedup,
               err_col,
               w_err);
    fmt::print("{:-<{}}-+-{:-<{}}-+-{:-<{}}-+-{:-<{}}-+-{:-<{}}-+\n",
               "",
               w_arch,
               "",
               w_time,
               "",
               w_tput,
               "",
               w_speedup,
               "",
               w_err);

    // Print each architecture row
    for (size_t i = 0; i < arch_list.size(); i++) {
        double time_seconds = profile_times[i] / 1000.0;
        double throughput_mbps = total_mb / time_seconds;

        std::string time_str = format_time(profile_times[i]);
        std::string tput_str = format_throughput(throughput_mbps);
        std::string speedup_str;
        if (arch_list[i] == "generic" || generic_time <= 0) {
            speedup_str = "-";
        } else {
            double speedup = generic_time / profile_times[i];
            speedup_str = fmt::format("{:.2f}x", speedup);
        }
        std::string err_str =
            (arch_list[i] == "generic") ? "-" : fmt::format("{:.1e}", arch_max_err[i]);
        std::string win_str =
            (arch_list[i] == best_arch_a || arch_list[i] == best_arch_u) ? " *" : "";

        fmt::print("{:<{}} | {:>{}} | {:>{}} | {:>{}} | {:>{}} |{}\n",
                   arch_list[i],
                   w_arch,
                   time_str,
                   w_time,
                   tput_str,
                   w_tput,
                   speedup_str,
                   w_speedup,
                   err_str,
                   w_err,
                   win_str);
    }

    // Print best arch summary (left-aligned, ":" at arch column width)
    auto print_best_line = [&](const char* label, const std::string& arch, double time) {
        std::string speedup_str;
        if (arch != "generic" && generic_time > 0) {
            speedup_str = fmt::format(" ({:.2f}x)", generic_time / time);
        }
        fmt::print("{:<{}} {}{}\n", label, w_arch, arch, speedup_str);
    };

    print_best_line("Best aligned arch          |", best_arch_a, best_time_a);
    print_best_line("Best unaligned arch        |", best_arch_u, best_time_u);

    // Print failure details after timing summary
    for (const auto& fi : failures) {
        fmt::print("\n{}: fail on arch {}\n", name, fi.arch_name);
        print_error_table(fi.fail_indices,
                          input_buffs,
                          inputsig,
                          test_data[generic_offset][fi.output_idx],
                          test_data[fi.arch_index][fi.output_idx],
                          outputsig[fi.output_idx],
                          tol_f);
    }

    fmt::print("{:-<88}\n", "");

    if (puppet_master_name == "NULL") {
        results->back().config_name = name;
    } else {
        results->back().config_name = puppet_master_name;
    }
    results->back().best_arch_a = best_arch_a;
    results->back().best_arch_u = best_arch_u;

    return fail_global;
}

// #88: run every impl (generic included) and compare each against an INDEPENDENT
// double-precision reference (the registry oracle), instead of impl-vs-generic.
// This catches defects all impls share. run_volk_tests is untouched; default qa
// is unaffected. Returns true if ANY impl diverges from the oracle past tol.
// Supported signatures: 1-3 buffers with an optional real (s32f) scalar and a
// float/complex-float output — covers the registered reference kernels.
bool run_volk_reference_test(volk_func_desc_t desc,
                             void (*manual_func)(),
                             std::string name,
                             const volk_reference_entry& ref,
                             lv_32fc_t scalar,
                             unsigned int vlen,
                             std::vector<volk_test_results_t>* results,
                             const std::vector<float>& float_edge_cases,
                             const std::vector<lv_32fc_t>& complex_edge_cases)
{
    results->push_back(volk_test_results_t());
    results->back().name = name;
    results->back().vlen = vlen;
    results->back().iter = 1;
    fmt::print("\nRUN_VOLK_REFERENCE_TEST: {}(vlen={}, tol={:.0e}, {})\n",
               name,
               vlen,
               ref.tol,
               ref.absolute ? "abs" : "rel");

    const unsigned int vlen_twiddle = 5;
    const unsigned int vlen_alloc = vlen + vlen_twiddle;

    volk_qa_aligned_mem_pool mem_pool;
    // Pass benchmark_mode=true to setup_test_data ONLY to bypass its "need >=2
    // archs" guard: reference mode compares a single impl (e.g. power_32fc's
    // generic-only) against the oracle — that single-impl case is exactly the
    // blind spot #88 exists to cover, so it must NOT be skipped.
    qa_test_data d = setup_test_data(
        desc, name, vlen_alloc, true, float_edge_cases, complex_edge_cases, mem_pool);
    if (!d.ok) {
        return false;
    }

    const bool s32f =
        (d.inputsc.size() == 1) && d.inputsc[0].is_float && !d.inputsc[0].is_complex;
    if (d.inputsc.size() != 0 && !s32f) {
        std::cerr << "reference mode: unsupported scalar signature for " << name
                  << std::endl;
        return false;
    }

    // Run every impl ONCE into its own buffer set, at the user vlen.
    for (size_t i = 0; i < d.arch_list.size(); i++) {
        const std::string arch = d.arch_list[i];
        switch (d.both_sigs.size()) {
        case 1:
            if (s32f)
                run_cast_test1_s32f((volk_fn_1arg_s32f)(manual_func),
                                    d.test_data[i],
                                    scalar.real(),
                                    vlen,
                                    1,
                                    arch);
            else
                run_cast_test1(
                    (volk_fn_1arg)(manual_func), d.test_data[i], vlen, 1, arch);
            break;
        case 2:
            if (s32f)
                run_cast_test2_s32f((volk_fn_2arg_s32f)(manual_func),
                                    d.test_data[i],
                                    scalar.real(),
                                    vlen,
                                    1,
                                    arch);
            else
                run_cast_test2(
                    (volk_fn_2arg)(manual_func), d.test_data[i], vlen, 1, arch);
            break;
        case 3:
            if (s32f)
                run_cast_test3_s32f((volk_fn_3arg_s32f)(manual_func),
                                    d.test_data[i],
                                    scalar.real(),
                                    vlen,
                                    1,
                                    arch);
            else
                run_cast_test3(
                    (volk_fn_3arg)(manual_func), d.test_data[i], vlen, 1, arch);
            break;
        default:
            std::cerr << "reference mode: unsupported arity for " << name << std::endl;
            return false;
        }
    }

    // Independent double-precision golden, computed from the PRISTINE inputs
    // (d.inbuffs is the original fill; impls run on per-arch memcpy copies, so
    // inbuffs is never mutated by a kernel).
    std::vector<const void*> oracle_in;
    for (size_t k = 0; k < d.inbuffs.size(); k++) {
        oracle_in.push_back(d.inbuffs[k]);
    }
    std::vector<void*> oracle_out;
    for (size_t j = 0; j < d.outputsig.size(); j++) {
        oracle_out.push_back(mem_pool.get_new(vlen_alloc * d.outputsig[j].size *
                                              (d.outputsig[j].is_complex ? 2 : 1)));
    }
    ref.fn(oracle_in, oracle_out, scalar, vlen);

    // Output type is a signature property — validate once up front and fail
    // LOUDLY (throw; the driver catches and reports a failure) on an unsupported
    // registration, rather than silently reporting ok.
    if (d.outputsig.empty()) {
        throw std::runtime_error("reference mode: kernel has no output: " + name);
    }
    for (size_t j = 0; j < d.outputsig.size(); j++) {
        if (!d.outputsig[j].is_float) {
            throw std::runtime_error("reference mode: non-float output unsupported for " +
                                     name);
        }
    }

    // Compare EVERY impl (generic included) against the golden.
    bool fail_global = false;
    for (size_t i = 0; i < d.arch_list.size(); i++) {
        bool arch_fail = false;
        double arch_max_err = 0.0;
        for (size_t j = 0; j < d.outputsig.size(); j++) {
            std::vector<unsigned int> fail_indices;
            double max_err = 0.0;
            bool fail;
            if (d.outputsig[j].is_complex) {
                fail = ccompare((float*)oracle_out[j],
                                (float*)d.test_data[i][j],
                                vlen,
                                ref.tol,
                                ref.absolute,
                                fail_indices,
                                max_err);
            } else {
                fail = fcompare((float*)oracle_out[j],
                                (float*)d.test_data[i][j],
                                vlen,
                                ref.tol,
                                ref.absolute,
                                fail_indices,
                                max_err);
            }
            arch_fail = arch_fail || fail;
            if (max_err > arch_max_err) {
                arch_max_err = max_err;
            }
        }
        volk_test_time_t result;
        result.name = d.arch_list[i];
        result.time = 0.0; // reference mode does not time the run; avoid an
                           // uninitialized read when the result is consumed
        result.units = "ref";
        result.pass = !arch_fail;
        results->back().results[d.arch_list[i]] = result;
        if (arch_fail) {
            fail_global = true;
            std::cout << "reference mismatch on arch: " << d.arch_list[i]
                      << " (max_err=" << arch_max_err << ")" << std::endl;
        }
    }
    return fail_global;
}

// #89: a single output buffer bracketed by sentinel guard regions, allocated in
// its OWN malloc (NOT the qa mem pool). The data region is EXACTLY data_bytes
// long -- no twiddle slack -- so an over-run cannot hide in allocator padding;
// it lands in the trailing guard (caught here) or, far enough past, in the ASan
// redzone of this allocation. The data start is aligned to volk_get_alignment()
// so `a_*` (aligned) impls stay valid; the leading guard is the bytes between the
// malloc base and that aligned start (the alignment slack is absorbed there, so
// the trailing guard is a known minimum width).
//
//   [ leading guard >= guard_bytes | data_bytes (aligned) | trailing guard ]
//
namespace {
class guarded_output_buffer
{
public:
    guarded_output_buffer(size_t data_bytes, size_t alignment, size_t guard_bytes)
        : data_bytes_(data_bytes)
    {
        // Over-allocate: leading guard + alignment slack + data + trailing guard.
        // align_up of (base + guard_bytes) consumes at most (alignment - 1) bytes,
        // so the leading guard is >= guard_bytes and the trailing guard >= guard_bytes.
        total_ = guard_bytes + alignment + data_bytes + guard_bytes;
        base_ = static_cast<uint8_t*>(std::malloc(total_));
        if (base_ == nullptr) {
            throw std::runtime_error("canary: guarded buffer malloc failed");
        }
        uintptr_t raw = reinterpret_cast<uintptr_t>(base_) + guard_bytes;
        uintptr_t aligned =
            (raw + (alignment - 1)) & ~static_cast<uintptr_t>(alignment - 1);
        data_ = reinterpret_cast<uint8_t*>(aligned);
        assert(data_ >= base_ + guard_bytes);
        assert(data_ + data_bytes_ + guard_bytes <= base_ + total_);
    }
    ~guarded_output_buffer() { std::free(base_); }
    guarded_output_buffer(const guarded_output_buffer&) = delete;
    guarded_output_buffer& operator=(const guarded_output_buffer&) = delete;

    void* data() const { return data_; }
    size_t data_bytes() const { return data_bytes_; }

    // Fill the WHOLE allocation (both guards + data) with one sentinel byte.
    void fill(uint8_t sentinel) { std::memset(base_, sentinel, total_); }

    // True if any guard byte differs from `sentinel` (an over/under-write).
    // `byte_off` is set to the offset relative to the data start: negative means
    // a write before index 0 (leading guard), >= data_bytes() means a write past
    // the end (trailing guard).
    bool guard_violation(uint8_t sentinel, ptrdiff_t& byte_off) const
    {
        for (uint8_t* p = base_; p < data_; ++p) {
            if (*p != sentinel) {
                byte_off = p - data_;
                return true;
            }
        }
        uint8_t* tail_start = data_ + data_bytes_;
        const uint8_t* tail_end = base_ + total_;
        for (uint8_t* p = tail_start; p < tail_end; ++p) {
            if (*p != sentinel) {
                byte_off = p - data_;
                return true;
            }
        }
        return false;
    }

private:
    uint8_t* base_;
    uint8_t* data_;
    size_t total_;
    size_t data_bytes_;
};
} // namespace

// #89: see the declaration in qa_utils.h. Runs every impl through its own
// guarded output buffers with the two-sentinel protocol and reports per-impl
// pass/fail into `results`. Supports the buffer signatures the qa harness can
// drive (1-4 buffers, with an optional real s32f or complex s32fc scalar);
// other shapes are skipped (reported as no failure) since the canary can only
// guard what it can call.
volk_canary_summary run_volk_canary_test(volk_func_desc_t desc,
                                         void (*manual_func)(),
                                         std::string name,
                                         lv_32fc_t scalar,
                                         unsigned int vlen,
                                         std::vector<volk_test_results_t>* results,
                                         const std::vector<float>& float_edge_cases,
                                         const std::vector<lv_32fc_t>& complex_edge_cases)
{
    volk_canary_summary summary;

    results->push_back(volk_test_results_t());
    results->back().name = name;
    results->back().vlen = vlen;
    results->back().iter = 2; // two runs per impl (S1 then S2)
    results->back().config_name = name;
    fmt::print("\nRUN_VOLK_CANARY_TEST: {}(vlen={})\n", name, vlen);

    volk_qa_aligned_mem_pool mem_pool;
    // Inputs are sized exactly `vlen` (no twiddle) and drawn from the pool; the
    // canary only guards the OUTPUT write side (input immutability / over-read is
    // #90, best-effort via ASan). benchmark_mode=true bypasses the ">=2 arch"
    // guard so single-impl kernels are still checked.
    qa_test_data d = setup_test_data(
        desc, name, vlen, true, float_edge_cases, complex_edge_cases, mem_pool);
    if (!d.ok) {
        return summary;
    }

    const bool s32f =
        (d.inputsc.size() == 1) && d.inputsc[0].is_float && !d.inputsc[0].is_complex;
    const bool s32fc =
        (d.inputsc.size() == 1) && d.inputsc[0].is_float && d.inputsc[0].is_complex;
    // These "cannot guard this kernel" cases leave summary.applied == false so the
    // driver reports the kernel as skipped (NOT ok). Diagnostics go to stdout so the
    // driver's per-(kernel,vlen) stdout muting suppresses them during the sweep,
    // consistent with how the guard/unwritten messages below are emitted.
    if (d.inputsc.size() != 0 && !s32f && !s32fc) {
        std::cout << "canary mode: unsupported scalar signature for " << name
                  << std::endl;
        return summary;
    }
    if (d.outputsig.empty()) {
        std::cout << "canary mode: kernel has no output buffer to guard: " << name
                  << std::endl;
        return summary;
    }
    if (d.both_sigs.size() > 4 || (d.both_sigs.size() == 4 && d.inputsc.size() != 0)) {
        std::cout << "canary mode: unsupported arity for " << name << std::endl;
        return summary;
    }

    const size_t alignment = volk_get_alignment();
    // Guard wide enough that a one-past write of any element type lands inside the
    // trailing guard (>= alignment + a few elements).
    const size_t guard_bytes = alignment + 64;

    const uint8_t S1 = 0xA5;
    const uint8_t S2 = 0x3C;

    for (size_t i = 0; i < d.arch_list.size(); i++) {
        const std::string arch = d.arch_list[i];
        summary.applied = true; // we are about to guard-check at least one impl
        summary.checked_impls.push_back(arch); // #92 triage detail

        // One guarded own-malloc buffer per output; inputs stay in the pool.
        std::vector<std::unique_ptr<guarded_output_buffer>> gbufs;
        gbufs.reserve(d.outputsig.size());
        for (size_t j = 0; j < d.outputsig.size(); j++) {
            const size_t out_bytes = static_cast<size_t>(vlen) * d.outputsig[j].size *
                                     (d.outputsig[j].is_complex ? 2 : 1);
            gbufs.push_back(std::make_unique<guarded_output_buffer>(
                out_bytes, alignment, guard_bytes));
        }

        // buffs layout matches setup_test_data: outputs first, then inputs.
        auto build_buffs = [&]() {
            std::vector<void*> buffs;
            for (size_t j = 0; j < d.outputsig.size(); j++) {
                buffs.push_back(gbufs[j]->data());
            }
            for (size_t k = 0; k < d.inputsig.size(); k++) {
                buffs.push_back(d.test_data[i][d.outputsig.size() + k]);
            }
            return buffs;
        };
        auto reload_inputs = [&]() {
            for (size_t k = 0; k < d.inputsig.size(); k++) {
                const size_t in_bytes = static_cast<size_t>(vlen) * d.inputsig[k].size *
                                        (d.inputsig[k].is_complex ? 2 : 1);
                memcpy(d.test_data[i][d.outputsig.size() + k], d.inbuffs[k], in_bytes);
            }
        };
        auto run_once = [&](std::vector<void*>& buffs) {
            switch (d.both_sigs.size()) {
            case 1:
                if (s32fc)
                    run_cast_test1_s32fc(
                        (volk_fn_1arg_s32fc)(manual_func), buffs, scalar, vlen, 1, arch);
                else if (s32f)
                    run_cast_test1_s32f((volk_fn_1arg_s32f)(manual_func),
                                        buffs,
                                        scalar.real(),
                                        vlen,
                                        1,
                                        arch);
                else
                    run_cast_test1((volk_fn_1arg)(manual_func), buffs, vlen, 1, arch);
                break;
            case 2:
                if (s32fc)
                    run_cast_test2_s32fc(
                        (volk_fn_2arg_s32fc)(manual_func), buffs, scalar, vlen, 1, arch);
                else if (s32f)
                    run_cast_test2_s32f((volk_fn_2arg_s32f)(manual_func),
                                        buffs,
                                        scalar.real(),
                                        vlen,
                                        1,
                                        arch);
                else
                    run_cast_test2((volk_fn_2arg)(manual_func), buffs, vlen, 1, arch);
                break;
            case 3:
                if (s32fc)
                    run_cast_test3_s32fc(
                        (volk_fn_3arg_s32fc)(manual_func), buffs, scalar, vlen, 1, arch);
                else if (s32f)
                    run_cast_test3_s32f((volk_fn_3arg_s32f)(manual_func),
                                        buffs,
                                        scalar.real(),
                                        vlen,
                                        1,
                                        arch);
                else
                    run_cast_test3((volk_fn_3arg)(manual_func), buffs, vlen, 1, arch);
                break;
            case 4:
                run_cast_test4((volk_fn_4arg)(manual_func), buffs, vlen, 1, arch);
                break;
            default:
                break;
            }
        };

        bool arch_guard = false;
        bool arch_unwritten = false;

        // ---- Run 1: prefill the whole guarded buffer with sentinel S1 ----
        for (auto& g : gbufs) {
            g->fill(S1);
        }
        reload_inputs();
        {
            std::vector<void*> buffs = build_buffs();
            run_once(buffs);
        }
        // Snapshot the data region after run 1 (before S2 overwrites it) and check
        // the guards for an over/under-write. Guard violations go to std::cerr (NOT
        // std::cout) so they survive the sweep's per-(kernel,vlen) stdout muting: a
        // touched guard is ALWAYS a real defect (unlike the unwritten check below,
        // which reduction kernels trip expectedly), so its arch/offset detail must
        // stay visible and actionable on the one run where it fires.
        std::vector<std::vector<uint8_t>> snap_s1(d.outputsig.size());
        for (size_t j = 0; j < gbufs.size(); j++) {
            ptrdiff_t off = 0;
            if (gbufs[j]->guard_violation(S1, off)) {
                arch_guard = true;
                std::cerr << name << ": canary guard touched on arch " << arch
                          << " (output " << j << ", byte offset " << off
                          << " relative to data start)\n";
            }
            const uint8_t* p = static_cast<const uint8_t*>(gbufs[j]->data());
            snap_s1[j].assign(p, p + gbufs[j]->data_bytes());
        }

        // ---- Run 2: prefill with the distinct sentinel S2 ----
        for (auto& g : gbufs) {
            g->fill(S2);
        }
        reload_inputs();
        {
            std::vector<void*> buffs = build_buffs();
            run_once(buffs);
        }
        for (size_t j = 0; j < gbufs.size(); j++) {
            ptrdiff_t off = 0;
            if (gbufs[j]->guard_violation(S2, off)) {
                arch_guard = true;
                std::cerr << name << ": canary guard touched on arch " << arch
                          << " (output " << j << ", byte offset " << off
                          << " relative to data start)\n";
            }
            // In-bounds-unwritten check: a byte the kernel wrote holds the same
            // (deterministic) value in both runs, so it cannot equal S1 after run 1
            // AND S2 after run 2. A byte that still equals both sentinels was never
            // written -- an in-bounds element left untouched (which ASan cannot see).
            const uint8_t* d2 = static_cast<const uint8_t*>(gbufs[j]->data());
            for (size_t b = 0; b < gbufs[j]->data_bytes(); b++) {
                if (snap_s1[j][b] == S1 && d2[b] == S2) {
                    arch_unwritten = true;
                    std::cout << name << ": canary found unwritten output byte on arch "
                              << arch << " (output " << j << ", byte offset " << b << ")"
                              << std::endl;
                    break;
                }
            }
        }

        volk_test_time_t result;
        result.name = arch;
        result.time = 0.0; // canary does not time the run; avoid an uninitialized read
        result.units = "canary"; // populate every field, even though unread on this path
        result.pass = !(arch_guard || arch_unwritten);
        results->back().results[arch] = result;
        summary.guard_violation = summary.guard_violation || arch_guard;
        summary.unwritten = summary.unwritten || arch_unwritten;
        // #92 triage detail: record offenders once per arch, from the per-arch
        // flags (never at the per-byte detection sites).
        if (arch_guard)
            summary.guard_impls.push_back(arch);
        if (arch_unwritten)
            summary.unwritten_impls.push_back(arch);
    }
    return summary;
}

volk_immutability_summary
run_volk_immutability_test(volk_func_desc_t desc,
                           void (*manual_func)(),
                           std::string name,
                           lv_32fc_t scalar,
                           unsigned int vlen,
                           std::vector<volk_test_results_t>* results,
                           const std::vector<float>& float_edge_cases,
                           const std::vector<lv_32fc_t>& complex_edge_cases)
{
    volk_immutability_summary summary;

    results->push_back(volk_test_results_t());
    results->back().name = name;
    results->back().vlen = vlen;
    results->back().iter = 1;
    results->back().config_name = name;
    fmt::print("\nRUN_VOLK_IMMUTABILITY_TEST: {}(vlen={})\n", name, vlen);

    volk_qa_aligned_mem_pool mem_pool;
    qa_test_data d = setup_test_data(
        desc, name, vlen, true, float_edge_cases, complex_edge_cases, mem_pool);
    if (!d.ok) {
        return summary;
    }

    const bool s32f =
        (d.inputsc.size() == 1) && d.inputsc[0].is_float && !d.inputsc[0].is_complex;
    const bool s32fc =
        (d.inputsc.size() == 1) && d.inputsc[0].is_float && d.inputsc[0].is_complex;
    // These "cannot check this kernel" cases leave summary.applied == false so the
    // driver reports the kernel as skipped (NOT ok). Diagnostics go to stdout so the
    // driver's per-(kernel,vlen) stdout muting suppresses them during the sweep.
    if (d.inputsc.size() != 0 && !s32f && !s32fc) {
        std::cout << "immutability mode: unsupported scalar signature for " << name
                  << std::endl;
        return summary;
    }
    // An in-place kernel (no separate output buffer) legitimately rewrites its single
    // buffer -- that is its contract, not a defect. Only out-of-place kernels have
    // input buffers the contract marks read-only, so immutability applies only when
    // there IS a distinct output buffer.
    if (d.outputsig.empty()) {
        std::cout << "immutability mode: in-place kernel (no separate output), input "
                     "write is by contract: "
                  << name << std::endl;
        return summary;
    }
    if (d.inputsig.empty()) {
        std::cout << "immutability mode: kernel has no input buffer to protect: " << name
                  << std::endl;
        return summary;
    }
    if (d.both_sigs.size() > 4 || (d.both_sigs.size() == 4 && d.inputsc.size() != 0)) {
        std::cout << "immutability mode: unsupported arity for " << name << std::endl;
        return summary;
    }

    // Invariant the per-input compare below relies on: scalars are erased from
    // inputsig into inputsc by setup_test_data, so d.inbuffs (the pristine pre-image,
    // built from the post-erasure non-scalar inputsig) is index-aligned with the
    // input buffers in d.test_data[i] at offset d.outputsig.size(). Assert it so a
    // future change to scalar handling fails loudly instead of comparing the wrong
    // buffers.
    assert(d.inbuffs.size() == d.inputsig.size());

    for (size_t i = 0; i < d.arch_list.size(); i++) {
        const std::string arch = d.arch_list[i];
        summary.applied = true;                // we are about to check at least one impl
        summary.checked_impls.push_back(arch); // #92 triage detail
        bool arch_mutated = false;

        // buffs layout matches setup_test_data: outputs first, then inputs. Inputs
        // are arch i's own pool copy (seeded from the pristine d.inbuffs[k], which the
        // kernel never receives).
        std::vector<void*> buffs;
        for (size_t j = 0; j < d.outputsig.size(); j++) {
            buffs.push_back(d.test_data[i][j]);
        }
        for (size_t k = 0; k < d.inputsig.size(); k++) {
            buffs.push_back(d.test_data[i][d.outputsig.size() + k]);
        }

        switch (d.both_sigs.size()) {
        case 1:
            if (s32fc)
                run_cast_test1_s32fc(
                    (volk_fn_1arg_s32fc)(manual_func), buffs, scalar, vlen, 1, arch);
            else if (s32f)
                run_cast_test1_s32f((volk_fn_1arg_s32f)(manual_func),
                                    buffs,
                                    scalar.real(),
                                    vlen,
                                    1,
                                    arch);
            else
                run_cast_test1((volk_fn_1arg)(manual_func), buffs, vlen, 1, arch);
            break;
        case 2:
            if (s32fc)
                run_cast_test2_s32fc(
                    (volk_fn_2arg_s32fc)(manual_func), buffs, scalar, vlen, 1, arch);
            else if (s32f)
                run_cast_test2_s32f((volk_fn_2arg_s32f)(manual_func),
                                    buffs,
                                    scalar.real(),
                                    vlen,
                                    1,
                                    arch);
            else
                run_cast_test2((volk_fn_2arg)(manual_func), buffs, vlen, 1, arch);
            break;
        case 3:
            if (s32fc)
                run_cast_test3_s32fc(
                    (volk_fn_3arg_s32fc)(manual_func), buffs, scalar, vlen, 1, arch);
            else if (s32f)
                run_cast_test3_s32f((volk_fn_3arg_s32f)(manual_func),
                                    buffs,
                                    scalar.real(),
                                    vlen,
                                    1,
                                    arch);
            else
                run_cast_test3((volk_fn_3arg)(manual_func), buffs, vlen, 1, arch);
            break;
        case 4:
            run_cast_test4((volk_fn_4arg)(manual_func), buffs, vlen, 1, arch);
            break;
        default:
            break;
        }

        // Post-run compare of each input against its pristine pre-image. Exact
        // std::memcmp (not a hash) so a mutation cannot hide behind a checksum
        // collision (acceptance #90-1); on mismatch, locate the first differing byte
        // for triage. The finding goes to std::cerr (NOT std::cout) so it survives the
        // sweep's per-(kernel,vlen) stdout muting -- unlike #89's canary, which has
        // expected reduction "partials" to suppress, immutability has no expected
        // findings, so a real mutation should always be loud and actionable.
        for (size_t k = 0; k < d.inputsig.size(); k++) {
            const size_t in_bytes = static_cast<size_t>(vlen) * d.inputsig[k].size *
                                    (d.inputsig[k].is_complex ? 2 : 1);
            const uint8_t* pre = static_cast<const uint8_t*>(d.inbuffs[k]);
            const uint8_t* post =
                static_cast<const uint8_t*>(d.test_data[i][d.outputsig.size() + k]);
            if (std::memcmp(pre, post, in_bytes) != 0) {
                summary.mutated = true;
                arch_mutated = true;
                size_t off = 0;
                while (off < in_bytes && pre[off] == post[off]) {
                    ++off;
                }
                std::cerr << name << ": input mutated on arch " << arch << " (input " << k
                          << ", byte offset " << off << ")\n";
            }
        }
        // #92 triage detail: record the offender once per arch, from the
        // per-arch flag (never at the per-input detection sites).
        if (arch_mutated)
            summary.mutated_impls.push_back(arch);
    }

    return summary;
}

// #91 is POSIX-only: MSVC has no sigaction/sigsetjmp/SIGBUS, and qa_utils.cc is
// compiled into volk_test_all too, so an unconditional use would break the
// Windows build of the DEFAULT qa suite. On _WIN32 the function compiles to an
// explicit "unsupported" skip (applied=false): the driver's negative control
// then fails closed with a clear stderr trail instead of the tree failing to
// build.
#ifdef _WIN32
volk_misaligned_summary
run_volk_misaligned_test(volk_func_desc_t /*desc*/,
                         void (* /*manual_func*/)(),
                         std::string name,
                         lv_32fc_t /*scalar*/,
                         float /*tol*/,
                         bool /*absolute_mode*/,
                         unsigned int /*vlen*/,
                         std::vector<volk_test_results_t>* /*results*/,
                         const std::vector<float>& /*float_edge_cases*/,
                         const std::vector<lv_32fc_t>& /*complex_edge_cases*/)
{
    std::cerr << "misaligned mode: unsupported on this platform (POSIX signal "
                 "isolation required): "
              << name << "\n";
    return volk_misaligned_summary();
}
#else

namespace {
// #91: SIGSEGV/SIGBUS/SIGILL isolation for misaligned-impl runs. An aligned SIMD
// load on a misaligned address raises a hardware signal, not a C++ exception, so
// catch(...) cannot trap it. sigsetjmp/siglongjmp is sound HERE because every
// exercised impl is a pure computational loop over harness-owned buffers (the
// driver skips puppets): no locks, allocations, or unwind-relevant state can be
// in flight at the faulting instruction, and the dispatch deliberately uses
// direct casted calls so no C++ frame with a non-trivial destructor sits between
// the sigsetjmp and the fault.
// Single-threaded qa binary by design: process-wide handlers + globals are safe
// (the faults are synchronous; there is no second thread to race).
sigjmp_buf g_misaligned_jmp;
volatile sig_atomic_t g_misaligned_sig = 0;
// Armed ONLY inside a sigsetjmp window. A fault while the handlers are installed
// but no window is open (buffer setup, compare, I/O) is a HARNESS bug, not a
// kernel defect -- jumping would misattribute it (or, before the first window,
// longjmp through a never-set jmp_buf: UB). Re-raise with default disposition so
// such a fault crashes loudly at its true location.
volatile sig_atomic_t g_misaligned_window_open = 0;
extern "C" void misaligned_fault_handler(int sig)
{
    if (!g_misaligned_window_open) {
        signal(sig, SIG_DFL);
        raise(sig);
        return;
    }
    g_misaligned_window_open = 0;
    g_misaligned_sig = sig;
    siglongjmp(g_misaligned_jmp, 1);
}
// Installs the handlers on construction, restores the previous ones on scope exit.
class scoped_fault_isolation
{
public:
    scoped_fault_isolation()
    {
        struct sigaction sa;
        memset(&sa, 0, sizeof(sa));
        sa.sa_handler = misaligned_fault_handler;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGSEGV, &sa, &old_segv_);
        sigaction(SIGBUS, &sa, &old_bus_);
        sigaction(SIGILL, &sa, &old_ill_);
    }
    ~scoped_fault_isolation()
    {
        sigaction(SIGSEGV, &old_segv_, nullptr);
        sigaction(SIGBUS, &old_bus_, nullptr);
        sigaction(SIGILL, &old_ill_, nullptr);
    }
    scoped_fault_isolation(const scoped_fault_isolation&) = delete;
    scoped_fault_isolation& operator=(const scoped_fault_isolation&) = delete;

private:
    struct sigaction old_segv_, old_bus_, old_ill_;
};

// #91: an own-malloc buffer whose data pointer is element-aligned but deliberately
// NOT volk_get_alignment()-aligned: one element past an alignment boundary --
// exactly the shape of a GNU Radio sub-buffer slice.
class misaligned_buffer
{
public:
    misaligned_buffer(size_t data_bytes, size_t alignment, size_t elem_bytes)
    {
        // Tail slack (8 elements) so a small over-WRITE -- the #89 defect class --
        // lands in our slack instead of corrupting malloc metadata mid-sweep; the
        // canary mode, not this one, is what detects such writes.
        raw_ = static_cast<uint8_t*>(malloc(data_bytes + alignment + elem_bytes * 9));
        if (raw_ == nullptr) {
            // Fail loud: continuing would fault OUTSIDE a sigsetjmp window and be
            // misattributed by the fault handler (or jump to a stale env).
            std::cerr << "misaligned_buffer: allocation failed\n";
            abort();
        }
        const uintptr_t base =
            (reinterpret_cast<uintptr_t>(raw_) + alignment - 1) & ~(alignment - 1);
        ptr_ = reinterpret_cast<void*>(base + elem_bytes);
        // Caller guarantees elem_bytes % alignment != 0 (the degenerate-platform
        // guard in run_volk_misaligned_test), so base+elem is off-boundary.
    }
    ~misaligned_buffer() { free(raw_); }
    misaligned_buffer(const misaligned_buffer&) = delete;
    misaligned_buffer& operator=(const misaligned_buffer&) = delete;
    void* data() const { return ptr_; }

private:
    uint8_t* raw_;
    void* ptr_;
};
} // namespace

volk_misaligned_summary
run_volk_misaligned_test(volk_func_desc_t desc,
                         void (*manual_func)(),
                         std::string name,
                         lv_32fc_t scalar,
                         float tol,
                         bool absolute_mode,
                         unsigned int vlen,
                         std::vector<volk_test_results_t>* results,
                         const std::vector<float>& float_edge_cases,
                         const std::vector<lv_32fc_t>& complex_edge_cases)
{
    volk_misaligned_summary summary;

    results->push_back(volk_test_results_t());
    results->back().name = name;
    results->back().vlen = vlen;
    results->back().iter = 1;
    results->back().config_name = name;
    fmt::print("\nRUN_VOLK_MISALIGNED_TEST: {}(vlen={})\n", name, vlen);

    volk_qa_aligned_mem_pool mem_pool;
    qa_test_data d = setup_test_data(
        desc, name, vlen, true, float_edge_cases, complex_edge_cases, mem_pool);
    if (!d.ok) {
        return summary;
    }

    const bool s32f =
        (d.inputsc.size() == 1) && d.inputsc[0].is_float && !d.inputsc[0].is_complex;
    const bool s32fc =
        (d.inputsc.size() == 1) && d.inputsc[0].is_float && d.inputsc[0].is_complex;
    // "Cannot check" cases leave summary.applied == false so the driver reports the
    // kernel as skipped (NOT ok). Diagnostics go to stdout so the driver's
    // per-(kernel,vlen) stdout muting suppresses them during the sweep.
    if (d.inputsc.size() != 0 && !s32f && !s32fc) {
        std::cout << "misaligned mode: unsupported scalar signature for " << name
                  << std::endl;
        return summary;
    }
    if (d.both_sigs.size() > 4 || (d.both_sigs.size() == 4 && d.inputsc.size() != 0)) {
        std::cout << "misaligned mode: unsupported arity for " << name << std::endl;
        return summary;
    }

    const size_t alignment = volk_get_alignment();
    // The kernel's own comparison parameters (mirrors run_volk_tests' derivation).
    const float tol_f = tol;
    const unsigned int tol_i = static_cast<unsigned int>(tol);

    // Degenerate-platform guard: if volk_get_alignment() does not exceed an element
    // size, an element offset can land back on an alignment boundary -- the
    // "misaligned" pointer would be aligned and the mode would silently test
    // nothing. Skip (applied=false -> reported skip, fail closed) instead.
    for (size_t j = 0; j < d.both_sigs.size(); j++) {
        const size_t elem = d.both_sigs[j].size * (d.both_sigs[j].is_complex ? 2 : 1);
        if (elem % alignment == 0) {
            std::cout << "misaligned mode: alignment " << alignment
                      << " too small to misalign element size " << elem << " for " << name
                      << std::endl;
            return summary;
        }
    }

    // Direct casted calls -- deliberately NOT the run_cast_testN wrappers, whose
    // by-value std::string parameter would sit on a frame the fault path longjmps
    // over (skipping a non-trivial destructor is UB). arch_cstr is materialized by
    // the caller before sigsetjmp. Single iteration. NOTE the s32fc forms take the
    // scalar BY POINTER (matching the volk_fn_*_s32fc typedefs).
    auto run_impl_direct = [&](std::vector<void*>& buffs, const char* arch_cstr) {
        switch (d.both_sigs.size()) {
        case 1:
            if (s32fc)
                ((volk_fn_1arg_s32fc)(manual_func))(buffs[0], &scalar, vlen, arch_cstr);
            else if (s32f)
                ((volk_fn_1arg_s32f)(manual_func))(
                    buffs[0], scalar.real(), vlen, arch_cstr);
            else
                ((volk_fn_1arg)(manual_func))(buffs[0], vlen, arch_cstr);
            break;
        case 2:
            if (s32fc)
                ((volk_fn_2arg_s32fc)(manual_func))(
                    buffs[0], buffs[1], &scalar, vlen, arch_cstr);
            else if (s32f)
                ((volk_fn_2arg_s32f)(manual_func))(
                    buffs[0], buffs[1], scalar.real(), vlen, arch_cstr);
            else
                ((volk_fn_2arg)(manual_func))(buffs[0], buffs[1], vlen, arch_cstr);
            break;
        case 3:
            if (s32fc)
                ((volk_fn_3arg_s32fc)(manual_func))(
                    buffs[0], buffs[1], buffs[2], &scalar, vlen, arch_cstr);
            else if (s32f)
                ((volk_fn_3arg_s32f)(manual_func))(
                    buffs[0], buffs[1], buffs[2], scalar.real(), vlen, arch_cstr);
            else
                ((volk_fn_3arg)(manual_func))(
                    buffs[0], buffs[1], buffs[2], vlen, arch_cstr);
            break;
        case 4:
            ((volk_fn_4arg)(manual_func))(
                buffs[0], buffs[1], buffs[2], buffs[3], vlen, arch_cstr);
            break;
        default:
            break;
        }
    };

    // Compare one both_sigs buffer against the generic aligned baseline, the same
    // per-type dispatch run_volk_tests uses. Comparing inputs as well as outputs is
    // what gives IN-PLACE kernels divergence coverage (their "output" IS the mutated
    // input buffer); for out-of-place kernels the input compare is a free no-op
    // (#90 guarantees inputs are unmutated, so both sides equal the pre-image).
    auto compare_buffer = [&](size_t j, void* expected, void* actual, double& max_err) {
        std::vector<unsigned int> fail_indices;
        max_err = 0.0;
        bool fail = false;
        const volk_type_t& t = d.both_sigs[j];
        const unsigned int n_ints = vlen * (t.is_complex ? 2 : 1);
        if (t.is_float) {
            if (t.size == 8) {
                if (t.is_complex)
                    fail = ccompare((double*)expected,
                                    (double*)actual,
                                    vlen,
                                    tol_f,
                                    absolute_mode,
                                    fail_indices,
                                    max_err);
                else
                    fail = fcompare((double*)expected,
                                    (double*)actual,
                                    vlen,
                                    tol_f,
                                    absolute_mode,
                                    fail_indices,
                                    max_err);
            } else {
                if (t.is_complex)
                    fail = ccompare((float*)expected,
                                    (float*)actual,
                                    vlen,
                                    tol_f,
                                    absolute_mode,
                                    fail_indices,
                                    max_err);
                else
                    fail = fcompare((float*)expected,
                                    (float*)actual,
                                    vlen,
                                    tol_f,
                                    absolute_mode,
                                    fail_indices,
                                    max_err);
            }
        } else {
            switch (t.size) {
            case 8:
                if (t.is_signed)
                    fail = icompare((int64_t*)expected,
                                    (int64_t*)actual,
                                    n_ints,
                                    tol_i,
                                    fail_indices,
                                    max_err);
                else
                    fail = icompare((uint64_t*)expected,
                                    (uint64_t*)actual,
                                    n_ints,
                                    tol_i,
                                    fail_indices,
                                    max_err);
                break;
            case 4:
                if (t.is_complex) {
                    // complex size-4 = pairs of 16-bit halves (run_volk_tests does
                    // the same): compare as int16/uint16.
                    if (t.is_signed)
                        fail = icompare((int16_t*)expected,
                                        (int16_t*)actual,
                                        n_ints,
                                        tol_i,
                                        fail_indices,
                                        max_err);
                    else
                        fail = icompare((uint16_t*)expected,
                                        (uint16_t*)actual,
                                        n_ints,
                                        tol_i,
                                        fail_indices,
                                        max_err);
                } else {
                    if (t.is_signed)
                        fail = icompare((int32_t*)expected,
                                        (int32_t*)actual,
                                        n_ints,
                                        tol_i,
                                        fail_indices,
                                        max_err);
                    else
                        fail = icompare((uint32_t*)expected,
                                        (uint32_t*)actual,
                                        n_ints,
                                        tol_i,
                                        fail_indices,
                                        max_err);
                }
                break;
            case 2:
                if (t.is_signed)
                    fail = icompare((int16_t*)expected,
                                    (int16_t*)actual,
                                    n_ints,
                                    tol_i,
                                    fail_indices,
                                    max_err);
                else
                    fail = icompare((uint16_t*)expected,
                                    (uint16_t*)actual,
                                    n_ints,
                                    tol_i,
                                    fail_indices,
                                    max_err);
                break;
            case 1:
                if (t.is_signed)
                    fail = icompare((int8_t*)expected,
                                    (int8_t*)actual,
                                    n_ints,
                                    tol_i,
                                    fail_indices,
                                    max_err);
                else
                    fail = icompare((uint8_t*)expected,
                                    (uint8_t*)actual,
                                    n_ints,
                                    tol_i,
                                    fail_indices,
                                    max_err);
                break;
            default:
                fail = true;
            }
        }
        return fail;
    };

    // Comparison design: SAME impl, aligned vs misaligned. Comparing a misaligned
    // u_impl against the generic baseline would conflate impl-vs-generic accuracy
    // (already #87's sweep, where edge-data precision differences legitimately
    // exceed tol for approximate kernels and large-vlen accumulations) with the
    // #91 question -- does MISALIGNMENT change this impl's output? Running the
    // same impl twice with identical inputs isolates alignment as the only
    // variable; the kernel's own tol absorbs legitimate reordering by impls that
    // peel to an alignment boundary.
    //
    // Output buffers on BOTH sides are zero-prefilled: reduction/index kernels
    // write only a fixed-size scalar into out[0..k), and output cardinality
    // cannot be derived from the signature (#89 lesson). With a common prefill,
    // never-written regions are 0 == 0 and only kernel-written elements compare.
    scoped_fault_isolation guard_signals;

    for (size_t i = 0; i < d.arch_list.size(); i++) {
        const std::string arch = d.arch_list[i];
        const size_t orig_idx = d.arch_to_orig_idx[arch];
        if (desc.impl_alignment[orig_idx]) {
            continue; // aligned-only impl: allowed to assume alignment, not under test
        }
        summary.applied = true;
        summary.checked_impls.push_back(arch); // #92 triage detail

        // ---- Reference run: this impl on its ALIGNED pool buffers ----
        std::vector<void*> abuffs;
        for (size_t j = 0; j < d.both_sigs.size(); j++) {
            abuffs.push_back(d.test_data[i][j]);
        }
        for (size_t j = 0; j < d.outputsig.size(); j++) {
            const size_t out_bytes = static_cast<size_t>(vlen) * d.outputsig[j].size *
                                     (d.outputsig[j].is_complex ? 2 : 1);
            memset(abuffs[j], 0, out_bytes);
        }
        const char* arch_cstr = arch.c_str(); // materialized BEFORE sigsetjmp
        g_misaligned_sig = 0;
        // setjmp-clobber discipline: NOTHING may be modified between a sigsetjmp
        // and its kernel call -- locals changed inside that window are
        // indeterminate after the longjmp. Each window holds exactly one call.
        if (sigsetjmp(g_misaligned_jmp, 1) == 0) {
            g_misaligned_window_open = 1; // volatile global: setjmp-clobber safe
            run_impl_direct(abuffs, arch_cstr);
            g_misaligned_window_open = 0;
        } else {
            // An "unaligned" impl crashing on ALIGNED buffers is a worse defect
            // than the one under test -- record it the same way and move on.
            summary.crashed = true;
            summary.crashed_impls.push_back(arch); // #92: once per arch (continues)
            std::cerr << name << ": impl crashed on ALIGNED buffers on arch " << arch
                      << " (signal " << static_cast<int>(g_misaligned_sig) << ", vlen "
                      << vlen << ")\n";
            continue;
        }

        // ---- Test run: the SAME impl on misaligned buffers, identical inputs ----
        std::vector<std::unique_ptr<misaligned_buffer>> mbufs;
        std::vector<void*> buffs;
        for (size_t j = 0; j < d.both_sigs.size(); j++) {
            const size_t elem = d.both_sigs[j].size * (d.both_sigs[j].is_complex ? 2 : 1);
            const size_t bytes = static_cast<size_t>(vlen) * elem;
            mbufs.push_back(std::make_unique<misaligned_buffer>(bytes, alignment, elem));
            buffs.push_back(mbufs.back()->data());
        }
        for (size_t j = 0; j < d.outputsig.size(); j++) {
            const size_t out_bytes = static_cast<size_t>(vlen) * d.outputsig[j].size *
                                     (d.outputsig[j].is_complex ? 2 : 1);
            memset(buffs[j], 0, out_bytes);
        }
        for (size_t k = 0; k < d.inputsig.size(); k++) {
            const size_t in_bytes = static_cast<size_t>(vlen) * d.inputsig[k].size *
                                    (d.inputsig[k].is_complex ? 2 : 1);
            memcpy(buffs[d.outputsig.size() + k], d.inbuffs[k], in_bytes);
        }

        g_misaligned_sig = 0;
        if (sigsetjmp(g_misaligned_jmp, 1) == 0) {
            g_misaligned_window_open = 1; // volatile global: setjmp-clobber safe
            run_impl_direct(buffs, arch_cstr);
            g_misaligned_window_open = 0;
        } else {
            summary.crashed = true;
            summary.crashed_impls.push_back(arch); // #92: once per arch (continues)
            std::cerr << name << ": impl crashed on misaligned buffers on arch " << arch
                      << " (signal " << static_cast<int>(g_misaligned_sig) << ", vlen "
                      << vlen << ")\n";
            continue; // recorded FAIL; next impl runs with its own fresh buffers
        }

        bool arch_diverged = false;
        for (size_t j = 0; j < d.both_sigs.size(); j++) {
            double max_err = 0.0;
            if (compare_buffer(j, abuffs[j], buffs[j], max_err)) {
                summary.diverged = true;
                arch_diverged = true;
                std::cerr << name << ": output diverged between aligned and "
                          << "misaligned runs on arch " << arch << " (buffer " << j
                          << ", vlen " << vlen << ", max_err " << max_err << ")\n";
            }
        }
        // #92 triage detail: record the offender once per arch, from the
        // per-arch flag (never at the per-buffer detection sites).
        if (arch_diverged)
            summary.diverged_impls.push_back(arch);
    }

    return summary;
}
#endif // !_WIN32 (POSIX signal isolation, #91)
