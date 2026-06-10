/* -*- c++ -*- */
/*
 * Copyright 2011 - 2020, 2022 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#ifndef VOLK_QA_UTILS_H
#define VOLK_QA_UTILS_H

#include <stdbool.h>          // for bool, false
#include <volk/volk.h>        // for volk_func_desc_t
#include <volk/volk_malloc.h> // for volk_malloc, volk_free
#include <cstdlib>            // for NULL
#include <cstring>            // for memset
#include <iosfwd>             // for std::ofstream (forward decl)
#include <map>                // for map
#include <string>             // for string, basic_string
#include <vector>             // for vector

#include "volk/volk_complex.h" // for lv_32fc_t

/************************************************
 * Warmup time configuration                    *
 ************************************************/
double volk_test_get_warmup_ms();
void volk_test_set_warmup_ms(double ms);
void volk_test_reset_warmup();

/************************************************
 * VOLK QA type definitions                     *
 ************************************************/
struct volk_type_t {
    bool is_float;
    bool is_scalar;
    bool is_signed;
    bool is_complex;
    int size;
    std::string str;
};

class volk_test_time_t
{
public:
    std::string name;
    double time;
    std::string units;
    bool pass;
};

class volk_test_results_t
{
public:
    std::string name;
    std::string config_name;
    unsigned int vlen = 0;
    unsigned int iter = 0;
    std::map<std::string, volk_test_time_t> results;
    std::string best_arch_a;
    std::string best_arch_u;
};

class volk_test_params_t
{
private:
    float _tol;
    lv_32fc_t _scalar;
    unsigned int _vlen;
    unsigned int _iter;
    unsigned int _trials = 1;
    bool _with_minmax = false;
    bool _benchmark_mode;
    bool _absolute_mode;
    std::string _kernel_regex;
    std::vector<float> _float_edge_cases;
    std::vector<lv_32fc_t> _complex_edge_cases;

public:
    // ctor
    volk_test_params_t(float tol,
                       lv_32fc_t scalar,
                       unsigned int vlen,
                       unsigned int iter,
                       bool benchmark_mode,
                       std::string kernel_regex)
        : _tol(tol),
          _scalar(scalar),
          _vlen(vlen),
          _iter(iter),
          _benchmark_mode(benchmark_mode),
          _absolute_mode(false),
          _kernel_regex(kernel_regex){};
    // setters
    void set_tol(float tol) { _tol = tol; };
    void set_scalar(lv_32fc_t scalar) { _scalar = scalar; };
    void set_vlen(unsigned int vlen) { _vlen = vlen; };
    void set_iter(unsigned int iter) { _iter = iter; };
    void set_trials(unsigned int trials) { _trials = trials; };
    void set_with_minmax(bool val) { _with_minmax = val; };
    void set_benchmark(bool benchmark) { _benchmark_mode = benchmark; };
    void set_regex(std::string regex) { _kernel_regex = regex; };
    void add_float_edge_cases(const std::vector<float>& edge_cases)
    {
        _float_edge_cases = edge_cases;
    };
    void add_complex_edge_cases(const std::vector<lv_32fc_t>& edge_cases)
    {
        _complex_edge_cases = edge_cases;
    };
    // getters
    float tol() { return _tol; };
    lv_32fc_t scalar() { return _scalar; };
    unsigned int vlen() { return _vlen; };
    unsigned int iter() { return _iter; };
    unsigned int trials() { return _trials; };
    bool with_minmax() { return _with_minmax; };
    bool benchmark_mode() { return _benchmark_mode; };
    bool absolute_mode() { return _absolute_mode; };
    std::string kernel_regex() { return _kernel_regex; };
    const std::vector<float>& float_edge_cases() const { return _float_edge_cases; };
    const std::vector<lv_32fc_t>& complex_edge_cases() const
    {
        return _complex_edge_cases;
    };
    volk_test_params_t make_absolute(float tol)
    {
        volk_test_params_t t(*this);
        t._tol = tol;
        t._absolute_mode = true;
        return t;
    }
    volk_test_params_t make_tol(float tol)
    {
        volk_test_params_t t(*this);
        t._tol = tol;
        return t;
    }
};

class volk_test_case_t
{
private:
    volk_func_desc_t _desc;
    void (*_kernel_ptr)();
    std::string _name;
    volk_test_params_t _test_parameters;
    std::string _puppet_master_name;

public:
    volk_func_desc_t desc() { return _desc; };
    void (*kernel_ptr())() { return _kernel_ptr; };
    std::string name() { return _name; };
    std::string puppet_master_name() { return _puppet_master_name; };
    volk_test_params_t test_parameters() { return _test_parameters; };
    // normal ctor
    volk_test_case_t(volk_func_desc_t desc,
                     void (*t_kernel_ptr)(),
                     std::string name,
                     volk_test_params_t test_parameters)
        : _desc(desc),
          _kernel_ptr(t_kernel_ptr),
          _name(name),
          _test_parameters(test_parameters),
          _puppet_master_name("NULL"){};
    // ctor for puppets
    volk_test_case_t(volk_func_desc_t desc,
                     void (*t_kernel_ptr)(),
                     std::string name,
                     std::string puppet_master_name,
                     volk_test_params_t test_parameters)
        : _desc(desc),
          _kernel_ptr(t_kernel_ptr),
          _name(name),
          _test_parameters(test_parameters),
          _puppet_master_name(puppet_master_name){};
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

/************************************************
 * VOLK QA functions                            *
 ************************************************/
volk_type_t volk_type_from_string(std::string);

std::vector<std::string> get_arch_list(volk_func_desc_t desc);

void get_signatures_from_name(std::vector<volk_type_t>& inputsig,
                              std::vector<volk_type_t>& outputsig,
                              std::string name);

float uniform(void);
void random_floats(float* buf, unsigned n);

void load_random_data(
    void* data,
    volk_type_t type,
    unsigned int n,
    const std::vector<float>& float_edge_cases = std::vector<float>(),
    const std::vector<lv_32fc_t>& complex_edge_cases = std::vector<lv_32fc_t>());

bool run_volk_tests(volk_func_desc_t,
                    void (*)(),
                    std::string,
                    volk_test_params_t,
                    std::vector<volk_test_results_t>* results = NULL,
                    std::string puppet_master_name = "NULL",
                    std::ofstream* csv_out = nullptr);

bool run_volk_tests(
    volk_func_desc_t,
    void (*)(),
    std::string,
    float,
    lv_32fc_t,
    unsigned int,
    unsigned int,
    std::vector<volk_test_results_t>* results = NULL,
    std::string puppet_master_name = "NULL",
    bool absolute_mode = false,
    bool benchmark_mode = false,
    const std::vector<float>& float_edge_cases = std::vector<float>(),
    const std::vector<lv_32fc_t>& complex_edge_cases = std::vector<lv_32fc_t>(),
    unsigned int trials = 1,
    bool with_minmax = false,
    std::ofstream* csv_out = nullptr);

// #88: compare every impl against an independent double-precision reference oracle
// (catches defects all impls share). Defined in qa_utils.cc; oracle from the
// volk_reference registry. Returns true if any impl diverges past the entry tol.
struct volk_reference_entry;
bool run_volk_reference_test(
    volk_func_desc_t,
    void (*)(),
    std::string,
    const volk_reference_entry&,
    lv_32fc_t,
    unsigned int,
    std::vector<volk_test_results_t>* results = NULL,
    const std::vector<float>& float_edge_cases = std::vector<float>(),
    const std::vector<lv_32fc_t>& complex_edge_cases = std::vector<lv_32fc_t>());

// #89: outcome of run_volk_canary_test, split so the driver can treat the two
// defect classes differently. A guard violation (a write past the end or before
// index 0 of a buffer) is always a defect, for any kernel. An unwritten in-bounds
// element is a defect for a MAP kernel, but expected for a reduction/index kernel
// whose output is a fixed-size scalar rather than num_points elements -- which the
// signature alone cannot distinguish -- so the driver surfaces it for triage
// rather than hard-failing.
struct volk_canary_summary {
    bool guard_violation = false; // over/under-write past a buffer (always a defect)
    bool unwritten = false;       // an in-bounds output element never written
    bool applied = false;         // false => the canary could not guard this kernel
                                  // (no output buffer / unsupported signature): the
                                  // driver reports such kernels as skipped, not ok
};

// #89: output-buffer canary. Allocates each output buffer with leading/trailing
// sentinel guard regions in its OWN malloc (bypassing the qa mem pool, so the
// data region is exactly num_points elements with no slack to hide an over-run,
// and so ASan redzones bracket it). For every impl, runs twice with two distinct
// sentinels: a touched guard flags an over/under-write; an in-bounds byte left at
// both sentinels across the two runs flags a never-written element. This canary is
// allocator-independent and authoritative; ASan is best-effort double coverage.
// Toggle is in the driver; run_volk_tests/default qa are untouched.
volk_canary_summary run_volk_canary_test(
    volk_func_desc_t,
    void (*)(),
    std::string,
    lv_32fc_t,
    unsigned int,
    std::vector<volk_test_results_t>* results = NULL,
    const std::vector<float>& float_edge_cases = std::vector<float>(),
    const std::vector<lv_32fc_t>& complex_edge_cases = std::vector<lv_32fc_t>());

// #90: outcome of run_volk_immutability_test. A kernel that writes any byte of an
// input buffer (which an out-of-place kernel's contract forbids) sets `mutated`.
// `applied` is false when the kernel has no separate input to protect: an in-place
// kernel (no output buffer -> its single buffer is the input it legitimately
// rewrites) or an unsupported signature -- such kernels are reported skipped, not
// ok, so an unchecked kernel never masquerades as covered.
struct volk_immutability_summary {
    bool mutated = false; // an input buffer differed after the call (always a defect)
    bool applied = false; // false => no separate input buffer to protect / unsupported
};

// #90: input-immutability canary. For every impl, byte-compares each input buffer
// after the call against its pristine pre-image (qa_test_data::inbuffs, a separate
// allocation the kernel never receives). An exact compare -- not a hash -- so a
// mutation cannot hide behind a checksum collision. Delivers the one defensible
// value of the abandoned const-input-signature sweep without touching any
// signature. Toggle is in the driver; run_volk_tests / default qa are untouched.
volk_immutability_summary run_volk_immutability_test(
    volk_func_desc_t,
    void (*)(),
    std::string,
    lv_32fc_t,
    unsigned int,
    std::vector<volk_test_results_t>* results = NULL,
    const std::vector<float>& float_edge_cases = std::vector<float>(),
    const std::vector<lv_32fc_t>& complex_edge_cases = std::vector<lv_32fc_t>());

// #91: outcome of run_volk_misaligned_test. For every impl the dispatch metadata
// marks unaligned (impl_alignment == false), the harness runs it on deliberately
// misaligned (element-aligned, non-volk_get_alignment()-aligned) buffers.
// `crashed` => the impl raised SIGSEGV/SIGBUS/SIGILL (e.g. movaps on a misaligned
// address) -- trapped and recorded, the run continues. `diverged` => the SAME
// impl produced different output on misaligned vs aligned buffers with identical
// inputs (beyond the kernel's own tol -- alignment is the only variable; the
// impl-vs-generic accuracy question belongs to the #87 sweep). `applied` is false
// when no impl could be exercised (unsupported signature / degenerate alignment)
// -- reported skip, never ok.
struct volk_misaligned_summary {
    bool crashed = false;
    bool diverged = false;
    bool applied = false;
};

// #91: misaligned-run check. The fault path (a hardware signal, not a C++
// exception) is trapped with a scoped SIGSEGV/SIGBUS/SIGILL handler +
// sigsetjmp/siglongjmp so one crashing impl becomes one recorded FAIL while the
// run continues. Toggle is in the driver; run_volk_tests / default qa untouched.
// tol/absolute_mode are the KERNEL'S own comparison parameters (from its
// volk_test_params_t) -- approximate kernels (log2, expfast, tan, ...) carry
// looser tolerances than the harness default, and using anything else
// false-positives them.
volk_misaligned_summary run_volk_misaligned_test(
    volk_func_desc_t,
    void (*)(),
    std::string,
    lv_32fc_t,
    float tol,
    bool absolute_mode,
    unsigned int,
    std::vector<volk_test_results_t>* results = NULL,
    const std::vector<float>& float_edge_cases = std::vector<float>(),
    const std::vector<lv_32fc_t>& complex_edge_cases = std::vector<lv_32fc_t>());

#define VOLK_PROFILE(func, test_params, results) \
    run_volk_tests(func##_get_func_desc(),       \
                   (void (*)())func##_manual,    \
                   std::string(#func),           \
                   test_params,                  \
                   results,                      \
                   "NULL")
#define VOLK_PUPPET_PROFILE(func, puppet_master_func, test_params, results) \
    run_volk_tests(func##_get_func_desc(),                                  \
                   (void (*)())func##_manual,                               \
                   std::string(#func),                                      \
                   test_params,                                             \
                   results,                                                 \
                   std::string(#puppet_master_func))
typedef void (*volk_fn_1arg)(void*,
                             unsigned int,
                             const char*); // one input, operate in place
typedef void (*volk_fn_2arg)(void*, void*, unsigned int, const char*);
typedef void (*volk_fn_3arg)(void*, void*, void*, unsigned int, const char*);
typedef void (*volk_fn_4arg)(void*, void*, void*, void*, unsigned int, const char*);
typedef void (*volk_fn_1arg_s32f)(
    void*, float, unsigned int, const char*); // one input vector, one scalar float input
typedef void (*volk_fn_2arg_s32f)(void*, void*, float, unsigned int, const char*);
typedef void (*volk_fn_3arg_s32f)(void*, void*, void*, float, unsigned int, const char*);
typedef void (*volk_fn_1arg_s32fc)(
    void*,
    lv_32fc_t*,
    unsigned int,
    const char*); // one input vector, one scalar float input
typedef void (*volk_fn_2arg_s32fc)(void*, void*, lv_32fc_t*, unsigned int, const char*);
typedef void (*volk_fn_3arg_s32fc)(
    void*, void*, void*, lv_32fc_t*, unsigned int, const char*);

#endif // VOLK_QA_UTILS_H
