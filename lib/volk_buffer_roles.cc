/* -*- c++ -*- */
/*
 * Copyright 2026 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "volk_buffer_roles.h"

// Sparse, opt-in buffer-role registry (#161). See volk_buffer_roles.h for the
// schema and "how to add a kernel". Covers the QA-roster fixed-output class
// (#191): every kernel_tests.h kernel whose single-element-per-output-buffer
// contract the canary otherwise reports as `part` (the rest of the
// num_points-sized buffer is legitimately unwritten). Kernels outside the QA
// roster (the deprecated max-star family) are outside the sweep's reach and
// stay unregistered. Declaring output_elems=1 lets the canary confirm the
// contracted element was written and emit a hard ok/FAIL. Contracts are scoped
// to num_points >= 1, the canary's vlen range.
static const std::vector<volk_buffer_roles_entry> g_registry{
    // result = sum_i a[i] * b[i] -- single-element dot-product outputs.
    { "volk_32f_x2_dot_prod_32f", 1 },
    { "volk_32fc_x2_dot_prod_32fc", 1 },
    { "volk_32fc_x2_conjugate_dot_prod_32fc", 1 },
    { "volk_32fc_32f_dot_prod_32fc", 1 },
    { "volk_16i_32fc_dot_prod_32fc", 1 },
    { "volk_16ic_x2_dot_prod_16ic", 1 },
    { "volk_32f_x2_dot_prod_16i", 1 },
    { "volk_64f_x2_dot_prod_64f", 1 },
    // result = sum_i inputBuffer[i] -- a single accumulated element.
    { "volk_32f_accumulator_s32f", 1 },
    { "volk_32fc_accumulator_s32fc", 1 },
    // single index written to target[0].
    { "volk_32f_index_max_16u", 1 },
    { "volk_32f_index_max_32u", 1 },
    { "volk_32f_index_min_16u", 1 },
    { "volk_32f_index_min_32u", 1 },
    { "volk_32fc_index_max_16u", 1 },
    { "volk_32fc_index_max_32u", 1 },
    { "volk_32fc_index_min_16u", 1 },
    { "volk_32fc_index_min_32u", 1 },
    // single-element statistic outputs (stddev_and_mean writes 1 elem in EACH
    // of its two output buffers).
    { "volk_32f_s32f_stddev_32f", 1 },
    { "volk_32f_s32f_calc_spectral_noise_floor_32f", 1 },
    { "volk_32f_stddev_and_mean_32f_x2", 1 },
    // polynomial evaluation reduced to a single float in target[0].
    { "volk_32f_x3_sum_of_poly_32f", 1 },
};

const std::vector<volk_buffer_roles_entry>& volk_buffer_roles_registry()
{
    return g_registry;
}

const volk_buffer_roles_entry* volk_buffer_roles_lookup(const std::string& name)
{
    for (const volk_buffer_roles_entry& e : g_registry) {
        if (name == e.name) {
            return &e;
        }
    }
    return nullptr;
}
