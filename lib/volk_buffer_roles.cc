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
// schema and "how to add a kernel". Proof-of-concept entries: reduction kernels
// whose single-element output the canary otherwise reports as `part` (the rest of
// the num_points-sized buffer is legitimately unwritten). Declaring output_elems=1
// lets the canary confirm the one contracted element was written and emit a clean
// pass. Grows over time -- comprehensive characterization is incremental follow-up.
static const std::vector<volk_buffer_roles_entry> g_registry{
    // result = sum_i input[i] * taps[i] -- writes a single float.
    { "volk_32f_x2_dot_prod_32f", 1 },
    // result = sum_i inputBuffer[i] -- writes a single float.
    { "volk_32f_accumulator_s32f", 1 },
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
