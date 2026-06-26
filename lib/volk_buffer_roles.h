/* -*- c++ -*- */
/*
 * Copyright 2026 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*
 * Per-kernel buffer-role registry for the kernel-correctness harness (#161).
 * A sparse, opt-in table (modeled on lib/volk_reference.h's reference-oracle
 * registry) that declares the buffer semantics a sweep cannot infer from a
 * kernel's signature alone, so the sweep can replace a hedged verdict with a
 * hard pass/FAIL.
 *
 * Currently declares OUTPUT CARDINALITY for the #89 output-canary sweep: how
 * many elements the kernel is contracted to write per output buffer. The canary
 * cannot otherwise distinguish a map kernel that under-wrote its output (a
 * defect) from a reduction/index/accumulator kernel that writes a fixed-size
 * scalar and legitimately leaves the rest of the buffer untouched -- it reports
 * both as `part` (partial). With a declared cardinality the canary confirms the
 * contracted region was written: a registered reduction becomes a clean pass and
 * a registered map that under-writes becomes a hard FAIL.
 *
 * Input-mutability roles (which input buffers are read-only) for the #90
 * input-immutability sweep are a planned extension; see this issue's
 * potential-future-enhancements note.
 *
 * HOW TO ADD A KERNEL: append one volk_buffer_roles_entry to g_registry in
 * volk_buffer_roles.cc, keyed by the exact kernel name. Set output_elems to 0
 * for a map kernel (writes `vlen` elements per output) or to the fixed element
 * count for a reduction (e.g. 1 for a dot product / accumulator / single-index).
 * Unregistered kernels keep today's verdict exactly.
 *
 * NOTE: output_elems is a single value applied to EVERY output buffer of the
 * kernel; it assumes all outputs share one cardinality (true for the single-output
 * reductions registered today). A kernel with multiple outputs of differing
 * cardinalities would need a per-output vector -- a planned extension, not yet
 * needed.
 */

#ifndef INCLUDED_VOLK_BUFFER_ROLES_H
#define INCLUDED_VOLK_BUFFER_ROLES_H

#include <string>
#include <vector>

struct volk_buffer_roles_entry {
    const char* name;          // exact kernel name, e.g. "volk_32f_x2_dot_prod_32f"
    unsigned int output_elems; // 0 => map (writes `vlen` per output); N>0 => fixed
                               // (reduction/index/accumulator writes N elements)
};

// The opt-in registry. Sparse by design: only registered kernels get a promoted
// canary verdict; everything else keeps today's `part`/`ok`. Grows over time.
const std::vector<volk_buffer_roles_entry>& volk_buffer_roles_registry();

// Returns the entry for `name`, or nullptr if the kernel is not registered.
const volk_buffer_roles_entry* volk_buffer_roles_lookup(const std::string& name);

#endif /* INCLUDED_VOLK_BUFFER_ROLES_H */
